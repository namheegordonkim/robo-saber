import glob
from contextlib import contextmanager
from queue import Queue
from threading import Thread, Semaphore
from typing import Tuple

import numpy as np
import torch
from datasets import Dataset as DDataset, load_dataset
from scipy.spatial.transform import Rotation
from torch.distributions import Categorical
from torch.utils.data import Dataset, Sampler, IterableDataset

from beaty_common.bsmg_xror_utils import open_xror, open_bsmg, quat_to_expm, load_cbo_and_3p, open_beatmap_from_unpacked_xror
from proj_utils.dirs import proj_dir
from beaty_common.torch_nets2 import CondTransformerGSVAE, TransformerGSVAE
from xror.xror import XROR

placeholder_3p = np.loadtxt(f"data/placeholder_3p.txt", dtype=np.float32)
placeholder_3p_sixd = np.loadtxt(f"data/placeholder_3p_sixd.txt", dtype=np.float32)

# mp.set_start_method("spawn", force=True)


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        return np.clip((arr - self.mean) / np.sqrt(self.var + self.epsilon), -1000, 1000)

    def unnormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * np.sqrt(self.var + self.epsilon) + self.mean


class DatasetMaker:
    """
    Given Numpy arrays, prepares HuggingFace training / validation data
    """

    def __init__(self):
        # To be dynamically populated during `setup()`
        self.x_tokenizer = None
        self.x_scaler = None

        self.x_vocab_size = 2000
        self.x_code_length = None

        self.y_tokenizer = None
        self.y_scaler = None

        self.y_vocab_size = 2000
        self.y_code_length = None

    def setup(self, setup_x: np.ndarray, setup_y: np.ndarray):
        self.x_scaler = RunningMeanStd(shape=(setup_x.shape[1:]))
        self.y_scaler = RunningMeanStd(shape=(setup_y.shape[1:]))

        self.x_tokenizer = BucketizeTokenizer(setup_x.shape[-1], self.x_vocab_size)
        self.y_tokenizer = BucketizeTokenizer(setup_y.shape[-1], self.y_vocab_size)

    def make(self, source_x: np.ndarray, source_y: np.ndarray) -> DDataset:
        num_patches = 4  # You can change this to the desired number of patches
        n_batches = source_x.shape[0]
        patch_size = 28 // int(num_patches**0.5)
        n_idxs_per_axis = 28 // patch_size
        x_patches = np.empty((n_batches, num_patches, patch_size, patch_size))

        # Extract patches
        patch_idx = 0
        for i in range(n_idxs_per_axis):
            for j in range(n_idxs_per_axis):
                patch = source_x[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                x_patches[:, patch_idx] = patch
                patch_idx += 1

        x_patches_tensor = torch.as_tensor(x_patches, dtype=torch.float)
        x_patches_tensor = x_patches_tensor / 255 * 2 - 1

        x_patches_encoded, x_patches_quantized = self.x_tokenizer.encode(x_patches_tensor.reshape(-1, patch_size**2), device="cpu")
        x_patches_encoded = x_patches_encoded.reshape(n_batches, -1)
        x_patches_quantized = x_patches_quantized.reshape(n_batches, num_patches, patch_size, patch_size)

        # Get quantization error
        q_deltas = torch.abs(x_patches_quantized.detach().cpu() - x_patches_tensor).detach().cpu().numpy()
        q_mean = q_deltas.mean()
        q_std = q_deltas.std()
        q_max = q_deltas.max()

        print(f"Mean x quantization error : {q_mean:.3f}")
        print(f"Std x quantization error : {q_std:.3f}")
        print(f"Max x quantization error : {q_max:.3f}")

        # + 12 and + 2 because:
        # 12 total output vocabulary; 0 for padding, 1 for <EOS>, and 2-11 for the 10 digits (0-9)
        # To make the input vocabulary non-overlapping with the output vocabulary, we start input vocabulary at 12
        input_ids_np = np.concatenate(
            [
                x_patches_encoded.cpu().detach().numpy().reshape(x_patches_encoded.shape[0], -1) + 12,
                source_y[:, None] + 2,
                np.ones((x_patches_encoded.shape[0], 1), dtype=int),
            ],
            axis=-1,
        )
        labels_np = input_ids_np * 1
        labels_np[:, :-2] = -100  # "unused" for HuggingFace Llama
        attention_mask_np = np.ones_like(input_ids_np)

        input_ids = input_ids_np.tolist()
        labels = labels_np.tolist()
        attention_mask = attention_mask_np.tolist()

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        dataset = DDataset.from_dict(data_dict)
        return dataset


class ThroughDataset(Dataset):
    """
    Sacrifice some readability to make life easier.
    Whatever input array/argument tensor provided will be the output for dataset.
    """

    def __init__(self, *args):
        self.args = args
        for a1, a2 in zip(self.args, self.args[1:]):
            assert a1.shape[0] == a2.shape[0]

    def __getitem__(self, index):
        indexed = tuple(a[index] for a in self.args)
        return indexed

    def __len__(self):
        return self.args[0].shape[0]


class RepeatSampler(Sampler):
    """Sampler that repeats samples to match a desired batch size."""

    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        indices = torch.randint(0, len(self.data_source), (self.batch_size,), dtype=torch.long)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


def expm_to_quat(expm: np.ndarray, eps: float = 1e-8):
    """ """
    half_angle = np.linalg.norm(expm, axis=-1)[..., None]
    # if half_angle < eps:
    #     quat = np.concatenate([expm, np.ones_like(half_angle)], axis=-1)
    #     quat /= np.linalg.norm(quat, axis=-1)[..., None]
    #     return quat
    c = np.cos(half_angle)
    s = np.sin(half_angle)
    quat = np.where(
        half_angle < eps,
        np.concatenate([expm, np.ones_like(half_angle)], axis=-1),
        np.concatenate([expm * s / half_angle, c], axis=-1),
    )
    return quat


def quat_to_expm_torch(quat: torch.Tensor, eps: float = 1e-8):
    """
    Quaternion is (x, y, z, w)
    """
    im = quat[..., :3]
    im_norm = torch.norm(im, dim=-1)
    half_angle = torch.arctan2(im_norm, quat[..., 3])
    expm = torch.where(
        im_norm[..., None] < eps,
        im,
        half_angle[..., None] * (im / im_norm[..., None]),
    )
    return expm


def expm_to_quat_torch(expm: torch.Tensor, eps: float = 1e-8):
    """ """
    half_angle = torch.linalg.norm(expm, dim=-1)[..., None]
    c = torch.cos(half_angle)
    s = torch.sin(half_angle)
    quat = torch.where(
        half_angle < eps,
        torch.cat([expm, torch.ones_like(half_angle)], dim=-1),
        torch.cat([expm * s / half_angle, c], dim=-1),
    )
    return quat


def quat_rotate(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[..., :-1]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[..., -1:] * uv + uuv)).view(original_shape)


def get_pos_expm_with_correction(frames_np):
    timestamps = frames_np[1:, ..., 0]
    part_data = frames_np[1:, ..., 1:]
    part_data = part_data.reshape(part_data.shape[0], 3, -1)
    my_pos = part_data[..., :3]
    my_quat = part_data[..., 3:]
    my_rot = Rotation.from_quat(my_quat.reshape(-1, 4))
    correction = Rotation.from_euler("X", 0.5 * np.pi)
    correction_matrix = correction.as_matrix()
    my_pos = np.einsum("nkj,ij->nki", my_pos, correction_matrix)
    my_rot = correction * my_rot
    correction = Rotation.from_euler("Z", 0.5 * np.pi)
    correction_matrix = correction.as_matrix()
    my_pos = np.einsum("nkj,ij->nki", my_pos, correction_matrix)
    my_rot = correction * my_rot
    my_rotmat = my_rot.as_matrix().reshape((my_quat.shape[0], 3, 3, 3))
    my_rotmat[..., [0, 1, 2]] = my_rotmat[..., [2, 0, 1]]
    my_rotmat[:, [0, 1, 2]] = my_rotmat[:, [0, 2, 1]]
    my_rotmat[:, 1:, ..., [0, 1, 2]] = my_rotmat[:, 1:, ..., [0, 2, 1]]
    my_rotmat[:, 1:, ..., 2] *= -1
    my_rotmat[:, 1, ..., [1, 2]] *= -1

    # left_correction_rotmat = Rotation.from_euler("X", -0.5 * np.pi)
    # right_correction_rotmat = Rotation.from_euler("X", 0.5 * np.pi)
    # my_rotmat[:, 1] = np.einsum("ik, nkj -> nij", left_correction_rotmat.as_matrix(), my_rotmat[:, 1])
    # my_rotmat[:, 2] = np.einsum("ik, nkj -> nij", right_correction_rotmat.as_matrix(), my_rotmat[:, 2])
    # my_rotmat[:, 1, ..., [1, 2]] *= -1
    my_rot = Rotation.from_matrix(my_rotmat.reshape((-1, 3, 3)))
    my_quat = my_rot.as_quat().reshape(*my_quat.shape)
    my_sixd = my_rot.as_matrix()[..., :2].reshape(*my_quat.shape[:-1], 6)
    my_pos[..., [0, 1]] -= my_pos[0][0, [0, 1]]
    my_pos[:, [0, 1, 2]] = my_pos[:, [0, 2, 1]]
    my_pos_sixd = np.concatenate([my_pos, my_sixd], axis=-1)
    my_pos_sixd = my_pos_sixd.reshape(-1, 27)
    my_expm = quat_to_expm(my_quat)
    my_pos_expm = np.concatenate([my_pos, my_expm], axis=-1)
    my_pos_expm = my_pos_expm.reshape(-1, 18)
    return my_pos_expm, my_pos_sixd, timestamps


def unity_to_zup(xyz: np.ndarray, quat: np.ndarray):
    my_pos = xyz
    my_pos[..., [0, 1, 2]] = my_pos[..., [2, 0, 1]]
    my_pos[..., 1] *= -1

    my_quat = quat
    my_quat[..., [0, 1, 2, 3]] = my_quat[..., [2, 0, 1, 3]]
    my_quat[..., [0, 2]] *= -1

    return my_pos, my_quat


def sample_segments(xs, ys, lengths, n_samples, segment_length, device):
    idxs = (torch.rand(n_samples) * lengths.shape[0]).to(dtype=torch.long, device=device)
    t_start = (torch.rand(n_samples, device=device) * (lengths[idxs] - segment_length)).to(dtype=torch.long)
    segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
    segment_its = torch.cat(
        [
            idxs[:, None, None].repeat_interleave(segment_length, 1),
            segment_ts[:, :, None],
        ],
        dim=2,
    )  # (n_samples
    # segment_xs = torch.take_along_dim(xs[idxs], segment_its[:, :, :, None], dim=1)
    # segment_ys = torch.take_along_dim(ys[idxs], segment_ts[:, :, None], dim=1)
    segment_xs = xs[segment_its[..., 0], segment_its[..., 1]]
    segment_ys = ys[segment_its[..., 0], segment_its[..., 1]]
    return segment_xs, segment_ys, segment_ts


@contextmanager
def maybe_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def collect_rollout(
    pred_net: CondTransformerGSVAE,
    gsvae_net: TransformerGSVAE,
    pred_ema: CondTransformerGSVAE,
    gsvae_ema: TransformerGSVAE,
    optim: torch.optim.Optimizer,
    game_notes: torch.Tensor,
    game_bombs: torch.Tensor,
    game_obstacles: torch.Tensor,
    game_history: torch.Tensor,
    game_frames: torch.Tensor,
    game_3p: torch.Tensor,
    playstyle_notes: torch.Tensor,
    playstyle_bombs: torch.Tensor,
    playstyle_obstacles: torch.Tensor,
    playstyle_history: torch.Tensor,
    playstyle_frames: torch.Tensor,
    playstyle_3p: torch.Tensor,
    history_len: int,
    chunk_length: int,
    matching_weight: float,
    train_yes: bool,
    matching_loss_type: str,
    n_cands: int,
):
    segment_ys = game_3p
    segment_ts = game_frames
    stride = gsvae_net.stride
    strided_chunk_length = chunk_length // stride

    first_history_idxs = np.arange(history_len)
    # w_ins = [segment_ys[:, [t]] for t in first_history_idxs]
    # yes = segment_ts[..., :history_len] < history_len * stride
    # for j in range(history_len):
    #     w_ins[j][yes[..., j]] = torch.as_tensor(placeholder_3p_sixd, device=w_ins[-1].device)
    losses = []
    pred_losses = []
    recon_losses = []
    matching_losses = []
    # for t in range(history_len, len(game_segments) - strided_chunk_length + 1):
    #     w_in = torch.cat(w_ins[-history_len:], dim=1)  # history
    #     x_in = game_segments[:, [t]]
    #     y_gt = segment_ys[:, t : t + strided_chunk_length]

    with maybe_no_grad(not train_yes):
        # Train on the fly here
        z_ref, k, z_soft_ref, z_hard_ref, y_ref = gsvae_net.forward(
            game_3p * 1,
            n=n_cands,
        )
        z, k, z_soft, z_hard, _ = pred_net.forward(
            game_notes * 1,
            game_bombs * 1,
            game_obstacles * 1,
            game_history * 1,
            playstyle_notes * 1,
            playstyle_bombs * 1,
            playstyle_obstacles * 1,
            playstyle_history * 1,
            playstyle_3p * 1,
            n=n_cands,
        )
        recon_loss = torch.nn.functional.mse_loss(y_ref, game_3p[:, None], reduction="none").mean((-1, -2, -3))

        sentence_length = pred_net.sentence_length
        vocab_size = pred_net.vocab_size
        z_ref = z_ref.view(-1, sentence_length, vocab_size)
        z = z.view(-1, sentence_length, vocab_size)

        if matching_loss_type == "jsd":
            # Jensen-Shannon
            dist_ref = Categorical(logits=z_ref)
            dist = Categorical(logits=z)
            jsd_m = Categorical(probs=0.5 * (dist_ref.probs + dist.probs))
            matching_loss_left = torch.distributions.kl.kl_divergence(dist_ref, jsd_m).mean(-1)
            matching_loss_right = torch.distributions.kl.kl_divergence(dist, jsd_m).mean(-1)
            matching_loss = 0.5 * (matching_loss_left + matching_loss_right)
            matching_loss = torch.clamp(matching_loss, 0, np.log(2))
            loss = recon_loss + matching_loss * matching_weight
        else:
            matching_loss = torch.nn.functional.mse_loss(z_hard, z_hard_ref, reduction="none").mean((-1, -2))
            loss = recon_loss + matching_loss

        if train_yes:
            optim.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(pred_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(gsvae_net.parameters(), 1.0)
            optim.step()

            # Update EMA.
            ema_halflife_kimg = 500
            ema_halflife_nimg = ema_halflife_kimg * 1000
            ema_beta = 0.5 ** (game_3p.shape[0] / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(pred_ema.parameters(), pred_net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            for p_ema, p_net in zip(gsvae_ema.parameters(), gsvae_net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        with torch.no_grad():
            z, k, z_soft, z_hard, _ = pred_ema.forward(
                game_notes * 1,
                game_bombs * 1,
                game_obstacles * 1,
                game_history * 1,
                playstyle_notes * 1,
                playstyle_bombs * 1,
                playstyle_obstacles * 1,
                playstyle_history * 1,
                playstyle_3p * 1,
                n=1,
            )
            y_out = gsvae_ema.decode(z_hard * 1)
            # if train_yes:
            #     # Teacher forcing
            #     new_w_in = y_gt[:, [0]] * 1
            # else:
            #     new_w_in = y_out[:, 0, [0]] * 1
            pred_loss = torch.nn.functional.mse_loss(y_out[:, 0], game_3p, reduction="none").mean((-1, -2))
            # w_ins.append(new_w_in)

        losses.append(loss.detach())
        pred_losses.append(pred_loss.detach())
        recon_losses.append(recon_loss.detach())
        matching_losses.append(matching_loss.detach())

    losses = torch.stack(losses, -1)
    pred_losses = torch.stack(pred_losses, -1)
    recon_losses = torch.stack(recon_losses, -1)
    matching_losses = torch.stack(matching_losses, -1)

    return losses, pred_losses, recon_losses, matching_losses


def get_rollouts(model, ref_model, game_segments, movement_segments, history_len, chunk_length):
    model.eval()
    ref_model.eval()

    segment_ys = movement_segments.three_p
    segment_ts = game_segments.frames

    y_outs = [segment_ys[:, [t]] for t in np.arange(history_len)]
    yes = segment_ts[..., :history_len] < history_len
    for j in range(history_len):
        y_outs[j][yes[..., j]] *= torch.nan
    rollout_xins = []
    rollout_wins = []
    rollout_ygts = []
    rollout_youts = []
    with torch.no_grad():
        for t in range(history_len, len(game_segments) - chunk_length + 1):
            w_in = torch.cat(y_outs[-history_len:], dim=1)  # history
            x_in = game_segments[:, [t]]
            y_gt = segment_ys[:, t : t + chunk_length]

            z, k, z_soft, z_hard, _ = model.forward(x_in.notes * 1, x_in.bombs * 1, x_in.obstacles * 1, w_in * 1)
            y_out = ref_model.decode(z_hard * 1)

            y_outs.append(y_out[:, 0, [0]])

            rollout_xins.append(x_in)
            rollout_wins.append(w_in)
            rollout_ygts.append(y_gt)
            rollout_youts.append(y_out[:, 0])

        rollout_xins = GameSegment.cat(rollout_xins, dim=0)
        rollout_wins = torch.cat(rollout_wins, dim=0)
        rollout_ygts = torch.cat(rollout_ygts, dim=0)
        rollout_youts = torch.cat(rollout_youts, dim=0)

    model.train()
    ref_model.train()
    return rollout_xins, rollout_wins, rollout_ygts, rollout_youts


class XRORDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_filenames = sorted(glob.glob(f"{data_dir}/*.pkl"))

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        filename = self.data_filenames[idx]
        d = torch.load(filename, weights_only=False)

        # quick static scaling. eventually needs refactoring
        p = d["my_pos_expm"]
        p = p.reshape(-1, 3, 6)
        ratio = 1.5044 / np.nanmedian(p[:100, 0, 1])  # head height
        p[..., :3] *= ratio
        d["my_pos_expm"] = p.reshape(-1, 18)

        # # For Debug
        # d["my_pos_expm"] = d["my_pos_expm"][:3]
        # d["song_np"] = d["song_np"][:3]
        # d["timestamps"] = d["timestamps"][:3]

        return d


class XRORDataset2(Dataset):
    def __init__(self, data_dir, logger):
        self.data_dir = data_dir
        self.data_filenames = sorted(glob.glob(f"{data_dir}/**/*.xror", recursive=True))
        self.logger = logger

        # TODO: build the (shard, file) pairs here. first open the manifest file and then populate indices

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        n_tries = 100
        for i in range(n_tries):
            try:
                filename = self.data_filenames[idx]
                d = open_xror(filename)

                # Drop if the file doesn't have enough frames or doesn't have obstacles
                if d["timestamps"].shape[0] < 1000 or np.isnan(d["gt_3p_np"]).sum() > 0 or np.isinf(d["gt_3p_np"]).sum() > 0:
                    self.data_filenames.pop(idx)
                else:
                    return d
            except Exception as e:
                self.data_filenames.pop(idx)
            # except KeyError as e:
            #     self.data_filenames.pop(idx)
            # except IndexError as e:
            #     self.logger.info(f"Index error occurred")
            #     self.data_filenames.pop(idx)
            finally:
                idx = np.random.randint(len(self.data_filenames))

        raise FileNotFoundError


def handler(sig, frame):
    print("Segfault")
    return


def safe_unpack(xror_raw):
    res = XROR.unpack(xror_raw)
    return res


class BoxrrHFDataset(Dataset):
    def __init__(self, index_pair_sets, data_dir, logger):
        self.data_dir = data_dir
        self.shard_names = sorted(glob.glob(f"{data_dir}/cschell___boxrr-23/default/0.0.0/**/*.arrow", recursive=True))
        # self.shard_sizes = np.loadtxt(f"{data_dir}/boxrr_shard_sizes.txt").astype(int)
        self.index_pair_sets = index_pair_sets

        # max_shard_size = self.shard_sizes.max()
        # idxs = np.arange(max_shard_size)[None].repeat(self.shard_sizes.shape[0], 0)
        # shard_idxs = np.arange(self.shard_sizes.shape[0])[:, None].repeat(max_shard_size, 1)
        # shard_sizes_repped = self.shard_sizes[:, None].repeat(max_shard_size, 1)
        # stacked = np.stack([shard_idxs, idxs], axis=-1)
        # yes = idxs < shard_sizes_repped
        # self.sf_pairs = stacked[yes]

        # yes = np.logical_and(self.sf_pairs[:, 0] == 1294, self.sf_pairs[:, 1] >= 355)
        # yes = self.sf_pairs[:, 0] == 1294
        # self.sf_pairs = self.sf_pairs[yes]

        # To valdate
        # self.sf_pairs = self.sf_pairs[[0]]

        self.logger = logger
        # self.pool = multiprocessing.Pool(1)
        # self.q = Queue()

    def __len__(self):
        # valid_idxs = np.where(self.sf_pairs[:, 0] > -1)[0]
        # return len(valid_idxs)
        return self.index_pair_sets.shape[0]

    def __getitem__(self, idx):
        ret = []
        index_pair_set = self.index_pair_sets[idx]

        for index_pair in index_pair_set:
            shard_idx, file_idx = index_pair
            shard_name = self.shard_names[shard_idx]
            shard_ds = DDataset.from_file(shard_name)
            xror_raw = shard_ds[file_idx.item()]

            xror_unpacked = XROR.unpack(xror_raw["xror"])
            beatmap, map_info = open_beatmap_from_unpacked_xror(xror_unpacked)
            left_handed = xror_unpacked.data["info"]["software"]["activity"].get("leftHanded", False)
            opened = load_cbo_and_3p(xror_unpacked, beatmap, map_info, left_handed=left_handed, rescale_yes=True)
            ret.append(opened)

        return ret
        # return xror_raw

        # n_tries = 100
        # for i in range(n_tries):
        #     # valid_idxs = np.where(self.sf_pairs[:, 0] > -1)[0]
        #     # actual_idx = valid_idxs[idx]
        #     actual_idx = idx
        #     shard_idx, file_idx = self.sf_pairs[actual_idx]
        #     shard_name = self.shard_names[shard_idx]
        #     try:
        #         shard_ds = DDataset.from_file(shard_name)
        #         xror_raw = shard_ds[file_idx.item()]["xror"]
        #         # proc = Process(target=safe_unpack, args=(xror_raw, q))
        #         # future = self.pool.apply_async(safe_unpack, args=(xror_raw,))
        #         xror = safe_unpack(xror_raw)
        #         # proc.start()
        #         # xror = future.get(10)
        #         # proc.join()
        #         # signal(SIGSEGV, handler)
        #         # signal(SIGFPE, handler)
        #         # xror = XROR.unpack(xror_raw)
        #         # if proc.exitcode != 0:
        #         #     print("Segfault detected")
        #         #     return None
        #
        #         d = open_unpacked_xror(xror, True)
        #
        #         # Drop if the file doesn't have enough frames or doesn't have obstacles
        #         if d["timestamps"].shape[0] < 1000 or np.isnan(d["gt_3p_np"]).sum() > 0 or np.isinf(d["gt_3p_np"]).sum() > 0:
        #             # self.sf_pairs = np.delete(self.sf_pairs, idx, 0)
        #             self.sf_pairs[actual_idx] = -1
        #             self.logger.info(f"Masked {actual_idx}")
        #             return None
        #
        #         else:
        #             return d
        #     except Exception as e:
        #         # self.sf_pairs = np.delete(self.sf_pairs, idx, 0)
        #         self.sf_pairs[actual_idx] = -1
        #         self.logger.info(f"Masked {actual_idx}")
        #         print(traceback.format_exc())
        #         # raise e
        #         # finally:
        #         # valid_idxs = np.where(self.sf_pairs[:, 0] > -1)[0]
        #         # idx = np.random.choice(len(valid_idxs))
        #         return None
        #
        # raise FileNotFoundError


class BoxrrCacheSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.sample_i = 0

    def __iter__(self):
        batch = []
        while len(batch) < self.batch_size:
            batch.append(self.dataset[self.sample_i])
            self.sample_i += 1
            if self.sample_i >= self.num_samples:
                self.sample_i = 0
                self.dataset.open_next_shard()
        yield batch

    def __len__(self):
        # Run forever
        return int(np.iinfo(np.int32).max)


def load_shards(q: Queue, msg_q: Queue, shard_names: list[str], init_i: int):
    # Perpetually iterate through the shards, pushing to the throttled queue
    while True:
        for shard_name in shard_names[init_i:]:
            msg = msg_q.get()
            if msg is None:
                print("load_shards exiting")
                q.put(None)
                return
            else:
                # print("loading")
                d = torch.load(shard_name, weights_only=False)
                # print("loaded")
                q.put(d)


class BoxrrCacheDataset(IterableDataset):
    def __init__(self, data_dir, batch_size, init_i, mode):
        self.data_dir = data_dir
        self.mode = mode  # one of ["main", "heldout", "all"]
        self.shard_names = sorted(glob.glob(f"{data_dir}/**/*.pkl", recursive=True))
        self.current_shard_i = -1
        self.current_shard = self.shard_names[0]
        # self.open_next_shard()
        self.sample_i = 0
        self.batch_size = batch_size
        q_size = min(20, len(self.shard_names))
        self.q = Queue(maxsize=q_size)
        self.msg_q = Queue(maxsize=q_size)
        self.proc = Thread(target=load_shards, args=(self.q, self.msg_q, self.shard_names, init_i))
        self.proc.daemon = True
        self.proc.start()
        # Fill the queue with first shard
        for _ in range(q_size):
            self.msg_q.put(1)
        self.d = self.q_get_and_flatten()

    def open_next_shard(self):
        self.current_shard_i += 1
        if self.current_shard_i >= len(self.shard_names):
            self.current_shard_i = 0
        self.current_shard = self.shard_names[self.current_shard_i]
        self.d = torch.load(self.current_shard, weights_only=False)

    def __len__(self):
        return int(np.iinfo(np.int32).max)

    def __iter__(self):
        while True:
            yield_me = []
            while len(yield_me) < self.batch_size:
                yield_me.append(self.d[self.sample_i])
                self.sample_i += 1
                if self.sample_i >= len(self.d):
                    self.sample_i = 0
                    # print("loading")
                    self.d = self.q_get_and_flatten()
                    # print("loaded")
                    if self.d is None:
                        break
                    self.msg_q.put(1)
            yield yield_me

    def q_get_and_flatten(self):
        d = self.q.get()
        if d is None:
            return None
        d.setdefault("main", {})
        d.setdefault("heldout", {})
        d.setdefault("etc", {})
        if self.mode == "main":
            d = d["main"]
        elif self.mode == "heldout":
            d = d["heldout"]
        elif self.mode == "all":
            d = {**d["main"], **d["heldout"], **d["etc"]}
        elif self.mode == "classy":
            d = {**d["heldout"], **d["etc"]}
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        # Flatten all k, v. v is a list, and we want to concat all lists
        d = [{"id": k, **item} for k, sublist in d.items() for item in sublist]
        if self.mode == "all":
            # Shuffle the list
            np.random.shuffle(d)
        return d


class R1Dataset(Dataset):
    def __init__(self, sf_pairs, data_dir, logger):
        self.data_dir = data_dir
        self.shard_names = sorted(glob.glob(f"{data_dir}/cschell___boxrr-23/default/0.0.0/**/*.arrow", recursive=True))
        self.sf_pairs = sf_pairs
        self.logger = logger

    def __len__(self):
        return self.sf_pairs.shape[0]

    def __getitem__(self, idx):
        song_hash, difficulty, mapcode, usercode = self.sf_pairs[idx]
        zip_path = f"{proj_dir}/datasets/BeatSaver/{song_hash}.zip"
        d = open_bsmg(zip_path, difficulty)
        return d


class BoxrrHFStreamDataset(Dataset):
    def __init__(self, umds_df, logger):
        self.ds = load_dataset("cschell/boxrr-23", cache_dir=f"{proj_dir}/datasets/boxrr-23", streaming=True, split="train")
        self.umds_df = umds_df
        self.logger = logger
        self.current_idx = 0

    def __len__(self):
        return self.umds_df.shape[0]

    def __getitem__(self, idx):
        actual_idx = self.umds_df.iloc[idx]["HF Index"]
        # xror = safe_unpack(xror_raw)
        diff = actual_idx - self.current_idx
        self.ds = self.ds.skip(diff)
        xror_raw = list(self.ds.take(1).iter(1))[0]["xror"][0]
        xror = XROR.unpack(xror_raw)
        beatmap, map_info = open_beatmap_from_unpacked_xror(xror)
        self.current_idx = actual_idx
        d = load_cbo_and_3p(xror, beatmap, map_info, True)
        return d


class BSMGDataset(Dataset):
    def __init__(self, data_dir, logger):
        self.data_dir = data_dir
        self.data_filenames = sorted(glob.glob(f"{data_dir}/**/*.xror", recursive=True))
        self.logger = logger

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        filename = self.data_filenames[idx]

        d = open_bsmg(filename, self.logger)

        return d


def cache_collate_fn(batch):
    # skip Nones
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    dd = {}
    for k in list(batch[0].keys()):
        dd[k] = []
        if isinstance(batch[0][k], np.ndarray) or isinstance(batch[0][k], torch.Tensor):
            for d in batch:
                dd[k].append(torch.as_tensor(d[k]))

            # Apply nanpads
            stacked_tensors = torch.stack(dd[k], dim=0)
            dd[k] = stacked_tensors.to(dtype=torch.float)
        else:
            for d in batch:
                dd[k].append(d[k])
    # dd["lengths"] = all_lengths.to(dtype=torch.long)
    return dd


def nanpad_collate_fn(batch):
    # skip Nones
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # Flatten the batch
    batch = [bb for b in batch for bb in b]

    dd = {}
    for k in list(batch[0].keys()):
        dd[k] = []
        if isinstance(batch[0][k], np.ndarray) or isinstance(batch[0][k], torch.Tensor):
            for d in batch:
                dd[k].append(torch.as_tensor(d[k]))

            # Apply nanpads
            all_lengths = torch.tensor([a.shape[0] for a in dd[k]])
            max_seq_len = all_lengths.max()
            max_seq_len = torch.clip(max_seq_len, min=200)
            lengths_to_go = torch.clip(max_seq_len - all_lengths, min=0)

            nanpads = [torch.ones((lengths_to_go[i], *(dd[k][i].shape[1:]))) * torch.nan for i in range(len(batch))]
            padded_tensors = [torch.cat([a, nanpads[i]], dim=0) for i, a in enumerate(dd[k])]
            stacked_tensors = torch.stack(padded_tensors, dim=0)
            dd[k] = stacked_tensors.to(dtype=torch.float)
        else:
            for d in batch:
                dd[k].append(d[k])
    dd["lengths"] = all_lengths.to(dtype=torch.long)
    return dd


def xror_to_tensor_collate_fn(batch):
    lst = []
    for x in batch:
        try:
            xror_unpacked = XROR.unpack(x["xror"])
            beatmap, map_info = open_beatmap_from_unpacked_xror(xror_unpacked)
            lst.append(load_cbo_and_3p(xror_unpacked, beatmap, map_info, True))
        except Exception as e:
            print(f"Error loading xror: {e}")
            continue
    return nanpad_collate_fn(lst)
    # return nanpad_collate_fn(xror)
