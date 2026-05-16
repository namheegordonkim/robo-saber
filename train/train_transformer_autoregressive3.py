import glob

import numpy as np
import os
from argparse import ArgumentParser

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pauli
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train.augmenters import PurviewXYAugmenter
from train.torch_nets import (
    MLPAutoregressive,
    TransformerDiscrete,
)
from beaty_common.train_utils import ThroughDataset, RepeatSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

torch._dynamo.config.optimize_ddp = False


class IndexDataset(Dataset):
    # I want `batch_size`
    def __init__(self, args):
        self.args = args

    def __len__(self):
        return len(self.args)

    def __getitem__(self, index):
        return index


# For DDP setup
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, args):
    # Preamble for DDP
    if world_size > 1:
        setup(rank, world_size)
    device = torch.device("cuda", rank)

    out_dir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)

    # if world_size > 1:
    #     dist.barrier()

    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")

    writer = SummaryWriter(logdir)
    if rank == 0:
        writer.add_text("args", str(args))

    data_dir = os.path.abspath(args.data_dir)
    data_paths = sorted(glob.glob(f"{data_dir}/*.pkl"))
    ds = []
    for data_path in data_paths:
        d = torch.load(data_path, weights_only=False)
        ds.append(d)

    augmenter = PurviewXYAugmenter()
    global_steps_elapsed = 0
    epochs_elapsed = 0
    n_steps_per_epoch = 100
    batch_size = args.batch_size
    peak_lr = args.peak_lr
    min_lr = 0
    model = None

    # max_scheduled_sampling_length = 10
    max_scheduled_sampling_length = args.max_scheduled_sampling_length
    n_total_epochs = args.n_total_epochs
    n_stages = 10
    n_epochs_per_stage = n_total_epochs // n_stages
    n_warmup_epochs = n_epochs_per_stage // 10
    n_cooldown_epochs = 0
    save_every = n_epochs_per_stage // 1
    eval_every = n_epochs_per_stage // 10

    initial_scheduled_sampling_length = args.init_scheduled_sampling_length
    pred_len = args.pred_len
    tok_horizon = args.tok_horizon

    # n_steps_by_schedule = []
    # for i in range(
    #     initial_scheduled_sampling_length, max_scheduled_sampling_length + 1
    # ):
    #     n_steps_by_schedule.append(i * n_epochs_per_stage)
    #     # n_steps_by_schedule.append(n_epochs_per_stage)
    #
    # # For training beyond H=10
    # n_steps_by_schedule.append(max_scheduled_sampling_length * n_epochs_per_stage * 10)
    # n_total_steps = sum(n_steps_by_schedule)

    pbar = tqdm(total=n_total_epochs, disable=(rank != 0))

    if not args.cached_yes:

        augmented = augmenter.augment(
            ds,
        )
        augmented_tensors = [
            tuple(torch.as_tensor(b, device=device, dtype=torch.float) for b in a)
            for a in augmented
        ]
        all_lengths = torch.tensor(
            [a.shape[0] for a, b, c in augmented_tensors], device=device
        )
        max_seq_len = all_lengths.max()
        lengths_to_go = max_seq_len - all_lengths

        x_nanpads = [
            torch.ones(
                (lengths_to_go[i], *augmented_tensors[i][0].shape[1:]), device=device
            )
            * torch.nan
            for i in range(len(augmented_tensors))
        ]
        y_nanpads = [
            torch.ones(
                (lengths_to_go[i], *augmented_tensors[i][1].shape[1:]), device=device
            )
            * torch.nan
            for i in range(len(augmented_tensors))
        ]
        t_nanpads = [
            torch.ones(
                (lengths_to_go[i], *augmented_tensors[i][2].shape[1:]), device=device
            )
            * torch.nan
            for i in range(len(augmented_tensors))
        ]
        padded_tensors = [
            tuple(
                (
                    torch.cat([a, x_nanpads[i]], dim=0),
                    torch.cat([b, y_nanpads[i]], dim=0),
                    torch.cat([c, t_nanpads[i]], dim=0),
                )
            )
            for i, (a, b, c) in enumerate(augmented_tensors)
        ]

        stacked_x, stacked_y, stacked_t = tuple(
            (
                torch.stack([a for a, b, c in padded_tensors], dim=0),
                torch.stack([b for a, b, c in padded_tensors], dim=0),
                torch.stack([c for a, b, c in padded_tensors], dim=0),
            )
        )

        dd = {
            "all_lengths": all_lengths,
            "stacked_x": stacked_x,
            "stacked_y": stacked_y,
        }
        torch.save(dd, f"{out_dir}/stacked.pkl")
    else:

        dd = torch.load(
            f"{proj_dir}/data/beaterson/song_3p_preproc/stacked.pkl",
            map_location=device,
        )
        stacked_x = dd["stacked_x"]
        stacked_y = dd["stacked_y"]
        # stacked_t = dd["stacked_t"]
        all_lengths = dd["all_lengths"]

    all_x = stacked_x.reshape(-1, stacked_x.shape[-1])
    all_y = stacked_y.reshape(-1, stacked_y.shape[-1])

    if args.train_idx is None:
        train_x = stacked_x[:-1]
        train_y = stacked_y[:-1]
        # train_t = stacked_t[:-1]
        train_lengths = all_lengths[:-1]
    else:
        train_x = stacked_x[args.train_idx]
        train_y = stacked_y[args.train_idx]
        # train_t = stacked_t[args.train_idx]
        train_lengths = all_lengths[args.train_idx]

    valid_x = stacked_x[[-1]]
    valid_y = stacked_y[[-1]]
    valid_lengths = all_lengths[[-1]]

    npr_st = np.random.get_state()
    np.random.seed(0)

    seen_idxs = (torch.rand(batch_size) * train_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    seen_start_idxs = (
        torch.rand(batch_size, device=device)
        * (train_lengths[seen_idxs] - max_scheduled_sampling_length - tok_horizon)
    ).to(dtype=torch.long)
    seen_it = torch.as_tensor(
        torch.stack([seen_idxs, seen_start_idxs], dim=-1), dtype=torch.long
    )

    # for debug
    # seen_it[..., -1] = 2

    unseen_idxs = (torch.rand(batch_size) * valid_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    unseen_start_idxs = (
        torch.rand(batch_size, device=device)
        * (valid_lengths[unseen_idxs] - max_scheduled_sampling_length - tok_horizon)
        + 1
    ).to(dtype=torch.long)
    unseen_it = torch.as_tensor(
        torch.stack([unseen_idxs, unseen_start_idxs], dim=-1), dtype=torch.long
    )

    np.random.set_state(npr_st)

    # Need to pad xs for MLP input
    dataset = IndexDataset(train_y)
    sampler = RepeatSampler(dataset, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        # collate_fn=lambda x: x,
    )
    scheduled_sampling_length = initial_scheduled_sampling_length

    # while global_steps_elapsed <= n_total_steps:
    while epochs_elapsed <= n_total_epochs:
        _, idxs = next(enumerate(dataloader))
        idxs = idxs.to(device=device, dtype=torch.long)
        # Model is dynamically defined so I don't have to precompute feature sizes at all
        pauli_root = os.path.abspath(
            os.path.join(os.getcwd(), "src")
        )  # NOTE: assume that all scripts are run from the parent directory of src.

        if model is None:
            logger.info(f"Model hasn't been initialized yet")
            if args.checkpoint_path is None:
                logger.info(f"Initializing from scratch...")
                # Need to pad xs for MLP input
                if args.arch == "mlp":
                    model = MLPAutoregressive(
                        train_x.shape[-1], 6, 1024, train_y.shape[-1], args.history_len
                    )
                elif args.arch == "dec":
                    model = TransformerDiscrete(
                        train_x.shape[-1],
                        6,
                        args.hidden_size,
                        train_y.shape[-1],
                        args.n_heads,
                        args.n_layers,
                        args.ff_size,
                        args.dropout,
                        args.vocab_size,
                        args.sentence_length,
                        args.n_parts,
                        args.history_len,
                        args.pred_len,
                        args,
                    )
                else:
                    raise NotImplementedError

                horizon_idxs = torch.arange(tok_horizon, device=device)[
                    None
                ].repeat_interleave(train_y.shape[1] - tok_horizon + 1, 0)
                horizon_idxs += torch.arange(horizon_idxs.shape[0], device=device)[
                    ..., None
                ]
                train_yy = train_y[:, horizon_idxs]

                x_nan_yes = torch.any(torch.isnan(train_x), dim=-1)
                y_nan_yes = torch.any(torch.isnan(train_y), dim=-1)
                yy_nan_yes = torch.isnan(train_yy).any(-1).any(-1)
                model.setup(
                    train_x[~x_nan_yes],
                    train_y[~y_nan_yes].reshape(-1, 6),
                    train_yy[~yy_nan_yes].reshape(-1, tok_horizon, 3, 6),
                    device,
                )
                # if args.arch in ["discrete", "dec"]:
                #     writer.add_scalar(
                #         "train/quantization_error", model.quantization_error, 0
                #     )
                optimizer = Adam(model.parameters(), peak_lr)
                # optimizer = AdamW(model.parameters(), peak_lr, weight_decay=1e0)
                global_steps_elapsed = 0
                epochs_elapsed = 0

                vqvae_setup_sample_size = batch_size * 10
                seen_idxs = (
                    torch.rand(vqvae_setup_sample_size) * train_lengths.shape[0]
                ).to(dtype=torch.long, device=device)
                seen_t_start = (
                    torch.rand(vqvae_setup_sample_size, device=device)
                    * (train_lengths[seen_idxs] - tok_horizon)
                ).to(dtype=torch.long)
                segment_ts = (
                    torch.arange(tok_horizon, device=device) + seen_t_start[:, None]
                )
                segment_ys = torch.take_along_dim(
                    train_y[seen_idxs], segment_ts[:, :, None], dim=1
                )
                model.setup2(segment_ys, device)

                print("hehe")

            else:
                logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
                model_dict = torch.load(args.checkpoint_path, weights_only=False)
                model = pauli.load(model_dict)
                # model1 = pauli.load(model_dict)
                # model = TransformerDiscrete(
                #     train_x.shape[-1],
                #     6,
                #     args.hidden_size,
                #     train_y.shape[-1],
                #     args.n_heads,
                #     args.n_layers,
                #     args.ff_size,
                #     args.dropout,
                #     2000,
                #     2,
                #     3,
                #     args.history_len,
                #     args.pred_len,
                #     args,
                # )
                # model.cond_proj = model1.cond_proj

                model.dropout = args.dropout
                for _, mod1 in model.transformer_encoder.layers.named_children():
                    for _, mod2 in mod1.named_children():
                        if isinstance(mod2, torch.nn.Dropout):
                            mod2.p = args.dropout

                model.load_state_dict(model_dict["model_state_dict"])
                # dist.barrier()
                # for param in model.parameters():
                #     dist.broadcast(param.data, src=0)

                # Ugly hack to make sure we can reuse input embeddings
                # model.hamm_emb = torch.nn.Embedding(2000, 1024)
                # model.sentence_length = args.sentence_length
                # model.xyzexpm_tokenizers = []
                # for i in range(3):
                #     xyzexpm_tokenizer = RVQTokenizer(
                #         args.tok_horizon * 6,
                #         args.sentence_length,
                #         2000,
                #         50,
                #         1,
                #         False,
                #     )
                #     model.xyzexpm_tokenizers.append(xyzexpm_tokenizer)
                #
                # horizon_idxs = torch.arange(tok_horizon, device=device)[
                #     None
                # ].repeat_interleave(train_y.shape[1] - tok_horizon + 1, 0)
                # horizon_idxs += torch.arange(horizon_idxs.shape[0], device=device)[
                #     ..., None
                # ]
                # train_yy = train_y[:, horizon_idxs]
                #
                # x_nan_yes = torch.any(torch.isnan(train_x), dim=-1)
                # y_nan_yes = torch.any(torch.isnan(train_y), dim=-1)
                # yy_nan_yes = torch.isnan(train_yy).any(-1).any(-1)
                # model.setup(
                #     train_x[~x_nan_yes],
                #     train_y[~y_nan_yes].reshape(-1, 6),
                #     train_yy[~yy_nan_yes].reshape(-1, tok_horizon, 3, 6),
                #     device,
                # )
                # if args.arch in ["discrete", "dec"]:
                #     writer.add_scalar(
                #         "train/quantization_error", model.quantization_error, 0
                #     )
                #
                # model.rms_nail = model1.rms_nail
                # model.rms_cond = model1.rms_cond

                model.to(device)
                optimizer = Adam(model.parameters(), peak_lr)
                # optimizer = AdamW(model.parameters(), peak_lr, weight_decay=1e0)
                if args.continue_yes:
                    optimizer.load_state_dict(model_dict["optimizer_state_dict"])
                    global_steps_elapsed = model_dict["global_steps_elapsed"]
                    epochs_elapsed = model_dict["epochs_elapsed"]
                    scheduled_sampling_length = model_dict["scheduled_sampling_length"]
                    # pbar.update(global_steps_elapsed)
                    pbar.update(epochs_elapsed)

            if not args.debug_yes:
                model = torch.compile(model)
                # pass
            if world_size > 1:
                model = DDP(model, device_ids=[rank])
            else:

                class DummyWrapper(torch.nn.Module):
                    def __init__(self, module):
                        super().__init__()
                        self.module = module

                    def forward(self, *aargs, **kwargs):
                        return self.module(*aargs, **kwargs)

                    def compute_loss(self, *aargs):
                        return self.module.compute_loss(*aargs)

                model = DummyWrapper(model)
        orig_mod = model.module._orig_mod if not args.debug_yes else model.module
        # orig_mod = model.module
        # if epochs_elapsed % save_every == 0 or global_steps_elapsed >= n_total_steps:
        if epochs_elapsed % save_every == 0 or epochs_elapsed >= n_total_epochs:
            if rank == 0:
                model_d = {
                    **pauli.dump(orig_mod, pauli_root),
                    "scheduled_sampling_length": scheduled_sampling_length,
                    "model_state_dict": orig_mod.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_steps_elapsed": global_steps_elapsed,
                    "epochs_elapsed": epochs_elapsed,
                    "args": args,
                }
                save_path = f"{out_dir}/mlp_{epochs_elapsed:06d}.pkl"
                torch.save(model_d, save_path)
                logger.info(f"Saved to {save_path}")

            # Sync up
            # dist.barrier()
            # for param in model.parameters():
            #     dist.broadcast(param.data, src=0)

        # Validation step
        if True and rank == 0:
            if epochs_elapsed % eval_every == 0:
                # Prepare some ground truth segments, compare to full autoregressive rollout
                y_hat_start = train_y[seen_it[:, 0], seen_it[:, 1]]
                y_hat_start = y_hat_start.reshape(y_hat_start.shape[0], 1, -1, 6)
                # y_hats = [y_hat_start for _ in range(args.history_len)]
                y_hats = [
                    torch.ones_like(y_hat_start) * torch.nan
                    for _ in range(args.history_len)
                ]
                for i in range(args.history_len):
                    j = args.history_len - i
                    yes = seen_it[:, 1] >= j
                    if yes.any():
                        use_me = train_y[seen_it[yes][:, 0], seen_it[yes][:, 1] - j]
                        use_me = use_me.reshape(use_me.shape[0], 1, -1, 6)
                        y_hats[i][yes] = use_me

                # Rollout here and then form an ad-hoc dataset?
                rollout_xins = []
                rollout_youts = []
                rollout_ytars = []
                with torch.no_grad():
                    model.eval()
                    for t in range(10):
                        x_in = train_x[seen_it[:, 0], seen_it[:, 1] + t]
                        w_in = torch.stack(y_hats[-args.history_len :], 1)
                        w_in = w_in.reshape(w_in.shape[0], args.history_len, 3, 6)

                        if args.pretrain_yes:
                            x_in[:] = torch.nan
                        if args.history_no:
                            w_in[:] = torch.nan

                        y_tar = train_y[seen_it[:, 0], seen_it[:, 1] + t][:, None]
                        y_tar = y_tar.reshape(y_tar.shape[0], 1, -1, 6)
                        y_hat, _ = model(x_in, w_in)
                        y_hat = y_hat[:, [0]]
                        # y_hat = w_in[:, -1] + delta_y_hat
                        y_hats.append(y_hat)

                        rollout_xins.append(x_in)
                        rollout_youts.append(y_hat)
                        rollout_ytars.append(y_tar)

                    rollout_xins = torch.concatenate(rollout_xins, dim=0)
                    rollout_youts = torch.concatenate(rollout_youts, dim=0)
                    rollout_ytars = torch.concatenate(rollout_ytars, dim=0)

                    mse_loss = torch.nn.MSELoss()
                    seen_loss = mse_loss(rollout_youts, rollout_ytars)
                logger.info(f"Eval MSE on Seen Segments: {seen_loss:.3f}")

                # Unseen
                # y_hats = []
                y_hat_start = train_y[unseen_it[:, 0], unseen_it[:, 1]]
                y_hat_start = y_hat_start.reshape(y_hat_start.shape[0], 1, -1, 6)
                y_hats = [
                    torch.ones_like(y_hat_start) * torch.nan
                    for _ in range(args.history_len)
                ]
                y_hats[-1] = y_hat_start
                # y_hats = [
                #     torch.ones_like(y_hat_start) * torch.nan
                #     for _ in range(args.history_len)
                # ]
                # Rollout here and then form an ad-hoc dataset?
                rollout_xins = []
                rollout_youts = []
                rollout_ytars = []
                with torch.no_grad():
                    for t in range(10):
                        x_in = valid_x[unseen_it[:, 0], unseen_it[:, 1] + t]
                        w_in = torch.stack(y_hats[-args.history_len :], 1)
                        w_in = w_in.reshape(w_in.shape[0], args.history_len, 3, 6)

                        if args.pretrain_yes:
                            x_in[:] = torch.nan
                        if args.history_no:
                            w_in[:] = torch.nan

                        y_tar = train_y[unseen_it[:, 0], unseen_it[:, 1] + t][:, None]
                        y_tar = y_tar.reshape(y_tar.shape[0], 1, -1, 6)
                        y_hat, _ = model(x_in, w_in)
                        y_hat = y_hat[:, [0]]
                        # y_hat = w_in[:, -1] + delta_y_hat
                        y_hats.append(y_hat)

                        rollout_xins.append(x_in)
                        rollout_youts.append(y_hat)
                        rollout_ytars.append(y_tar)

                    rollout_xins = torch.concatenate(rollout_xins, dim=0)
                    rollout_youts = torch.concatenate(rollout_youts, dim=0)
                    rollout_ytars = torch.concatenate(rollout_ytars, dim=0)

                    mse_loss = torch.nn.MSELoss()
                    unseen_loss = mse_loss(rollout_youts, rollout_ytars)

                logger.info(f"Eval MSE on Unseen Segments: {unseen_loss:.3f}")
                writer.add_scalar(
                    "eval/train_mse/steps", seen_loss.data, global_steps_elapsed
                )
                writer.add_scalar(
                    "eval/train_mse/epochs", seen_loss.data, epochs_elapsed
                )
                writer.add_scalar(
                    "eval/valid_mse/steps", unseen_loss.data, global_steps_elapsed
                )
                writer.add_scalar(
                    "eval/valid_mse/epochs", unseen_loss.data, epochs_elapsed
                )

        peak_lr_dived = peak_lr
        # peak_lr_dived = peak_lr / (2 ** (scheduled_sampling_length - 1))
        # peak_lr_dived = np.clip(peak_lr_dived, 1e-5, np.inf)
        if epochs_elapsed % n_epochs_per_stage < n_warmup_epochs:
            lr = min_lr + (peak_lr_dived - min_lr) * np.clip(
                (epochs_elapsed % n_epochs_per_stage) / n_warmup_epochs, 0, 1
            )
        elif (
            epochs_elapsed % n_epochs_per_stage
            >= n_epochs_per_stage - n_cooldown_epochs
        ):
            lr = min_lr
        else:
            # cosine decay
            lr = (
                min_lr
                + (peak_lr_dived - min_lr)
                * (
                    1
                    + np.cos(
                        np.pi
                        * (epochs_elapsed % n_epochs_per_stage - n_warmup_epochs)
                        / (n_epochs_per_stage - n_warmup_epochs - n_cooldown_epochs)
                    )
                )
                / 2
            )

        # Comment out for lr scheduling
        if not args.lr_decay_yes:
            lr = peak_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # return
        # if global_steps_elapsed < n_total_steps:
        if epochs_elapsed < n_total_epochs:
            # Minibatch loading: should actually iterate sequence samples not just raw rows
            # for i, (x, y) in enumerate(dataloader):
            lengths = train_lengths[idxs]
            t_start = (
                torch.rand((batch_size,), device=device)
                # torch.ones((batch_size,), device=device)
                * (lengths - max_scheduled_sampling_length - tok_horizon)
            ).to(dtype=torch.long)

            it = torch.stack([idxs, t_start], dim=-1)

            # for debug
            # it[..., -1] = 2

            y_hat_start = train_y[it[:, 0], it[:, 1]]
            y_hat_start = y_hat_start.reshape(y_hat_start.shape[0], 1, -1, 6)
            y_hats = [
                torch.ones_like(y_hat_start) * torch.nan
                for _ in range(args.history_len)
            ]

            for i in range(args.history_len):
                j = args.history_len - i
                yes = it[..., -1] >= j
                if yes.any():
                    use_me = train_y[it[yes][:, 0], it[yes][:, 1] - j]
                    use_me = use_me.reshape(use_me.shape[0], 1, -1, 6)
                    y_hats[i][yes] = use_me

            # Rollout here and then form an ad-hoc dataset?
            rollout_xins = []
            rollout_wins = []
            rollout_gts = []
            rollout_ts = []
            model.eval()
            with torch.no_grad():
                for t in range(scheduled_sampling_length):
                    x_in = train_x[it[:, 0], it[:, 1] + t]
                    w_in = torch.stack(y_hats[-args.history_len :], 1)
                    w_in = w_in.reshape(w_in.shape[0], args.history_len, 3, 6)

                    if args.pretrain_yes:
                        x_in[:] = torch.nan
                    if args.history_no:
                        w_in[:] = torch.nan

                    y_hat, _ = model(x_in, w_in, topk=1)  # don't collect logits here
                    y_hat = y_hat[:, [0]]
                    y_tar_ts = (it[:, 1] + t)[:, None] + torch.arange(
                        tok_horizon, device=train_y.device
                    )
                    y_tar = torch.take_along_dim(
                        train_y[it[:, 0]], y_tar_ts[:, :, None], dim=1
                    )
                    y_tar = y_tar.reshape(y_tar.shape[0], -1, 3, 6)

                    noise = torch.randn_like(y_hat) * args.sigma
                    y_hats.append(y_hat + noise)

                    rollout_xins.append(x_in)
                    rollout_wins.append(w_in)
                    rollout_gts.append(y_tar)
                    # rollout_ts.append(t_tar)

            rollout_xins = torch.cat(rollout_xins, dim=0)
            rollout_wins = torch.cat(rollout_wins, dim=0)
            rollout_gts = torch.cat(rollout_gts, dim=0)
            if True or args.arch not in ["discrete", "dec"]:
                dataset2 = ThroughDataset(rollout_xins, rollout_wins, rollout_gts)
            else:
                # Pre-tokenization to marginally get faster
                rollout_gts = rollout_gts.reshape(
                    rollout_gts.shape[0], orig_mod.n_parts, orig_mod.part_size
                )
                rollout_gts = orig_mod.rms_hamm.normalize(rollout_gts)
                rollout_gts_tok, rollout_gts_quantized = orig_mod.hamm_tokenizer.encode(
                    rollout_gts.reshape(-1, 6), device=device
                )
                rollout_gts = rollout_gts_tok.reshape(
                    -1, orig_mod.n_parts, orig_mod.sentence_length
                )
                dataset2 = ThroughDataset(
                    rollout_xins,
                    rollout_wins,
                    rollout_gts,
                )
            dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)
            n_microbatches = args.n_microbatches
            model.train()
            # for x_in, w_in, y_tar, t_tar in dataloader2:
            for x_in, w_in, y_tar in dataloader2:

                x_in = x_in.to(device, non_blocking=True, dtype=torch.float)
                w_in = w_in.to(device, non_blocking=True, dtype=torch.float)
                if args.arch not in ["discrete"]:
                    y_tar = y_tar.to(device, non_blocking=True, dtype=torch.float)
                else:
                    y_tar = y_tar.to(device, non_blocking=True, dtype=torch.long)
                # y_tar = y_tar[:, :-1]

                if args.symmetry_yes:
                    flip_yes = np.random.random(x_in.shape[0]) < 0.5
                    # Study symmetry here
                    pos = w_in[flip_yes, ..., :3]
                    pos[:, :, [1, 2]] = pos[:, :, [2, 1]]
                    pos[..., [1]] *= -1
                    w_in[flip_yes, ..., :3] = pos

                    expm = w_in[flip_yes, ..., 3:]
                    expm[:, :, [1, 2]] = expm[:, :, [2, 1]]
                    expm[..., [0, 2]] *= -1
                    w_in[flip_yes, ..., 3:] = expm

                    # color
                    x_in[flip_yes, ..., 3] = 1 - x_in[flip_yes, ..., 3]
                    # y location
                    x_in[flip_yes, ..., 1] = 3 - x_in[flip_yes, ..., 1]
                    # cut direction
                    # tmp = x_in[flip_yes, ..., 4] * 1
                    left_yes = (
                        (8 > x_in[flip_yes, ..., 4])
                        * (x_in[flip_yes, ..., 4] > 1)
                        * (x_in[flip_yes, ..., 4] % 2 == 0)
                    )
                    right_yes = (
                        (8 > x_in[flip_yes, ..., 4])
                        * (x_in[flip_yes, ..., 4] > 1)
                        * (x_in[flip_yes, ..., 4] % 2 == 1)
                    )
                    x_in[flip_yes, ..., 4][left_yes] = (
                        x_in[flip_yes, ..., 4][left_yes] + 1
                    )
                    x_in[flip_yes, ..., 4][right_yes] = (
                        x_in[flip_yes, ..., 4][right_yes] - 1
                    )

                    pos = y_tar[flip_yes, ..., :3]
                    pos[:, :, [1, 2]] = pos[:, :, [2, 1]]
                    pos[..., [1]] *= -1
                    y_tar[flip_yes, ..., :3] = pos

                    expm = y_tar[flip_yes, ..., 3:]
                    expm[:, :, [1, 2]] = expm[:, :, [2, 1]]
                    expm[..., [0, 2]] *= -1
                    y_tar[flip_yes, ..., 3:] = expm

                microbatches_x_in = torch.tensor_split(x_in, n_microbatches)
                microbatches_w_in = torch.tensor_split(w_in, n_microbatches)
                microbatches_y_tar = torch.tensor_split(y_tar, n_microbatches)

                optimizer.zero_grad()

                for xx, ww, y_tartar in zip(
                    microbatches_x_in, microbatches_w_in, microbatches_y_tar
                ):
                    # y_hat = model(xx, yy, y_tartar)
                    if args.arch in ["discrete", "dec", "transformer"]:
                        loss = model.compute_loss(
                            xx,
                            ww.reshape(ww.shape[0], args.history_len, 3, 6),
                            y_tartar,
                        )
                    else:
                        y_hat, loss = model(
                            xx,
                            ww,
                            y_tartar,
                        )
                        loss = torch.nn.functional.mse_loss(y_hat, y_tartar)
                        if loss.isnan().any():
                            print("Bruh")
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e2)
                optimizer.step()
                global_steps_elapsed += 1

                # pbar.update(1)
                # pbar.set_postfix({"loss": loss.item(), "steps": global_steps_elapsed})

            epochs_elapsed += 1
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item(), "epochs": epochs_elapsed})
            # if epochs_elapsed >= n_total_epochs:
            #     break

            if epochs_elapsed % n_epochs_per_stage == 0:
                scheduled_sampling_length += 1
                scheduled_sampling_length = np.clip(
                    scheduled_sampling_length, 1, max_scheduled_sampling_length
                )
                # break
            if rank == 0:
                writer.add_scalar("train/loss/steps", loss.item(), global_steps_elapsed)
                writer.add_scalar("train/loss/epochs", loss.item(), epochs_elapsed)
                writer.add_scalar("train/lr/steps", lr, global_steps_elapsed)
                writer.add_scalar("train/lr/epochs", lr, epochs_elapsed)
                writer.add_scalar(
                    "train/scheduled_sampling_length",
                    scheduled_sampling_length,
                    global_steps_elapsed,
                )
        else:
            break

    # dist.barrier()
    pbar.close()
    if rank == 0:
        writer.flush()

    if world_size > 1:
        cleanup()


def main():
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12346,
            stdout_to_server=True,
            stderr_to_server=True,
            suspend=False,
        )
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    current_path = os.getcwd()
    print("Current Path:", current_path)

    # For DDP serialization and setup
    world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.spawn(
            train,
            args=(
                world_size,
                args,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        train(0, world_size, args)

    # logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=144)
    parser.add_argument("--ff_size", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--n_microbatches", type=int, default=1)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--n_total_epochs", type=int, default=int(3e4))
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--train_idx", nargs="+", type=int)
    parser.add_argument("--max_scheduled_sampling_length", type=int, default=10)
    parser.add_argument("--cached_yes", action="store_true")
    parser.add_argument("--history_len", type=int, default=2)
    parser.add_argument("--tf_ratio", type=float, default=0.6)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--residual_yes", action="store_true")
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument(
        "--arch",
        type=str,
        choices=["transformer", "mlp", "discrete", "dec"],
        default="transformer",
    )
    parser.add_argument("--symmetry_yes", action="store_true")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--init_scheduled_sampling_length", type=int, default=1)
    parser.add_argument("--lr_decay_yes", action="store_true")
    parser.add_argument("--pretrain_yes", action="store_true")
    parser.add_argument("--history_no", action="store_true")
    parser.add_argument("--tokenize_notes_yes", action="store_true")
    parser.add_argument("--tokenize_history_yes", action="store_true")
    parser.add_argument("--tok_horizon", type=int, default=1)
    parser.add_argument("--sentence_length", type=int, default=1)
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
