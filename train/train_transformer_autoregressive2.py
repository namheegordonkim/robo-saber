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
    TransformerContinuous,
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

    max_scheduled_sampling_length = 1
    # max_scheduled_sampling_length = 1
    n_total_steps = args.n_total_steps
    n_total_epochs = args.n_total_epochs
    # increase_every = n_total_steps // 10
    # increase_every = n_total_epochs // max_scheduled_sampling_length
    n_epochs_per_stage = args.n_epochs_per_stage
    # n_warmup_global_steps = increase_every // 10
    n_warmup_epochs = n_epochs_per_stage // 10
    # n_cooldown_global_steps = increase_every // 10
    # n_cooldown_epochs = n_epochs_per_stage // 10
    n_cooldown_epochs = 0
    # save_every = n_total_steps // 10
    # save_every = n_total_epochs // max_scheduled_sampling_length
    save_every = n_epochs_per_stage // 1
    eval_every = n_epochs_per_stage // 10

    n_steps_by_schedule = []
    for i in range(1, max_scheduled_sampling_length + 1):
        n_steps_by_schedule.append(i * n_epochs_per_stage)
    n_total_steps = sum(n_steps_by_schedule)

    pbar = tqdm(total=n_total_steps, disable=(rank != 0))

    # augmented = augmenter.augment(
    #     ds,
    #     batch_size * n_steps_per_epoch,
    # )
    # augmented_tensors = [
    #     tuple(torch.as_tensor(b, device=device, dtype=torch.float) for b in a)
    #     for a in augmented
    # ]
    # all_lengths = torch.tensor(
    #     [a.shape[0] for a, b in augmented_tensors], device=device
    # )
    # max_seq_len = all_lengths.max()
    # lengths_to_go = max_seq_len - all_lengths
    #
    # x_nanpads = [
    #     torch.ones(
    #         (lengths_to_go[i], *augmented_tensors[i][0].shape[1:]), device=device
    #     )
    #     * torch.nan
    #     for i in range(len(augmented_tensors))
    # ]
    # y_nanpads = [
    #     torch.ones(
    #         (lengths_to_go[i], *augmented_tensors[i][1].shape[1:]), device=device
    #     )
    #     * torch.nan
    #     for i in range(len(augmented_tensors))
    # ]
    # padded_tensors = [
    #     tuple(
    #         (torch.cat([a, x_nanpads[i]], dim=0), torch.cat([b, y_nanpads[i]], dim=0))
    #     )
    #     for i, (a, b) in enumerate(augmented_tensors)
    # ]
    #
    # stacked_x, stacked_y = tuple(
    #     (
    #         torch.stack([a for a, b in padded_tensors], dim=0),
    #         torch.stack([b for a, b in padded_tensors], dim=0),
    #     )
    # )

    # if rank == 0:
    #     dd = {
    #         "all_lengths": all_lengths,
    #         "stacked_x": stacked_x,
    #         "stacked_y": stacked_y,
    #     }
    #     torch.save(dd, f"{out_dir}/stacked.pkl")

    dd = torch.load(
        f"{proj_dir}/data/beaterson/song_3p_preproc/stacked.pkl", map_location=device
    )
    stacked_x = dd["stacked_x"]
    stacked_y = dd["stacked_y"]
    all_lengths = dd["all_lengths"]

    all_x = stacked_x.reshape(-1, stacked_x.shape[-1])
    all_y = stacked_y.reshape(-1, stacked_y.shape[-1])

    if args.train_idx is None:
        train_x = stacked_x[:-1]
        train_y = stacked_y[:-1]
        train_lengths = all_lengths[:-1]
    else:
        train_x = stacked_x[[args.train_idx]]
        train_y = stacked_y[[args.train_idx]]
        train_lengths = all_lengths[[args.train_idx]]

    valid_x = stacked_x[[-1]]
    valid_y = stacked_y[[-1]]
    valid_lengths = all_lengths[[-1]]

    npr_st = np.random.get_state()
    np.random.seed(0)

    seen_idxs = (torch.rand(batch_size) * train_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    seen_start_idxs = (
        torch.rand(batch_size, device=device) * (train_lengths[seen_idxs] - 10)
    ).to(dtype=torch.long)
    seen_it = torch.as_tensor(
        torch.stack([seen_idxs, seen_start_idxs], dim=-1), dtype=torch.long
    )

    unseen_idxs = (torch.rand(batch_size) * valid_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    unseen_start_idxs = (
        torch.rand(batch_size, device=device) * (valid_lengths[unseen_idxs] - 10)
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
    scheduled_sampling_length = 1

    while global_steps_elapsed <= n_total_steps:
        # while epochs_elapsed <= n_total_epochs:
        _, idxs = next(enumerate(dataloader))
        idxs = idxs.to(device=device, dtype=torch.long)
        # idxs = idxs.pin_memory().to(device="cuda", dtype=torch.long)
        # Model is dynamically defined so I don't have to precompute feature sizes at all
        model_src = None
        sys_modules = None
        pauli_root = os.path.abspath(
            os.path.join(os.getcwd(), "src")
        )  # NOTE: assume that all scripts are run from the parent directory of src.

        if model is None:
            logger.info(f"Model hasn't been initialized yet")
            if args.checkpoint_path is None:
                logger.info(f"Initializing from scratch...")
                # Need to pad xs for MLP input
                if args.arch == "transformer":
                    model = TransformerContinuous(
                        train_x.shape[-1],
                        train_y.shape[-1],
                        args.hidden_size,
                        train_y.shape[-1],
                        args.n_heads,
                        args.n_layers,
                        args.ff_size,
                        0.0,
                    )
                elif args.arch == "mlp":
                    model = MLPAutoregressive(
                        train_x.shape[-1], train_y.shape[-1], 1024, train_y.shape[-1]
                    )
                else:
                    raise NotImplementedError
                x_nan_yes = torch.any(torch.isnan(train_x), dim=-1)
                y_nan_yes = torch.any(torch.isnan(train_y), dim=-1)
                # all_x[x_nan_yes] = 0
                # all_y[y_nan_yes] = 0
                # model = MLP(xs.shape[-1], xs.shape[-1], 1024, ys.shape[-1])
                model = model.to(device)
                model.rms_nail = model.rms_nail.to(device)
                model.rms_cond = model.rms_cond.to(device)
                model.rms_nail.update(train_x[~x_nan_yes])
                model.rms_cond.update(train_y[~y_nan_yes])
                # optimizer = AdamW(model.parameters(), peak_lr)
                optimizer = Adam(model.parameters(), peak_lr)
                global_steps_elapsed = 0
                epochs_elapsed = 0
            else:
                logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
                model_dict = torch.load(args.checkpoint_path, weights_only=False)
                model = pauli.load(model_dict)

                model.load_state_dict(model_dict["model_state_dict"])
                # dist.barrier()
                # for param in model.parameters():
                #     dist.broadcast(param.data, src=0)

                # optimizer = AdamW(model.parameters(), peak_lr)
                optimizer = Adam(model.parameters(), peak_lr)
                if args.continue_yes:
                    optimizer.load_state_dict(model_dict["optimizer_state_dict"])
                    global_steps_elapsed = model_dict["global_steps_elapsed"]
                    epochs_elapsed = model_dict["epochs_elapsed"]
                    scheduled_sampling_length = model_dict["scheduled_sampling_length"]
                    pbar.update(global_steps_elapsed)

            if not args.debug_yes:
                model = torch.compile(model)
            if world_size > 1:
                model = DDP(model, device_ids=[rank])
            else:

                class DummyWrapper(torch.nn.Module):
                    def __init__(self, module):
                        super().__init__()
                        self.module = module

                    def forward(self, *aargs):
                        return self.module(*aargs)

                model = DummyWrapper(model)
        if (
            global_steps_elapsed % save_every == 0
            or global_steps_elapsed >= n_total_steps
        ):
            # if epochs_elapsed % save_every == 0 or epochs_elapsed >= n_total_epochs:
            if rank == 0:
                orig_mod = (
                    model.module._orig_mod if not args.debug_yes else model.module
                )
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
        if rank == 0:
            if epochs_elapsed % eval_every == 0:
                # Prepare some ground truth segments, compare to full autoregressive rollout
                old_y_hat = train_y[seen_it[:, 0], seen_it[:, 1]]
                old_y_hat = torch.stack([old_y_hat, old_y_hat], dim=1)

                # Rollout here and then form an ad-hoc dataset?
                rollout_xins = []
                rollout_yins = []
                rollout_gts = []
                with torch.no_grad():
                    for t in range(10):
                        x_in = train_x[seen_it[:, 0], seen_it[:, 1] + t]
                        y_in = old_y_hat * 1
                        y_hat = model(x_in, y_in)
                        y_tar = train_y[seen_it[:, 0], seen_it[:, 1] + t + 1]
                        old_y_hat[:, 0:] = old_y_hat[:, 1:]
                        old_y_hat[:, -1] = y_hat.detach()

                        rollout_xins.append(x_in)
                        rollout_yins.append(y_in[:, -1])
                        rollout_gts.append(y_tar)

                    rollout_xins = torch.concatenate(rollout_xins, dim=0)
                    rollout_yins = torch.concatenate(rollout_yins, dim=0)
                    rollout_gts = torch.concatenate(rollout_gts, dim=0)

                    mse_loss = torch.nn.MSELoss()
                    seen_loss = mse_loss(rollout_yins, rollout_gts)
                logger.info(f"Eval MSE on Seen Segments: {seen_loss:.3f}")

                # Unseen
                old_y_hat = valid_y[unseen_it[:, 0], unseen_it[:, 1]]
                old_y_hat = torch.stack([old_y_hat, old_y_hat], dim=1)

                # Rollout here and then form an ad-hoc dataset?
                rollout_xins = []
                rollout_yins = []
                rollout_gts = []
                with torch.no_grad():
                    for t in range(10):
                        x_in = valid_x[unseen_it[:, 0], unseen_it[:, 1] + t]
                        y_in = old_y_hat * 1
                        y_hat = model(x_in, y_in)
                        y_tar = valid_y[unseen_it[:, 0], unseen_it[:, 1] + t + 1]
                        old_y_hat[:, 0:] = old_y_hat[:, 1:]
                        old_y_hat[:, -1] = y_hat.detach()

                        rollout_xins.append(x_in)
                        rollout_yins.append(y_in[:, -1])
                        rollout_gts.append(y_tar)

                    rollout_xins = torch.concatenate(rollout_xins, dim=0)
                    rollout_yins = torch.concatenate(rollout_yins, dim=0)
                    rollout_gts = torch.concatenate(rollout_gts, dim=0)

                    mse_loss = torch.nn.MSELoss()
                    unseen_loss = mse_loss(rollout_yins, rollout_gts)

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
            # # linear decay
            # lr = min_lr + (peak_lr_dived - min_lr) * (
            #     1
            #     - (
            #         min(
            #             1,
            #             (epochs_elapsed % n_epochs_per_stage - n_warmup_epochs)
            #             # / (n_total_steps - n_warmup_global_steps),
            #             / (n_epochs_per_stage - n_warmup_epochs - n_cooldown_epochs),
            #         )
            #     )
            # )
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
        lr = peak_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # return
        if global_steps_elapsed < n_total_steps:
            # if epochs_elapsed < n_total_epochs:
            # Minibatch loading: should actually iterate sequence samples not just raw rows
            # for i, (x, y) in enumerate(dataloader):
            # TODO: t_start needs to be defined in terms of actual sequence length
            lengths = train_lengths[idxs]
            t_start = (
                torch.rand((batch_size,), device=device)
                * (lengths - scheduled_sampling_length - 1)
            ).to(dtype=torch.long)
            it = torch.stack([idxs, t_start], dim=-1)

            old_y_hat = train_y[it[:, 0], it[:, 1]]
            old_y_hat = torch.stack([old_y_hat, old_y_hat], dim=1)

            # Rollout here and then form an ad-hoc dataset?
            rollout_xins = []
            rollout_yins = []
            rollout_gts = []
            with torch.no_grad():
                for t in range(scheduled_sampling_length):
                    x_in = train_x[it[:, 0], it[:, 1] + t]
                    y_in = old_y_hat * 1
                    y_hat = model(x_in, y_in)
                    y_tar = train_y[it[:, 0], it[:, 1] + t + 1]
                    old_y_hat[:, 0:] = old_y_hat[:, 1:]
                    old_y_hat[:, -1] = y_hat.detach()

                    rollout_xins.append(x_in)
                    rollout_yins.append(y_in)
                    rollout_gts.append(y_tar)

            rollout_xins = torch.concatenate(rollout_xins, dim=0)
            rollout_yins = torch.concatenate(rollout_yins, dim=0)
            rollout_gts = torch.concatenate(rollout_gts, dim=0)
            # torch.save(rollout_xins, f"{out_dir}/rollout_xins_t=1000.pkl")

            dataset2 = ThroughDataset(rollout_xins, rollout_yins, rollout_gts)
            dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)
            n_microbatches = args.n_microbatches
            for x_in, y_in, y_tar in dataloader2:

                x_in = x_in.to(device, non_blocking=True, dtype=torch.float)
                y_in = y_in.to(device, non_blocking=True, dtype=torch.float)
                y_tar = y_tar.to(device, non_blocking=True, dtype=torch.float)

                microbatches_x_in = torch.tensor_split(x_in, n_microbatches)
                microbatches_y_in = torch.tensor_split(y_in, n_microbatches)
                microbatches_y_tar = torch.tensor_split(y_tar, n_microbatches)

                optimizer.zero_grad()

                for xx, yy, y_tartar in zip(
                    microbatches_x_in, microbatches_y_in, microbatches_y_tar
                ):
                    y_hat = model(xx, yy)
                    loss = torch.nn.functional.mse_loss(y_hat, y_tartar)
                    loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e2)
                optimizer.step()
                global_steps_elapsed += 1

                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "steps": global_steps_elapsed})

            epochs_elapsed += 1
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
        # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # run_dir = os.getcwd()
    # out_dir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    # os.makedirs(out_dir, exist_ok=True)
    # logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    # os.makedirs(logdir, exist_ok=True)
    # logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    # logger.info(f"Starting")
    #
    # writer = SummaryWriter(logdir)
    # writer.add_text("args", str(args))

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
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=144)
    parser.add_argument("--ff_size", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--n_microbatches", type=int, default=1)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--n_total_steps", type=int, default=int(3e3))
    parser.add_argument("--n_total_epochs", type=int, default=int(3e4))
    parser.add_argument("--n_epochs_per_stage", type=int, default=int(5e2))
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--train_idx", type=int)
    parser.add_argument(
        "--arch", type=str, choices=["transformer", "mlp"], default="transformer"
    )
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
