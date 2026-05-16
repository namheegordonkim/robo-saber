import glob

import numpy as np
import os
from argparse import ArgumentParser

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pauli
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train import torch_nets
from train.augmenters import PurviewXYAugmenter
from train.torch_nets import (
    TransformerContinuous,
)
from beaty_common.train_utils import ThroughDataset, RepeatSampler


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

    run_dir = os.getcwd()
    out_dir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    os.makedirs(out_dir, exist_ok=True)
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")

    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    n_epochs = 50

    current_path = os.getcwd()
    print("Current Path:", current_path)

    data_dir = os.path.join(out_dir, args.data_dir)
    data_paths = sorted(glob.glob(f"{data_dir}/*.pkl"))
    # data_path = os.path.join(
    #     proj_dir, "data", "beaterson", "interim", "thats_life_raw_input_output.pkl"
    # )
    ds = []
    for data_path in data_paths:
        d = torch.load(data_path, weights_only=False)
        ds.append(d)

    augmenter = PurviewXYAugmenter()
    # augmenter = VanillaAugmenter()
    global_steps_elapsed = 0
    epochs_elapsed = 0
    n_steps_per_epoch = 100
    # batch_size = 1000
    batch_size = 512
    init_lr = 3e-4
    model = None

    max_scheduled_sampling_length = 100
    n_total_steps = int(2.5e5)
    # increase_every = n_total_steps // 10
    increase_every = n_total_steps // 100

    save_every = n_total_steps // 10
    # save_every = 100
    # n_total_steps = int(1e7)
    pbar = tqdm(total=n_total_steps)
    augmented = augmenter.augment(
        ds,
        batch_size * n_steps_per_epoch,
    )
    augmented_tensors = [
        tuple(torch.as_tensor(b, device="cuda", dtype=torch.float) for b in a)
        for a in augmented
    ]
    all_lengths = torch.tensor(
        [a.shape[0] for a, b in augmented_tensors], device="cuda"
    )
    max_seq_len = all_lengths.max()
    lengths_to_go = max_seq_len - all_lengths

    x_nanpads = [
        torch.ones(
            (lengths_to_go[i], *augmented_tensors[i][0].shape[1:]), device="cuda"
        )
        * torch.nan
        for i in range(len(augmented_tensors))
    ]
    y_nanpads = [
        torch.ones(
            (lengths_to_go[i], *augmented_tensors[i][1].shape[1:]), device="cuda"
        )
        * torch.nan
        for i in range(len(augmented_tensors))
    ]
    padded_tensors = [
        tuple(
            (torch.cat([a, x_nanpads[i]], dim=0), torch.cat([b, y_nanpads[i]], dim=0))
        )
        for i, (a, b) in enumerate(augmented_tensors)
    ]
    stacked_x, stacked_y = tuple(
        (
            torch.stack([a for a, b in padded_tensors], dim=0),
            torch.stack([b for a, b in padded_tensors], dim=0),
        )
    )

    # xs_train, ys_train = augmenter.augment(
    #     # # d["song_and_xror_merged"],
    #     # [d["song_np"]],
    #     # # d["my_pos_sixd"],
    #     # [d["my_pos_expm"]],
    #     # [d["timestamps"]],
    #     ds,
    #     batch_size * n_steps_per_epoch,
    #     )[
    #     0
    # ]  # peak at just one x-y sample
    # xs_prevprev = xs_train[:-2]
    # ys_prevprev = ys_train[:-2]
    # xs_prev = xs_train[1:-1]
    # ys_prev = ys_train[1:-1]
    # xs = xs_train[2:]
    # ys = ys_train[2:]

    # Need to pad xs for MLP input
    class IndexDataset(Dataset):
        # I want `batch_size`
        def __init__(self, args):
            self.args = args
            # self.list_of_seqs = list_of_seqs
            # for a1, a2 in zip(self.args, self.args[1:]):
            #     assert a1.shape[0] == a2.shape[0]

        def __len__(self):
            return len(self.args)

        def __getitem__(self, index):
            # true_index = 0
            # indexed = tuple(torch.as_tensor(a[index]) for a in self.args)
            # idxs = torch.arange(indexed[0].shape[0], device=indexed[0].device)
            # t_start = torch.randint(0, indexed[0].shape[0] - 1, (1,))[None]
            # return *indexed, t_start
            return index

    dataset = IndexDataset(augmented)
    # dataset = MockThroughDataset(xs_train[None], ys_train[None])
    # dataset = ThroughDataset(
    #     xs_prevprev,
    #     ys_prevprev,
    #     xs_prev,
    #     ys_prev,
    #     xs,
    #     ys,
    # )
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
    # x = x.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)
    # y = y.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)
    all_x = stacked_x.reshape(-1, stacked_x.shape[-1])
    all_y = stacked_y.reshape(-1, stacked_y.shape[-1])
    while global_steps_elapsed <= n_total_steps:
        _, idxs = next(enumerate(dataloader))
        idxs = idxs.to(device="cuda", dtype=torch.long)
        # idxs = idxs.pin_memory().to(device="cuda", dtype=torch.long)
        # Model is dynamically defined so I don't have to precompute feature sizes at all
        if model is None:
            logger.info(f"Model hasn't been initialized yet")
            if args.checkpoint_path is None:
                logger.info(f"Initializing from scratch...")
                # xs, ys = augmenter.augment(
                #     # d["song_and_xror_merged"],
                #     [d["song_np"]],
                #     # d["my_pos_sixd"],
                #     [d["my_pos_expm"]],
                #     [d["timestamps"]],
                #     1,
                # )[
                #     0
                # ]  # peak at just one x-y sample
                # Need to pad xs for MLP input
                model = TransformerContinuous(
                    all_x.shape[-1],
                    all_y.shape[-1],
                    args.hidden_size,
                    all_y.shape[-1],
                    args.n_heads,
                    args.n_layers,
                    args.ff_size,
                    0.0,
                )
                all_x[torch.isnan(all_x)] = 0
                all_y[torch.isnan(all_y)] = 0
                # model = MLP(xs.shape[-1], xs.shape[-1], 1024, ys.shape[-1])
                model = model.cuda()
                model.rms_nail = model.rms_nail.cuda()
                model.rms_cond = model.rms_cond.cuda()
                model.rms_nail.update(all_x)
                model.rms_cond.update(all_y)
                optimizer = AdamW(model.parameters(), init_lr)
                global_steps_elapsed = 0
                epochs_elapsed = 0
            else:
                logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
                model_dict = torch.load(args.checkpoint_path, weights_only=False)
                model = pauli.load(model_dict)
                model.load_state_dict(model_dict["model_state_dict"])
                model = model.to("cuda")
                optimizer = Adam(model.parameters(), init_lr)
                optimizer.load_state_dict(model_dict["optimizer_state_dict"])
                global_steps_elapsed = model_dict["global_steps_elapsed"]
                epochs_elapsed = model_dict["epochs_elapsed"]
                pbar.update(global_steps_elapsed)

        if (
            global_steps_elapsed % save_every == 0
            or global_steps_elapsed >= n_total_steps
        ):
            model_d = {
                **pauli.dump(model, torch_nets.__file__),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_steps_elapsed": global_steps_elapsed,
                "epochs_elapsed": epochs_elapsed,
                "args": args,
            }
            save_path = f"{out_dir}/mlp_{epochs_elapsed:06d}.pkl"
            torch.save(model_d, save_path)
            logger.info(f"Saved to {save_path}")

        # Validation step
        logger.info(f"TODO: validation with heldout")
        # with torch.no_grad():
        #     xs_valid = xs_valid.cuda()
        #     ys_valid = ys_valid.cuda()
        #     z, decoded, mu, log_var = model.forward(xs_valid, ys_valid, False)
        #     KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        #     MSE = torch.nn.functional.mse_loss(xs_valid, decoded)
        #     loss = MSE + KLD * args.kld_weight
        #
        #     writer.add_scalar("valid/MSEloss", MSE.item(), global_steps_elapsed)
        #     writer.add_scalar("valid/KLDloss", KLD.item(), global_steps_elapsed)
        #
        #     # for j in range(10):
        #     #     writer.add_scalar(
        #     #         f"valid/mean{j}", torch.mean(z[:, j].abs()), global_steps_elapsed
        #     #     )
        #     #     writer.add_scalar(
        #     #         f"valid/std{j}", torch.std(z[:, j]), global_steps_elapsed
        #     #     )

        # logger.info(
        #     f"Epoch {epochs_elapsed}:\tMSE: {MSE.item():.3f}\tKLD: {KLD.item():.3f}"
        # )

        # lr = init_lr
        lr = init_lr * (1 - (min(1, global_steps_elapsed / n_total_steps)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # return
        if global_steps_elapsed < n_total_steps:
            # Minibatch loading: should actually iterate sequence samples not just raw rows
            # for i, (x, y) in enumerate(dataloader):
            # TODO: t_start needs to be defined in terms of actual sequence length
            lengths = all_lengths[idxs]
            t_start = (
                torch.rand((batch_size,), device="cuda")
                * (lengths - scheduled_sampling_length - 1)
            ).to(dtype=torch.long)
            # t_start = t_start.pin_memory().to("cuda", non_blocking=True)[:, None, None]
            it = torch.stack([idxs, t_start], dim=-1)

            old_y_hat = stacked_y[it[:, 0], it[:, 1]]
            # old_y_hat = torch.take_along_dim(stacked_y, it[..., None], 1)[:, 0]
            # old_y_hat = torch.gather(stacked_y, 2, it[..., None])[:, 0]
            # same as above without having to use [:, 0] and all
            # old_y_hat = torch.

            # Rollout here and then form an ad-hoc dataset?
            rollout_xins = []
            rollout_yins = []
            rollout_gts = []
            # rollout_prs = []
            with torch.no_grad():
                for t in range(scheduled_sampling_length):
                    # x_in = torch.take_along_dim(x, t_start[..., None] + t, 1)[:, 0]
                    x_in = stacked_x[it[:, 0], it[:, 1] + t]
                    y_in = old_y_hat * 1
                    y_hat = model.forward(x_in, y_in, train_yes=False)
                    # y_tar = torch.take_along_dim(y, t_start + t + 1, 1)[:, 0]
                    y_tar = stacked_y[it[:, 0], it[:, 1] + t + 1]
                    old_y_hat = y_hat.detach()

                    rollout_xins.append(x_in)
                    rollout_yins.append(y_in)
                    rollout_gts.append(y_tar)
                    # rollout_prs.append(y_hat)

            rollout_xins = torch.concatenate(rollout_xins, dim=0)
            rollout_yins = torch.concatenate(rollout_yins, dim=0)
            rollout_gts = torch.concatenate(rollout_gts, dim=0)
            # rollout_prs = torch.concatenate(rollout_prs, dim=0)

            dataset2 = ThroughDataset(rollout_xins, rollout_yins, rollout_gts)
            dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)
            for x_in, y_in, y_tar in dataloader2:

                y_hat = model.forward(x_in, y_in, train_yes=False)
                loss = torch.nn.functional.mse_loss(y_hat, y_tar)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_steps_elapsed += 1

                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "steps": global_steps_elapsed})
                if global_steps_elapsed >= n_total_steps:
                    break

                if global_steps_elapsed % increase_every == 0:
                    scheduled_sampling_length += 1
                    scheduled_sampling_length = np.clip(
                        scheduled_sampling_length, 1, max_scheduled_sampling_length
                    )
                    break

            epochs_elapsed += 1

            writer.add_scalar("train/loss", loss.item(), global_steps_elapsed)
            writer.add_scalar("train/lr", lr, global_steps_elapsed)
            writer.add_scalar(
                "train/scheduled_sampling_length",
                scheduled_sampling_length,
                global_steps_elapsed,
            )
        else:
            break

    pbar.close()
    writer.flush()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--ff_size", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
