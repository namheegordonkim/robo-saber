import glob
import os
from argparse import ArgumentParser

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
import pauli
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train.augmenters import PurviewXYAugmenter
from train.torch_nets import (
    ConvVQVAE,
    MlpVQVAE,
    MlpFSQVAE,
    ConvFSQVAE,
    MlpGSVAE,
)
from beaty_common.train_utils import ThroughDataset

torch._dynamo.config.optimize_ddp = False


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

    device = torch.device("cuda")

    outdir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    data_dir = os.path.abspath(args.data_dir)
    data_paths = sorted(glob.glob(f"{data_dir}/*.pkl"))
    ds = []
    for data_path in data_paths:
        d = torch.load(data_path, weights_only=False)
        ds.append(d)
    augmenter = PurviewXYAugmenter()
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
        torch.save(dd, f"{outdir}/stacked.pkl")
    else:

        dd = torch.load(
            f"{proj_dir}/data/beaterson/song_3p_preproc/stacked.pkl",
            map_location=device,
        )
        # stacked_x = dd["stacked_x"]
        stacked_y = dd["stacked_y"]
        # stacked_t = dd["stacked_t"]
        all_lengths = dd["all_lengths"]

    # train_y = stacked_y
    # train_lengths = all_lengths
    train_y = stacked_y[:-1]
    train_lengths = all_lengths[:-1]
    # train_y = stacked_y[:1]
    # train_lengths = all_lengths[:1]

    # train_lengths[:] = 1000

    valid_y = stacked_y[[-1]]
    valid_lengths = all_lengths[[-1]]

    npr_st = np.random.get_state()
    np.random.seed(0)

    batch_size = args.batch_size
    segment_length = args.segment_length
    seen_idxs = (torch.rand(batch_size) * train_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    seen_t_start = (
        torch.rand(batch_size, device=device)
        * (train_lengths[seen_idxs] - segment_length)
    ).to(dtype=torch.long)
    segment_ts = torch.arange(segment_length, device=device) + seen_t_start[:, None]
    seen_segment_ys = torch.take_along_dim(
        train_y[seen_idxs], segment_ts[:, :, None], dim=1
    )

    unseen_idxs = (torch.rand(batch_size) * valid_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    unseen_t_start = (
        torch.rand(batch_size, device=device)
        * (valid_lengths[unseen_idxs] - segment_length)
    ).to(dtype=torch.long)
    segment_ts = torch.arange(segment_length, device=device) + unseen_t_start[:, None]
    unseen_segment_ys = torch.take_along_dim(
        valid_y[unseen_idxs], segment_ts[:, :, None], dim=1
    )

    np.random.set_state(npr_st)

    n_total_epochs = args.n_total_epochs
    pbar = tqdm(total=n_total_epochs)
    n_total_epochs = args.n_total_epochs
    epochs_elapsed = 0
    batch_size = args.batch_size
    n_microbatches = args.n_microbatches
    model = None

    save_every = n_total_epochs // 10
    eval_every = save_every // 10
    n_warmup_epochs = save_every // 10
    # n_vanilla_epochs = save_every
    n_vanilla_epochs = 0
    # code_reset_every = eval_every
    code_reset_every = 1
    recon_loss_fn = torch.nn.MSELoss()

    while epochs_elapsed <= n_total_epochs:
        quant_yes = epochs_elapsed >= n_vanilla_epochs
        if model is None:
            logger.info(f"Initializing a model")
            # model = MlpFSQVAE(
            #     train_y.shape[-1],
            #     args.hidden_size,
            #     args.latent_size,
            #     args.vocab_size,
            #     args.sentence_length,
            # ).to(device)
            if args.arch == "mlp":
                model = MlpVQVAE(
                    int(segment_length * train_y.shape[-1]),
                    args.hidden_size,
                    args.latent_size,
                    args.vocab_size,
                    args.sentence_length,
                    args.n_convs,
                    args.n_blocks,
                ).to(device)
            elif args.arch == "conv":
                model = ConvVQVAE(
                    train_y.shape[-1],
                    args.hidden_size,
                    args.kernel_size,
                    args.latent_size,
                    args.vocab_size,
                    args.sentence_length,
                    args.segment_length,
                    args.padding,
                    args.n_convs,
                ).to(device)
            elif args.arch == "convfsq":
                model = ConvFSQVAE(
                    train_y.shape[-1],
                    args.hidden_size,
                    args.kernel_size,
                    args.latent_size,
                    args.vocab_size,
                    args.sentence_length,
                    args.segment_length,
                    args.padding,
                    args.n_convs,
                )
            elif args.arch == "mlpfsq":
                model = MlpFSQVAE(
                    int(segment_length * train_y.shape[-1]),
                    args.hidden_size,
                    args.latent_size,
                    args.vocab_size,
                    args.sentence_length,
                    args.n_convs,
                    args.n_blocks,
                )
            elif args.arch == "mlpgs":
                model = MlpGSVAE(
                    int(segment_length * train_y.shape[-1]),
                    args.hidden_size,
                    args.latent_size,
                    args.vocab_size,
                    args.sentence_length,
                    args.n_convs,
                    args.n_blocks,
                )
            else:
                raise NotImplementedError
            model = model.to(device)
            optimizer = RAdam(model.parameters(), lr=args.peak_lr)
            if "gs" not in args.arch:
                old_b = [
                    model.codebooks[j].data.detach().clone()
                    for j in range(model.sentence_length)
                ]

            n_samples = np.minimum(batch_size * 10, 4096)
            idxs = (torch.rand(n_samples) * train_lengths.shape[0]).to(
                dtype=torch.long, device=device
            )
            t_start = (
                torch.rand(n_samples, device=device)
                * (train_lengths[idxs] - segment_length)
            ).to(dtype=torch.long)
            segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
            segment_ys = torch.take_along_dim(
                train_y[idxs], segment_ts[:, :, None], dim=1
            )
            segment_ys = segment_ys.to(device)
            model.setup(segment_ys)

            if not args.debug_yes:
                model = torch.compile(model)

        orig_mod = model._orig_mod if not args.debug_yes else model

        if epochs_elapsed % save_every == 0 or epochs_elapsed >= n_total_epochs:
            pauli_root = os.path.abspath(
                os.path.join(os.getcwd(), "src")
            )  # NOTE: assume that all scripts are run from the parent directory of src.
            model_d = {
                **pauli.dump(orig_mod, pauli_root),
                "model_state_dict": orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs_elapsed": epochs_elapsed,
                "args": args,
            }
            save_path = f"{outdir}/mlp_{epochs_elapsed:06d}.pkl"
            torch.save(model_d, save_path)
            logger.info(f"Saved to {save_path}")

        if epochs_elapsed % eval_every == 0:
            with torch.no_grad():
                x_in = seen_segment_ys
                z_t, k, z_q, x_hat = model.forward(seen_segment_ys, quant_yes=quant_yes)
                recon_loss = recon_loss_fn(x_hat, x_in)

                counts, _ = np.histogram(
                    k[..., 0].reshape(-1).detach().cpu().numpy(),
                    bins=args.vocab_size,
                    range=(0, args.vocab_size),
                )
                n_nonzeros = np.sum(counts > 0)
                writer.add_scalar("train/recon_loss", recon_loss.item(), epochs_elapsed)
                writer.add_scalar("train/nonzero_codes", n_nonzeros, epochs_elapsed)

                if "gs" not in args.arch:
                    code_loss = torch.nn.functional.mse_loss(z_t, z_q.detach())
                    commitment_loss = torch.nn.functional.mse_loss(z_q, z_t.detach())
                    writer.add_scalar(
                        "train/code_loss", code_loss.item(), epochs_elapsed
                    )
                    writer.add_scalar(
                        "train/commitment_loss", commitment_loss.item(), epochs_elapsed
                    )
                    logger.info(
                        f"Seen ReconLoss: {recon_loss.item():.2e}\tCodeLoss: {code_loss.item():.2e}\tCommLoss: {commitment_loss.item():.2e}\t# Nonzero Codes: {n_nonzeros}"
                    )
                else:
                    # q = F.softmax(z_t, dim=-1)
                    # p = torch.ones_like(q) / args.vocab_size
                    # cross_entropy = -torch.sum(q * torch.log(p + 1e-8), dim=-1)
                    # kld_loss = torch.mean(cross_entropy)

                    q = F.softmax(
                        z_t.reshape(
                            z_t.shape[0],
                            z_t.shape[1],
                            args.sentence_length,
                            args.vocab_size,
                        ),
                        dim=-1,
                    )
                    entropy = -torch.sum(q * torch.log(q + 1e-8), dim=-1)
                    max_entropy = np.log(args.vocab_size)
                    kld_loss = max_entropy - torch.mean(entropy)

                    logger.info(
                        f"Seen ReconLoss: {recon_loss.item():.2e}\t# Nonzero Codes: {n_nonzeros}\tKLD Loss: {kld_loss.item():.2e}"
                    )
                x_in = unseen_segment_ys
                z_t, k, z_q, x_hat = model.forward(
                    unseen_segment_ys, quant_yes=quant_yes
                )
                recon_loss = recon_loss_fn(x_hat, x_in)
                counts, _ = np.histogram(
                    k[..., 0].reshape(-1).detach().cpu().numpy(),
                    bins=args.vocab_size,
                    range=(0, args.vocab_size),
                )
                n_nonzeros = np.sum(counts > 0)

                writer.add_scalar("valid/recon_loss", recon_loss.item(), epochs_elapsed)
                if "gs" not in args.arch:
                    code_loss = torch.nn.functional.mse_loss(z_t, z_q.detach())
                    commitment_loss = torch.nn.functional.mse_loss(z_q, z_t.detach())
                    writer.add_scalar(
                        "valid/code_loss", code_loss.item(), epochs_elapsed
                    )
                    writer.add_scalar(
                        "valid/commitment_loss", commitment_loss.item(), epochs_elapsed
                    )
                    logger.info(
                        f"Unseen ReconLoss: {recon_loss.item():.2e}\tCodeLoss: {code_loss.item():.2e}\tCommLoss: {commitment_loss.item():.2e}\t# Nonzero Codes: {n_nonzeros}"
                    )
                else:
                    # q = F.softmax(z_t, dim=-1)
                    # p = torch.ones_like(q) / args.vocab_size
                    # cross_entropy = -torch.sum(q * torch.log(p + 1e-8), dim=-1)
                    # kld_loss = torch.mean(cross_entropy)

                    q = F.softmax(
                        z_t.reshape(
                            z_t.shape[0],
                            z_t.shape[1],
                            args.sentence_length,
                            args.vocab_size,
                        ),
                        dim=-1,
                    )
                    entropy = -torch.sum(q * torch.log(q + 1e-8), dim=-1)
                    max_entropy = np.log(args.vocab_size)
                    kld_loss = max_entropy - torch.mean(entropy)
                    logger.info(
                        f"Unseen ReconLoss: {recon_loss.item():.2e}\t# Nonzero Codes: {n_nonzeros}\tKLD Loss: {kld_loss.item():.2e}"
                    )
                writer.add_scalar("valid/nonzero_codes", n_nonzeros, epochs_elapsed)

        if epochs_elapsed >= n_total_epochs:
            break

        min_lr = 0
        if args.lr_decay_yes:
            if epochs_elapsed % save_every < n_warmup_epochs:
                lr = min_lr + (args.peak_lr - min_lr) * np.clip(
                    (epochs_elapsed % save_every) / n_warmup_epochs, 0, 1
                )
            else:
                # cosine decay
                lr = (
                    min_lr
                    + (args.peak_lr - min_lr)
                    * (
                        1
                        + np.cos(
                            np.pi
                            * (epochs_elapsed % save_every - n_warmup_epochs)
                            / (save_every - n_warmup_epochs)
                        )
                    )
                    / 2
                )
        else:
            lr = args.peak_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        writer.add_scalar("train/lr", lr, epochs_elapsed)

        if args.tau_decay_yes:
            min_tau = 1e-5
            if epochs_elapsed % save_every < n_warmup_epochs:
                tau = min_tau + (args.peak_tau - min_tau) * np.clip(
                    (epochs_elapsed % save_every) / n_warmup_epochs, 0, 1
                )
            else:
                # cosine decay
                tau = (
                    min_tau
                    + (args.peak_tau - min_tau)
                    * (
                        1
                        + np.cos(
                            np.pi
                            * (epochs_elapsed % save_every - n_warmup_epochs)
                            / (save_every - n_warmup_epochs)
                        )
                    )
                    / 2
                )
            model.tau = tau

        epoch_ks = []
        for _ in range(args.n_steps_per_epoch):
            idxs = (torch.rand(batch_size) * train_lengths.shape[0]).to(
                dtype=torch.long, device=device
            )
            t_start = (
                torch.rand(batch_size, device=device)
                * (train_lengths[idxs] - segment_length)
            ).to(dtype=torch.long)
            segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
            segment_ys = torch.take_along_dim(
                train_y[idxs], segment_ts[:, :, None], dim=1
            )
            segment_ys = segment_ys.to(device)
            dataset = ThroughDataset(segment_ys)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            dropout_mask = torch.zeros(
                model.sentence_length, dtype=torch.bool, device=device
            )
            if model.sentence_length > 1:
                n = np.random.randint(1, model.sentence_length)
                dropout_mask[:n] = True
            else:
                dropout_mask[:] = True
            for (batch_x_in,) in dataloader:
                microbatches_x_in = torch.tensor_split(batch_x_in, n_microbatches)
                optimizer.zero_grad()
                for (x_in,) in zip(microbatches_x_in):
                    z_t, k, z_q, x_hat = model.forward(x_in, dropout_mask, quant_yes)
                    epoch_ks.append(k)
                    recon_loss = recon_loss_fn(x_hat, x_in)
                    if "gs" not in args.arch:
                        # code_loss = torch.nn.functional.mse_loss(z_q, z_t.detach())
                        commitment_loss = torch.nn.functional.mse_loss(
                            z_t, z_q.detach()
                        )
                        loss = (
                            recon_loss
                            # + args.code_loss_weight * code_loss
                            + args.commitment_loss_weight * commitment_loss
                        )
                    else:
                        q = F.softmax(
                            z_t.reshape(
                                z_t.shape[0],
                                z_t.shape[1],
                                args.sentence_length,
                                args.vocab_size,
                            ),
                            dim=-1,
                        )
                        # # q = F.softmax(z_t, dim=-1)
                        # # p = torch.ones_like(q) / args.vocab_size
                        # entropy = -torch.sum(q * torch.log(q), dim=-1)
                        # # cross_entropy = -torch.sum(q * torch.log(p), dim=-1)
                        # kld_loss = -torch.mean(entropy)

                        q = F.softmax(
                            z_t.reshape(
                                z_t.shape[0],
                                z_t.shape[1],
                                args.sentence_length,
                                args.vocab_size,
                            ),
                            dim=-1,
                        )
                        entropy = -torch.sum(q * torch.log(q + 1e-8), dim=-1)
                        max_entropy = np.log(args.vocab_size)
                        kld_loss = max_entropy - torch.mean(entropy)
                        loss = recon_loss + args.kld_weight * kld_loss
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if "gs" not in args.arch:
            # EMA process for codebook
            gamma = 0.99
            for j in range(model.sentence_length):
                model.codebooks[j].data = (1 - gamma) * model.codebooks[
                    j
                ].data + gamma * old_b[j]

        if epochs_elapsed == n_vanilla_epochs - 1:
            n_samples = np.minimum(batch_size * 10, 4096)
            idxs = (torch.rand(n_samples) * train_lengths.shape[0]).to(
                dtype=torch.long, device=device
            )
            t_start = (
                torch.rand(n_samples, device=device)
                * (train_lengths[idxs] - segment_length)
            ).to(dtype=torch.long)
            segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
            segment_ys = torch.take_along_dim(
                train_y[idxs], segment_ts[:, :, None], dim=1
            )
            segment_ys = segment_ys.to(device)
            # model.setup2(segment_ys)

        if (
            "fsq" not in args.arch
            and "gs" not in args.arch
            and epochs_elapsed >= n_vanilla_epochs
            and epochs_elapsed % code_reset_every == 0
        ):
            # n_samples = np.minimum(batch_size * 10, 4096)
            # idxs = (torch.rand(n_samples) * train_lengths.shape[0]).to(
            #     dtype=torch.long, device=device
            # )
            # t_start = (
            #     torch.rand(n_samples, device=device)
            #     * (train_lengths[idxs] - segment_length)
            # ).to(dtype=torch.long)
            # segment_ts = (
            #     torch.arange(segment_length, device=device) + t_start[:, None]
            # )
            # segment_ys = torch.take_along_dim(
            #     train_y[idxs], segment_ts[:, :, None], dim=1
            # )
            # segment_ys = segment_ys.to(device)
            # model.setup(segment_ys)

            # with torch.no_grad():
            #     epoch_ks = []
            #     for _ in range(100):
            #         idxs = (torch.rand(batch_size) * train_lengths.shape[0]).to(
            #             dtype=torch.long, device=device
            #         )
            #         t_start = (
            #             torch.rand(batch_size, device=device)
            #             * (train_lengths[idxs] - segment_length)
            #         ).to(dtype=torch.long)
            #         segment_ts = (
            #             torch.arange(segment_length, device=device) + t_start[:, None]
            #         )
            #         segment_ys = torch.take_along_dim(
            #             train_y[idxs], segment_ts[:, :, None], dim=1
            #         )
            #         segment_ys = segment_ys.to(device)
            #         z_t, k, z_q, x_hat = model.forward(segment_ys)
            #         epoch_ks.append(k)

            # Code resetting
            # epoch_ks = k
            epoch_ks = torch.cat(epoch_ks, dim=0)
            # epoch_ks = epoch_ks.reshape(-1, epoch_ks.shape[-1])
            quantized = torch.zeros_like(z_t)
            for i in range(epoch_ks.shape[-1]):
                counts = torch.histc(
                    epoch_ks[..., :, i],
                    bins=args.vocab_size,
                    min=0,
                    max=args.vocab_size,
                )
                zeros_yes = counts == 0
                n_zeros = zeros_yes.sum().item()
                residuals = z_t - quantized
                residuals = residuals.reshape(-1, residuals.shape[-1])
                assign_idxs = (
                    torch.rand(n_zeros, device=z_t.device) * residuals.shape[0]
                ).to(torch.long)
                # assign_idxs = torch.randperm(residuals.shape[0], device=device)[
                #     :np.minimum(n_zeros, residuals.shape[0])
                # ]
                # model.codebooks[i].data[zeros_yes] = residuals[assign_idxs]
                model.codebooks[i].data[zeros_yes] = residuals[assign_idxs]
                quantized += model.codebooks[i].data[k[..., :, i]]
                # logger.info(f"Codebook {i} has {n_zeros} zeros")
                # print(n_zeros)

            # # Perform a check to see if n_zeros has gone down
            # z_t, k, z_q, x_hat = model.forward(segment_ys)
            # for i in range(k.shape[-1]):
            #     counts = torch.histc(
            #         k[..., :, i],
            #         bins=args.vocab_size,
            #         min=0,
            #         max=args.vocab_size,
            #     )
            #     zeros_yes = counts == 0
            #     n_zeros = zeros_yes.sum()
            #     print(n_zeros)

        if "gs" not in args.arch:
            old_b = [
                model.codebooks[j].data.detach().clone()
                for j in range(model.sentence_length)
            ]

        epochs_elapsed += 1
        pbar.update(1)
        pbar.set_postfix({"loss": loss.item(), "epochs": epochs_elapsed})

    pbar.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_total_epochs", type=int, default=int(3e4))
    parser.add_argument("--n_steps_per_epoch", type=int, default=10)
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--peak_tau", type=float, default=1)
    parser.add_argument("--lr_decay_yes", action="store_true")
    parser.add_argument("--tau_decay_yes", action="store_true")
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--cached_yes", action="store_true")
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--sentence_length", type=int, default=1)
    parser.add_argument("--code_loss_weight", type=float, default=1.0)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
    parser.add_argument("--n_microbatches", type=int, default=1)
    parser.add_argument("--segment_length", type=int, default=1)
    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["mlp", "conv", "mlpfsq", "convfsq", "mlpgs"],
    )
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--n_convs", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--kld_weight", type=float, default=1.0)
    args = parser.parse_args()

    main()
