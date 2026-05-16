import copy
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import pauli
from beaty_common.bsmg_xror_utils import load_cbo_and_3p
from beaty_common.data_utils import SegmentSampler
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train.torch_nets2 import CondTransformerGSVAE, TransformerGSVAE
from beaty_common.train_utils import nanpad_collate_fn, collect_rollout
from xror.xror import XROR

warnings.filterwarnings("ignore")

torch._dynamo.config.optimize_ddp = False
torch.set_warn_always(False)

device = torch.device("cuda")


def main(args):
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    outdir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    batch_size = args.batch_size
    minibatch_size = args.minibatch_size
    segment_length = args.segment_length
    chunk_length = args.chunk_length
    segment_sampler_batch_size = 64
    stride = args.stride
    n_total_batches = args.n_total_batches
    matching_weight = args.matching_weight
    history_len = args.history_len

    save_every = n_total_batches // 10
    eval_every = save_every // 10
    n_warmup_batches = save_every // 10

    umds_df = pd.read_csv(f"{proj_dir}/data/boxrr_umds_v15.csv")
    _, unique_idxs = np.unique(umds_df["HF Index"].values, return_index=True)
    umds_df = umds_df.iloc[unique_idxs]
    ninety_quantile_yes = umds_df["Map-Difficulty Count"] > umds_df["Map-Difficulty Count"].quantile(0.9)
    umds_df = umds_df[~ninety_quantile_yes]
    umds_df["Song Hash and Difficulty"] = umds_df["Song Hash"] + umds_df["Difficulty Level"]
    R_df = pd.read_csv(f"{proj_dir}/data/R_v15.csv")
    R_df["Song Hash and Difficulty"] = R_df["Song Hash"] + R_df["Difficulty Level"]
    R_yes = umds_df["Song Hash and Difficulty"].isin(R_df["Song Hash and Difficulty"])
    umds_df = umds_df[~R_yes]
    bomb_yes = umds_df["bombs"] > 0
    obst_yes = umds_df["obstacles"] > 0
    umds_df = umds_df.loc[bomb_yes | obst_yes]
    hf_idxs = umds_df["HF Index"].values

    train_dataset = load_dataset("cschell/boxrr-23", cache_dir=f"{proj_dir}/datasets/boxrr-23", streaming=True, split="train")
    train_dataset = train_dataset.filter(lambda _, idx: idx in hf_idxs, with_indices=True)
    train_dataset = train_dataset.map(lambda example: load_cbo_and_3p(XROR.unpack(example["xror"]), True), remove_columns=["__key__", "__url__", "xror", "xror_info"])
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=nanpad_collate_fn, pin_memory=True, num_workers=num_workers, shuffle=False)
    train_dataloader_iter = iter(train_dataloader)  # makes sense to use this if dataset size is fixed, but we have on-the-fly validation
    segment_sampler = SegmentSampler()

    # Set aside evaluation set
    npr_st = np.random.get_state()
    np.random.seed(0)
    d = next(train_dataloader_iter)
    for k, v in d.items():
        d[k] = v.pin_memory().to(device=device)
    seen_game_segments, seen_movement_segments = segment_sampler.sample_for_training(
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        d["timestamps"],
        d["gt_3p_np"],
        d["lengths"],
        segment_length,
        minibatch_size,
        segment_sampler_batch_size,
        stride,
        2.0,
        40,
        -0.5,
    )
    seen_segment_ys = seen_movement_segments.three_p
    np.random.set_state(npr_st)

    pbar = tqdm(total=n_total_batches)

    # Initialize model
    if args.checkpoint_path is None:
        logger.info(f"Initializing a model")
        if args.arch == "tftf":
            gsvae_net = TransformerGSVAE(
                seen_segment_ys.shape[-1],
                args.hidden_size,
                args.embed_size,
                args.vocab_size,
                args.sentence_length,
                args.chunk_length,
                args.stride,
                args.num_heads,
                args.num_layers,
            )
            pred_net = CondTransformerGSVAE(
                seen_game_segments.notes.shape[-1],
                seen_game_segments.bombs.shape[-1],
                seen_game_segments.obstacles.shape[-1],
                seen_movement_segments.three_p.shape[-1],
                args.hidden_size,
                args.hidden_size,
                args.sentence_length,
                args.vocab_size,
                args.num_heads,
                args.num_layers,
            )
        else:
            raise NotImplementedError
        gsvae_net = gsvae_net.to(device)
        pred_net = pred_net.to(device)
        parameters = []
        parameters.extend(pred_net.parameters())
        parameters.extend(gsvae_net.parameters())
        optimizer = AdamW(parameters, lr=args.peak_lr)

        gsvae_net.setup(seen_movement_segments)
        pred_net.setup(seen_game_segments, seen_movement_segments)

        # EMA
        gsvae_ema = copy.deepcopy(gsvae_net).to(device=device).eval().requires_grad_(False)
        pred_ema = copy.deepcopy(pred_net).to(device=device).eval().requires_grad_(False)

    else:
        logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
        checkpoint_dict = torch.load(args.checkpoint_path, weights_only=False)

        pred_net_d = checkpoint_dict["pred_net_d"]
        pred_net = pauli.load(pred_net_d)
        pred_net_state_dict = checkpoint_dict["pred_net_state_dict"]
        for k in list(pred_net_state_dict.keys()):
            kk = k.replace("_orig_mod.", "")
            pred_net_state_dict[kk] = pred_net_state_dict.pop(k)
        pred_net.load_state_dict(pred_net_state_dict)
        pred_net = pred_net.to(device)
        pred_net.requires_grad_(True)
        pred_net.note_rms.requires_grad_(False)
        pred_net.bomb_rms.requires_grad_(False)
        pred_net.obstacle_rms.requires_grad_(False)
        pred_net.history_rms.requires_grad_(False)

        gsvae_dict = checkpoint_dict["gsvae_net_d"]
        gsvae_net = pauli.load(gsvae_dict)
        gsvae_net_state_dict = checkpoint_dict["gsvae_net_state_dict"]
        for k in list(gsvae_net_state_dict.keys()):
            kk = k.replace("_orig_mod.", "")
            gsvae_net_state_dict[kk] = gsvae_net_state_dict.pop(k)
        gsvae_net.load_state_dict(gsvae_net_state_dict)
        gsvae_net = gsvae_net.to(device)
        gsvae_net.requires_grad_(True)
        gsvae_net.input_rms.requires_grad_(False)

        # EMA
        gsvae_ema = copy.deepcopy(gsvae_net).to(device=device).eval().requires_grad_(False)
        pred_ema = copy.deepcopy(pred_net).to(device=device).eval().requires_grad_(False)

        parameters = []
        parameters.extend(pred_net.parameters())
        parameters.extend(gsvae_net.parameters())
        optimizer = AdamW(parameters, lr=args.peak_lr)

        if args.continue_yes:
            gsvae_ema_state_dict = checkpoint_dict["gsvae_ema_state_dict"]
            for k in list(gsvae_ema_state_dict.keys()):
                kk = k.replace("_orig_mod.", "")
                gsvae_ema_state_dict[kk] = gsvae_ema_state_dict.pop(k)
            gsvae_ema.load_state_dict(gsvae_ema_state_dict)

            pred_ema_state_dict = checkpoint_dict["pred_ema_state_dict"]
            for k in list(pred_ema_state_dict.keys()):
                kk = k.replace("_orig_mod.", "")
                pred_ema_state_dict[kk] = pred_ema_state_dict.pop(k)
            pred_ema.load_state_dict(pred_ema_state_dict)

            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            batches_elapsed = checkpoint_dict["batches_elapsed"]
            pbar.update(batches_elapsed)

    if not args.debug_yes:
        pred_net = torch.compile(pred_net)
        gsvae_net = torch.compile(gsvae_net)

    pred_orig_mod = pred_net._orig_mod if not args.debug_yes else pred_net
    gsvae_orig_mod = gsvae_net._orig_mod if not args.debug_yes else gsvae_net

    # Main training loop
    batches_elapsed = 0
    samples_elapsed = 0
    while True:
        if batches_elapsed % save_every == 0 or batches_elapsed >= n_total_batches:
            pauli_root = os.path.abspath(os.path.join(os.getcwd(), "src"))  # NOTE: assume that all scripts are run from the parent directory of src.
            checkpoint_dict = {
                "pred_net_d": pauli.dump(pred_orig_mod, pauli_root),
                "gsvae_net_d": pauli.dump(gsvae_orig_mod, pauli_root),
                "pred_net_state_dict": pred_net.state_dict(),
                "pred_ema_state_dict": pred_ema.state_dict(),
                "gsvae_net_state_dict": gsvae_orig_mod.state_dict(),
                "gsvae_ema_state_dict": gsvae_ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "batches_elapsed": batches_elapsed,
                "samples_elapsed": samples_elapsed,
                "args": args,
            }
            save_path = f"{outdir}/codebook_matching_{samples_elapsed // 1000:06d}.pkl"
            torch.save(checkpoint_dict, save_path)
            logger.info(f"Saved to {save_path}")

        if batches_elapsed % eval_every == 0:
            losses, pred_losses, recon_losses, matching_losses = collect_rollout(
                pred_ema,
                gsvae_ema,
                pred_ema,
                gsvae_ema,
                optimizer,
                seen_game_segments,
                seen_movement_segments,
                history_len,
                chunk_length,
                matching_weight,
                False,
                args.matching_loss_type,
                args.n_cands,
            )
            pred_loss = pred_losses.mean()
            matching_loss = matching_losses.mean()
            recon_loss = recon_losses.mean()

            logger.info(f"Batch {batches_elapsed} ksamples {samples_elapsed // 1000} Seen PredLoss: {pred_loss.item():.2e} MatchingLoss: {matching_loss.item():.2e} ReconLoss: {recon_loss.item():.2e}")
            for unit, value in [["batches", batches_elapsed], ["ksamples", samples_elapsed // 1000]]:
                writer.add_scalar(f"train/pred_loss/{unit}", pred_loss.item(), value)
                writer.add_scalar(f"train/codebook_loss/{unit}", matching_loss.item(), value)
                writer.add_scalar(f"train/recon_loss/{unit}", recon_loss.item(), value)

        if batches_elapsed >= n_total_batches:
            break

        # Learning rate decay
        min_lr = 0
        if args.lr_decay_yes:
            if batches_elapsed % save_every < n_warmup_batches:
                lr = min_lr + (args.peak_lr - min_lr) * np.clip((batches_elapsed % save_every) / n_warmup_batches, 0, 1)
            else:
                # cosine decay
                lr = min_lr + (args.peak_lr - min_lr) * (1 + np.cos(np.pi * (batches_elapsed % save_every - n_warmup_batches) / (save_every - n_warmup_batches))) / 2
        else:
            lr = args.peak_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        writer.add_scalar("train/lr/batches", lr, batches_elapsed)
        writer.add_scalar("train/lr/ksamples", lr, samples_elapsed)

        # Load training data here
        d = next(train_dataloader_iter)
        if d is not None:
            for k, v in d.items():
                d[k] = v.pin_memory().to(device=device)
            game_segments, movement_segments = segment_sampler.sample_for_training(
                d["notes_np"],
                d["bombs_np"],
                d["obstacles_np"],
                d["timestamps"],
                d["gt_3p_np"],
                d["lengths"],
                segment_length,
                minibatch_size,
                segment_sampler_batch_size,
                stride,
                2.0,
                40,
                -0.5,
            )

            losses, pred_losses, recon_losses, matching_losses = collect_rollout(
                pred_net,
                gsvae_net,
                pred_ema,
                gsvae_ema,
                optimizer,
                game_segments,
                movement_segments,
                history_len,
                chunk_length,
                matching_weight,
                True,
                args.matching_loss_type,
                args.n_cands,
            )

        batches_elapsed += 1
        samples_elapsed += batch_size
        pbar.update(1)
        pbar.set_postfix({"recon_loss": recon_losses.mean().item(), "matching_loss": matching_losses.mean().item(), "batches": batches_elapsed})

    pbar.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--minibatch_size", type=int, default=512)
    parser.add_argument("--n_total_batches", type=int, default=8500)
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--peak_lr", type=float, default=5e-5)
    parser.add_argument("--lr_decay_yes", action="store_true")
    parser.add_argument("--cached_yes", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--sentence_length", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=8)
    parser.add_argument("--segment_length", type=int, default=270)
    parser.add_argument("--chunk_length", type=int, default=64)
    parser.add_argument("--n_cands", type=int, default=1)
    parser.add_argument("--history_len", type=int, default=2)
    parser.add_argument("--matching_loss_type", type=str, default="mse", choices=["mse", "jsd"])
    parser.add_argument("--matching_weight", type=float, default=4e-4)
    parser.add_argument("--arch", type=str, default="tftf", choices=["tftf"])
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    main(args)
