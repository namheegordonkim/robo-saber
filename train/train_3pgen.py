import copy
import glob
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

try:
    import pynvml  # from nvidia-ml-py package

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

import pauli
from beaty_common.data_utils import SegmentSampler
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train.torch_nets2 import CondTransformerGSVAE, TransformerGSVAE, ConditionalEDM
from beaty_common.train_utils import (
    collect_rollout,
    xror_to_tensor_collate_fn,
    BoxrrCacheDataset,
    cache_collate_fn,
)

warnings.filterwarnings("ignore")

torch._dynamo.config.optimize_ddp = False
torch.set_warn_always(False)

device = torch.device("cuda")


def get_gpu_metrics(device_id=0):
    """Get GPU utilization and VRAM usage metrics."""
    metrics = {}

    # Memory metrics from PyTorch (in MB)
    metrics["vram_allocated_mb"] = torch.cuda.memory_allocated(device_id) / 1024**2
    metrics["vram_reserved_mb"] = torch.cuda.memory_reserved(device_id) / 1024**2
    metrics["vram_max_allocated_mb"] = torch.cuda.max_memory_allocated(device_id) / 1024**2

    # GPU utilization from NVML if available
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_utilization_pct"] = util.gpu
            metrics["gpu_memory_utilization_pct"] = util.memory

            # Total and used memory from NVML (in MB)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["vram_total_mb"] = mem_info.total / 1024**2
            metrics["vram_used_mb"] = mem_info.used / 1024**2
        except Exception as e:
            pass

    return metrics


def main(args, remaining_args):
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
    writer.add_text("remaining_args", str(remaining_args))

    batch_size = args.batch_size
    minibatch_size = args.minibatch_size
    segment_length = args.segment_length
    chunk_length = args.chunk_length
    segment_sampler_batch_size = 64
    stride = args.stride
    n_total_batches = args.n_total_batches
    matching_weight = args.matching_weight
    history_len = args.history_len
    n_segs = args.n_segs

    # Task scale applied
    n_total_batches = max(int(n_total_batches * args.task_scale), 1)

    save_every = max(1, args.save_every if args.save_every is not None else n_total_batches // 20)
    eval_every = max(1, args.eval_every if args.eval_every is not None else save_every // 20)
    n_warmup_batches = max(1, save_every // 20)

    checkpoint_dict = {}
    if args.checkpoint_path is not None:
        logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
        checkpoint_dict = torch.load(args.checkpoint_path, weights_only=False)
        n_segs = checkpoint_dict["args"].n_segs
        logger.info(f"Using {n_segs=}")
        args.n_segs = n_segs

    writer.add_text("args", str(args))

    if args.jab:
        umds_df = pd.read_csv(f"{proj_dir}/data/boxrr_umds_v15.csv")
        # _, unique_idxs = np.unique(umds_df["HF Index"].values, return_index=True)
        #
        # umds_df = umds_df.iloc[unique_idxs]
        # _, unique_idxs = np.unique(umds_df["Song Hash and Difficulty"].values, return_index=True)
        # ninety_quantile_yes = umds_df["Map-Difficulty Count"] >= np.sort(umds_df.iloc[unique_idxs]["Map-Difficulty Count"])[-300]
        #
        # heldout_df = umds_df[ninety_quantile_yes]
        # _, unique_idxs = np.unique(heldout_df["Song Hash and Difficulty"].values, return_index=True)
        # heldout_df = heldout_df.iloc[unique_idxs]
        # heldout_df.to_csv(f"{proj_dir}/data/heldout.csv", index=False)
        #
        # umds_df = pd.read_csv(f"{proj_dir}/data/uptown_funk.csv")
        # hf_idxs = umds_df["HF Index"].values
        umds_df = umds_df[umds_df["Song Hash and Difficulty"] == "D110E413FB7FB462B692F1F17B835CF8B7280884Expert"]
        umds_df = umds_df.iloc[[0]]
        hf_idxs = umds_df["HF Index"].values
        shard_idxs = umds_df["Shard Index"].values
        hf_idxs = datapoint_idxs = umds_df["Datapoint Index"].values
        shard_names = sorted(
            glob.glob(
                f"{proj_dir}/datasets/boxrr-23/cschell___boxrr-23/default/0.0.0/**/*.arrow",
                recursive=True,
            )
        )
        train_dataset = Dataset.from_file(shard_names[shard_idxs[0]], split="train")
        train_dataset = train_dataset.filter(lambda _, idx: idx in hf_idxs, with_indices=True)
        # train_dataset = train_dataset.map(lambda example: open_unpacked_xror(XROR.unpack(example["xror"]), True), remove_columns=["__key__", "__url__", "xror"])
        num_workers = 0
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, True, batch_size),
            batch_size=batch_size,
            collate_fn=xror_to_tensor_collate_fn,
            pin_memory=True,
            num_workers=num_workers,
        )

    else:
        umds_df = pd.read_csv(f"{proj_dir}/data/boxrr_umds_v15.csv")
        heldout_df = pd.read_csv(f"{proj_dir}/data/heldout.csv")
        yes = ~umds_df["Song Hash and Difficulty"].isin(heldout_df["Song Hash and Difficulty"])
        umds_df = umds_df[yes]

        R_df = pd.read_csv(f"{proj_dir}/data/R_v16.csv")
        R_yes = umds_df["Song Hash and Difficulty"].isin(R_df["Song Hash and Difficulty"])
        umds_df = umds_df[~R_yes]

        # bomb_yes = umds_df["bombs"] > 0
        # obst_yes = umds_df["obstacles"] > 0
        # umds_df = umds_df.loc[bomb_yes | obst_yes]

        hf_idxs = umds_df["HF Index"].values
        shuffle_idxs = np.random.permutation(len(hf_idxs))
        sf_pairs = umds_df.iloc[shuffle_idxs][["Shard Index", "Datapoint Index"]].values
        # hf_idxs = np.random.permutation(hf_idxs)

        # train_dataset = BoxrrHFDataset(sf_pairs, f"{proj_dir}/datasets/boxrr-23", logger)
        train_dataset = BoxrrCacheDataset(f"{proj_dir}/{args.data_dir}", batch_size=batch_size, init_i=0, mode="main")
        # train_dataset = load_dataset("cschell/boxrr-23", cache_dir=f"{proj_dir}/datasets/boxrr-23", streaming=True, split="train")
        # train_dataset = train_dataset.filter(lambda _, idx: idx in hf_idxs, with_indices=True)
        # train_dataset = train_dataset.map(lambda example: open_unpacked_xror(XROR.unpack(example["xror"]), True), remove_columns=["__key__", "__url__", "xror"])
        num_workers = 0
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=xror_to_tensor_collate_fn, pin_memory=True, num_workers=num_workers)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=None,
            collate_fn=cache_collate_fn,
            pin_memory=True,
            num_workers=num_workers,
        )
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=nanpad_collate_fn, pin_memory=True, num_workers=num_workers)
        # sampler = BoxrrCacheSampler(train_dataset, batch_size=batch_size, shuffle=False)
        # train_dataloader = DataLoader(train_dataset, sampler=sampler, collate_fn=nanpad_collate_fn, pin_memory=True, num_workers=num_workers)

    train_dataloader_iter = iter(train_dataloader)  # makes sense to use this if dataset size is fixed, but we have on-the-fly validation
    segment_sampler = SegmentSampler()

    # Set aside evaluation set
    npr_st = np.random.get_state()
    np.random.seed(0)
    if args.jab:
        d = next(iter(train_dataloader))
    else:
        d = next(train_dataloader_iter)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.pin_memory().to(device=device)

    # purview_notes = 20  # used to be 40
    # floor_time = 0  # used to be -0.5
    # seen_game_segments, seen_movement_segments = segment_sampler.sample_for_training(
    #     d["notes_np"],
    #     d["bombs_np"],
    #     d["obstacles_np"],
    #     d["timestamps"],
    #     d["gt_3p_np"],
    #     d["lengths"],
    #     segment_length,
    #     minibatch_size,
    #     segment_sampler_batch_size,
    #     stride,
    #     2.0,
    #     purview_notes,
    #     floor_time,
    # )
    # seen_segment_ys = seen_movement_segments.three_p
    seen_notes = d["notes"]
    seen_bombs = d["bombs"]
    seen_obstacles = d["obstacles"]
    seen_frames = d["t"]
    seen_my_3p = d["my_3p"]
    seen_history = d["history"]

    # Randomly choose among segments which one is the game and which ones are the playstyle
    seen_game_seg_idxs = torch.randint(0, n_segs, size=(seen_notes.shape[0],), device=device)
    seen_ref_seg_idxs = torch.arange(n_segs, device=device)[None, :].repeat(seen_notes.shape[0], 1)
    seen_ref_seg_idxs = seen_ref_seg_idxs[seen_ref_seg_idxs != seen_game_seg_idxs[:, None]].view(seen_notes.shape[0], n_segs - 1)

    seen_game_notes = torch.take_along_dim(seen_notes, seen_game_seg_idxs[:, None, None, None], dim=1)[:, 0]
    seen_game_bombs = torch.take_along_dim(seen_bombs, seen_game_seg_idxs[:, None, None, None], dim=1)[:, 0]
    seen_game_obstacles = torch.take_along_dim(seen_obstacles, seen_game_seg_idxs[:, None, None, None], dim=1)[:, 0]
    seen_game_frames = torch.take_along_dim(seen_frames, seen_game_seg_idxs[:, None], dim=1)[:, 0]
    seen_game_3p = torch.take_along_dim(seen_my_3p, seen_game_seg_idxs[:, None, None, None], dim=1)[:, 0]
    seen_game_history = torch.take_along_dim(seen_history, seen_game_seg_idxs[:, None, None, None], dim=1)[:, 0]

    seen_playstyle_notes = torch.take_along_dim(seen_notes, seen_ref_seg_idxs[:, :, None, None], dim=1)
    seen_playstyle_bombs = torch.take_along_dim(seen_bombs, seen_ref_seg_idxs[:, :, None, None], dim=1)
    seen_playstyle_obstacles = torch.take_along_dim(seen_obstacles, seen_ref_seg_idxs[:, :, None, None], dim=1)
    seen_playstyle_frames = torch.take_along_dim(seen_frames, seen_ref_seg_idxs, dim=1)
    seen_playstyle_3p = torch.take_along_dim(seen_my_3p, seen_ref_seg_idxs[:, :, None, None], dim=1)
    seen_playstyle_history = torch.take_along_dim(seen_history, seen_ref_seg_idxs[:, :, None, None], dim=1)

    np.random.set_state(npr_st)

    pbar = tqdm(total=n_total_batches)
    # Main training loop
    batches_elapsed = 0
    samples_elapsed = 0

    # Initialize model
    if args.checkpoint_path is None:
        logger.info(f"Initializing a model")
        parameters = []
        if args.arch == "ccm":
            gsvae_net = TransformerGSVAE(
                seen_my_3p.shape[-1],
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
                seen_notes.shape[-1],
                seen_bombs.shape[-1],
                seen_obstacles.shape[-1],
                seen_history.shape[-1],
                args.hidden_size,
                args.hidden_size,
                args.sentence_length,
                args.vocab_size,
                args.num_heads,
                args.num_layers,
            )
            gsvae_net = gsvae_net.to(device)
            pred_net = pred_net.to(device)
            parameters.extend(pred_net.parameters())
            parameters.extend(gsvae_net.parameters())

        elif args.arch == "ad":
            net = ConditionalEDM(
                input_size=seen_segments.shape[-1],
                cond_size=seen_segments.shape[-1],
                hidden_size=args.hidden_size,
                embed_size=args.hidden_size,
                history_length=args.history_length,
                chunk_length=args.chunk_length,
                stride=args.stride,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                sigma_max=args.sigma_max,
                sigma_data=args.sigma_data,
            )
            parameters.extend(net.parameters())

        else:
            raise NotImplementedError
        optimizer = AdamW(parameters, lr=args.peak_lr)

        gsvae_net.setup(seen_my_3p)
        pred_net.setup(seen_notes, seen_bombs, seen_obstacles, seen_history)

        # EMA
        gsvae_ema = copy.deepcopy(gsvae_net).to(device=device).eval().requires_grad_(False)
        pred_ema = copy.deepcopy(pred_net).to(device=device).eval().requires_grad_(False)

    else:

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
        pred_net.threep_rms.requires_grad_(False)

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
        if args.continue_yes:
            batches_elapsed = checkpoint_dict["batches_elapsed"]
            samples_elapsed = checkpoint_dict["samples_elapsed"]
            pbar.update(batches_elapsed)

    if not args.debug_yes:
        pred_net = torch.compile(pred_net)
        gsvae_net = torch.compile(gsvae_net)

    while True:
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

        if batches_elapsed % save_every == 0 or batches_elapsed >= n_total_batches:
            pred_orig_mod = pred_net._orig_mod if not args.debug_yes else pred_net
            gsvae_orig_mod = gsvae_net._orig_mod if not args.debug_yes else gsvae_net
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
                seen_game_notes,
                seen_game_bombs,
                seen_game_obstacles,
                seen_game_history,
                seen_game_frames,
                seen_game_3p,
                seen_playstyle_notes,
                seen_playstyle_bombs,
                seen_playstyle_obstacles,
                seen_playstyle_history,
                seen_playstyle_frames,
                seen_playstyle_3p,
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

            # Get GPU metrics
            gpu_metrics = get_gpu_metrics(device.index if device.index is not None else 0)

            gpu_info_str = f"VRAM: {gpu_metrics.get('vram_allocated_mb', 0):.0f}MB"
            if "gpu_utilization_pct" in gpu_metrics:
                gpu_info_str += f" GPU: {gpu_metrics['gpu_utilization_pct']:.1f}%"
            if "gpu_memory_utilization_pct" in gpu_metrics:
                gpu_info_str += f" MemUtil: {gpu_metrics['gpu_memory_utilization_pct']:.1f}%"

            logger.info(f"Batch {batches_elapsed} ksamples {samples_elapsed // 1000} Seen PredLoss: {pred_loss.item():.2e} MatchingLoss: {matching_loss.item():.2e} ReconLoss: {recon_loss.item():.2e} {gpu_info_str}")
            for unit, value in [
                ["batches", batches_elapsed],
                ["ksamples", samples_elapsed // 1000],
            ]:
                writer.add_scalar(f"train/pred_loss/{unit}", pred_loss.item(), value)
                writer.add_scalar(f"train/codebook_loss/{unit}", matching_loss.item(), value)
                writer.add_scalar(f"train/recon_loss/{unit}", recon_loss.item(), value)
                writer.add_scalar(f"train/lr/{unit}", lr, value)

                # Log GPU metrics only for ksamples
                if unit == "ksamples":
                    writer.add_scalar(
                        f"gpu/vram_allocated_mb/{unit}",
                        gpu_metrics.get("vram_allocated_mb", 0),
                        value,
                    )
                    writer.add_scalar(
                        f"gpu/vram_reserved_mb/{unit}",
                        gpu_metrics.get("vram_reserved_mb", 0),
                        value,
                    )
                    writer.add_scalar(
                        f"gpu/vram_max_allocated_mb/{unit}",
                        gpu_metrics.get("vram_max_allocated_mb", 0),
                        value,
                    )
                    if "gpu_utilization_pct" in gpu_metrics:
                        writer.add_scalar(
                            f"gpu/utilization_pct/{unit}",
                            gpu_metrics["gpu_utilization_pct"],
                            value,
                        )
                        writer.add_scalar(
                            f"gpu/memory_utilization_pct/{unit}",
                            gpu_metrics["gpu_memory_utilization_pct"],
                            value,
                        )
                    if "vram_used_mb" in gpu_metrics:
                        writer.add_scalar(
                            f"gpu/vram_used_mb/{unit}",
                            gpu_metrics["vram_used_mb"],
                            value,
                        )
                        writer.add_scalar(
                            f"gpu/vram_total_mb/{unit}",
                            gpu_metrics["vram_total_mb"],
                            value,
                        )

        if batches_elapsed >= n_total_batches:
            break

        # Load training data here
        if args.jab:
            d = next(iter(train_dataloader))
        else:
            d = next(train_dataloader_iter)
        if d is not None:
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.pin_memory().to(device=device)

                # game_segments, movement_segments = segment_sampler.sample_for_training(
                #     d["notes_np"],
                #     d["bombs_np"],
                #     d["obstacles_np"],
                #     d["timestamps"],
                #     d["gt_3p_np"],
                #     d["lengths"],
                #     segment_length,
                #     minibatch_size,
                #     segment_sampler_batch_size,
                #     stride,
                #     2.0,
                #     40,
                #     -0.5,
                # )
        notes = d["notes"]
        bombs = d["bombs"]
        obstacles = d["obstacles"]
        frames = d["t"]
        my_3p = d["my_3p"]
        history = d["history"]

        # Randomly choose among segments which one is the game and which ones are the playstyle
        game_seg_idxs = torch.randint(0, n_segs, size=(notes.shape[0],), device=device)
        ref_seg_idxs = torch.arange(n_segs, device=device)[None, :].repeat(notes.shape[0], 1)
        ref_seg_idxs = ref_seg_idxs[ref_seg_idxs != game_seg_idxs[:, None]].view(notes.shape[0], n_segs - 1)

        game_notes = torch.take_along_dim(notes, game_seg_idxs[:, None, None, None], dim=1)[:, 0]
        game_bombs = torch.take_along_dim(bombs, game_seg_idxs[:, None, None, None], dim=1)[:, 0]
        game_obstacles = torch.take_along_dim(obstacles, game_seg_idxs[:, None, None, None], dim=1)[:, 0]
        game_frames = torch.take_along_dim(frames, game_seg_idxs[:, None], dim=1)[:, 0]
        game_3p = torch.take_along_dim(my_3p, game_seg_idxs[:, None, None, None], dim=1)[:, 0]
        game_history = torch.take_along_dim(history, game_seg_idxs[:, None, None, None], dim=1)[:, 0]

        playstyle_notes = torch.take_along_dim(notes, ref_seg_idxs[:, :, None, None], dim=1)
        playstyle_bombs = torch.take_along_dim(bombs, ref_seg_idxs[:, :, None, None], dim=1)
        playstyle_obstacles = torch.take_along_dim(obstacles, ref_seg_idxs[:, :, None, None], dim=1)
        playstyle_frames = torch.take_along_dim(frames, ref_seg_idxs, dim=1)
        playstyle_3p = torch.take_along_dim(my_3p, ref_seg_idxs[:, :, None, None], dim=1)
        playstyle_history = torch.take_along_dim(history, ref_seg_idxs[:, :, None, None], dim=1)

        losses, pred_losses, recon_losses, matching_losses = collect_rollout(
            pred_net,
            gsvae_net,
            pred_ema,
            gsvae_ema,
            optimizer,
            game_notes,
            game_bombs,
            game_obstacles,
            game_history,
            game_frames,
            game_3p,
            playstyle_notes,
            playstyle_bombs,
            playstyle_obstacles,
            playstyle_history,
            playstyle_frames,
            playstyle_3p,
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

        # Update progress bar with losses and GPU metrics
        postfix_dict = {
            "recon_loss": recon_losses.mean().item(),
            "matching_loss": matching_losses.mean().item(),
            "batches": batches_elapsed,
        }
        if batches_elapsed % 10 == 0:  # Update GPU metrics every 10 batches to avoid overhead
            gpu_metrics_pbar = get_gpu_metrics(device.index if device.index is not None else 0)
            postfix_dict["vram_mb"] = int(gpu_metrics_pbar.get("vram_allocated_mb", 0))
            if "gpu_utilization_pct" in gpu_metrics_pbar:
                postfix_dict["gpu_util%"] = int(gpu_metrics_pbar["gpu_utilization_pct"])
        pbar.set_postfix(postfix_dict)

    pbar.close()
    train_dataset.msg_q.put(None)
    while not train_dataset.q.empty():
        train_dataset.q.get()
    train_dataset.proc.join()
    writer.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--minibatch_size", type=int, default=512)
    parser.add_argument("--n_total_batches", type=int, default=1_000_000)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--peak_lr", type=float, default=5e-5)
    parser.add_argument("--lr_decay_yes", action="store_true")
    parser.add_argument("--cached_yes", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--sentence_length", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=8)
    parser.add_argument("--segment_length", type=int, default=72)
    parser.add_argument("--chunk_length", type=int, default=64)
    parser.add_argument("--n_cands", type=int, default=1)
    parser.add_argument("--history_len", type=int, default=2)
    parser.add_argument("--matching_loss_type", type=str, default="jsd", choices=["mse", "jsd"])
    parser.add_argument("--matching_weight", type=float, default=1e-4)
    parser.add_argument("--arch", type=str, default="ccm", choices=["ccm", "ad"])
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--jab", action="store_true")
    parser.add_argument("--n_inner", type=int, default=1)
    parser.add_argument("--playstyle_no", action="store_true")
    parser.add_argument(
        "--n_segs",
        type=int,
        default=6,
        help="Number of segments to use in each datapoint. One is used for prediction, the rest is used for playstyle reference.",
    )
    parser.add_argument("--task_scale", type=float, default=1.0)
    args, remaining_args = parser.parse_known_args()

    main(args, remaining_args)
