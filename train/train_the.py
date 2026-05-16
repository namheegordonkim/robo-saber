import copy
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.optim import AdamW, RAdam
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import pynvml  # from nvidia-ml-py package

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

import pauli
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train.torch_nets2 import TransformerGSVAE, GameplayEncoder, SentinelPredictor
from beaty_common.train_utils import BoxrrCacheDataset, cache_collate_fn

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
    metrics["vram_max_allocated_mb"] = (
        torch.cuda.max_memory_allocated(device_id) / 1024**2
    )

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
    writer.add_text("args", str(args))
    writer.add_text("remaining_args", str(remaining_args))

    batch_size = args.batch_size
    n_total_batches = args.n_total_batches

    n_total_batches = max(int(n_total_batches * args.task_scale), 1)

    save_every = max(
        1, args.save_every if args.save_every is not None else n_total_batches // 10
    )
    eval_every = max(
        1, args.eval_every if args.eval_every is not None else save_every // 10
    )
    n_warmup_batches = max(1, save_every // 10)

    umds_df = pd.read_csv(f"{proj_dir}/data/boxrr_umds_v17.csv")
    unique_player_ids = umds_df["User ID"].unique()
    id_to_cat = {k: v for (v, k) in enumerate(unique_player_ids)}

    checkpoint_dict = {}
    if args.checkpoint_path is not None:
        logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
        checkpoint_dict = torch.load(args.checkpoint_path, weights_only=False)
        n_segs = checkpoint_dict["args"].n_segs
        logger.info(f"Using {n_segs=}")
        args.n_segs = n_segs

    unseen_dataset = BoxrrCacheDataset(
        f"{proj_dir}/{args.data_dir}", batch_size=batch_size, init_i=0, mode="classy"
    )
    unseen_dataloader = DataLoader(
        unseen_dataset, batch_size=None, collate_fn=cache_collate_fn, pin_memory=True
    )

    train_dataset = BoxrrCacheDataset(
        f"{proj_dir}/{args.data_dir}", batch_size=batch_size, init_i=1, mode="classy"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=None, collate_fn=cache_collate_fn, pin_memory=True
    )

    train_dataloader_iter = iter(
        train_dataloader
    )  # makes sense to use this if dataset size is fixed, but we have on-the-fly validation

    # Set aside evaluation set
    npr_st = np.random.get_state()
    np.random.seed(0)

    d = next(iter(unseen_dataloader))
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.pin_memory().to(device=device)
    unseen_notes, unseen_bombs, unseen_obstacles, unseen_history, unseen_3p = (
        get_gameplay_tensors(d, args.n_segs)
    )
    unseen_cats = torch.tensor([id_to_cat[i] for i in d["id"]], device=device)

    d = next(train_dataloader_iter)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.pin_memory().to(device=device)

    setup_notes = d["notes"]
    setup_bombs = d["bombs"]
    setup_obstacles = d["obstacles"]
    setup_frames = d["t"]
    setup_3p = d["my_3p"]
    setup_history = d["history"]

    seen_notes, seen_bombs, seen_obstacles, seen_history, seen_3p = (
        get_gameplay_tensors(d, args.n_segs)
    )
    seen_cats = torch.tensor([id_to_cat[i] for i in d["id"]], device=device)

    np.random.set_state(npr_st)

    pbar = tqdm(total=n_total_batches)

    # Initialize model
    if args.checkpoint_path is None:
        logger.info(f"Initializing a model")
        parameters = []
        if args.what == "gsvae":
            gsvae = TransformerGSVAE(
                setup_3p.shape[-1],
                args.hidden_size,
                args.embed_size,
                args.vocab_size,
                args.sentence_length,
                args.chunk_length,
                args.stride,
                args.num_heads,
                args.num_layers,
            )
            gsvae_ema = (
                copy.deepcopy(gsvae).to(device=device).eval().requires_grad_(False)
            )
            gsvae_d = pauli.dump(
                gsvae, os.path.abspath(os.path.join(os.getcwd(), "src"))
            )
            gsvae = gsvae.to(device)
            parameters.extend(gsvae.parameters())
            gsvae.setup(setup_3p)
            module_dat = [[gsvae, gsvae_ema, gsvae_d]]

        elif args.what == "classy":
            module_list = []
            classy_enc = GameplayEncoder(
                setup_notes.shape[-1],
                setup_bombs.shape[-1],
                setup_obstacles.shape[-1],
                setup_history.shape[-1],
                args.hidden_size,
                args.hidden_size,
                args.num_heads,
                args.num_layers,
            )
            module_list.append(classy_enc)

            if args.arch == "cls":
                classy_head = SentinelPredictor(
                    args.hidden_size,
                    unique_player_ids.shape[0],
                    args.hidden_size,
                    1,
                    1,
                )
                module_list.append(classy_head)

            else:
                classy_head = SentinelPredictor(
                    args.hidden_size,
                    args.embed_size,
                    args.hidden_size,
                    1,
                    1,
                )
                classy_emb = torch.nn.Embedding(
                    unique_player_ids.shape[0], args.embed_size
                )
                module_list.append(classy_head)
                module_list.append(classy_emb)

            # classy_head = torch.nn.Sequential(
            #     torch.nn.LeakyReLU(),
            #     torch.nn.Linear(args.hidden_size, args.hidden_size),
            #     torch.nn.LeakyReLU(),
            #     torch.nn.Linear(args.embed_size, unique_player_ids.shape[0]),
            # )
            # classy_head = classy_head.to(device)

            module_dat = []
            for m in module_list:
                m = m.to(device)
                parameters.extend(m.parameters())
                ema = copy.deepcopy(m).to(device=device).eval().requires_grad_(False)
                de = pauli.dump(m, os.path.abspath(os.path.join(os.getcwd(), "src")))
                module_dat.append([m, ema, de])

            classy_enc_ema = module_dat[0][1]
            classy_head_ema = module_dat[1][1]
            if args.arch == "reg":
                classy_emb_ema = module_dat[2][1]

            classy_enc.setup(setup_notes, setup_bombs, setup_obstacles, setup_history)

        else:
            raise NotImplementedError
        # optimizer = AdamW(parameters, lr=args.peak_lr)
        optimizer = RAdam(parameters, lr=args.peak_lr)

    else:
        logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
        # checkpoint_dict = torch.load(args.checkpoint_path, weights_only=False)
        loaded_mod_dat = checkpoint_dict["mod_dat"]

        parameters = []
        module_dat = []
        for i in range(len(loaded_mod_dat)):
            net_state_dict, ema_state_dict, net_d = loaded_mod_dat[i]
            net = pauli.load(net_d)
            for k in list(net_state_dict.keys()):
                kk = k.replace("_orig_mod.", "")
                net_state_dict[kk] = net_state_dict.pop(k)
            net.load_state_dict(net_state_dict)
            net = net.to(device)
            net.requires_grad_(True)
            for name, param in net.named_parameters():
                if "rms" in name:
                    param.requires_grad_(False)
            ema = copy.deepcopy(net).to(device=device).eval().requires_grad_(False)
            ema.load_state_dict(ema_state_dict)
            loaded_mod_dat[i] = [net, ema, net_d]
            parameters.extend(net.parameters())
            module_dat.append([net, ema, net_d])

        if args.what == "classy":
            classy_enc = module_dat[0][0]
            classy_enc_ema = module_dat[0][1]
            classy_head = module_dat[1][0]
            classy_head_ema = module_dat[1][1]
            if args.arch == "reg":
                classy_emb = module_dat[2][0]
                classy_emb_ema = module_dat[2][1]
        elif args.what == "gsvae":
            gsvae = module_dat[0][0]
            gsvae_ema = module_dat[0][1]

        optimizer = AdamW(parameters, lr=args.peak_lr)
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

        if args.continue_yes:
            batches_elapsed = checkpoint_dict["batches_elapsed"]
            samples_elapsed = checkpoint_dict["samples_elapsed"]
            pbar.update(batches_elapsed)

    if not args.debug_yes:
        for m in module_dat:
            print("Compiling model...")
            m[0] = torch.compile(m[0])

    # Main training loop
    batches_elapsed = 0
    samples_elapsed = 0
    while True:
        if batches_elapsed % save_every == 0 or batches_elapsed >= n_total_batches:
            pauli_root = os.path.abspath(
                os.path.join(os.getcwd(), "src")
            )  # NOTE: assume that all scripts are run from the parent directory of src.
            save_me = [
                [module.state_dict(), ema.state_dict(), module_d]
                for (module, ema, module_d) in module_dat
            ]
            checkpoint_dict = {
                "mod_dat": save_me,
                "optimizer_state_dict": optimizer.state_dict(),
                "batches_elapsed": batches_elapsed,
                "samples_elapsed": samples_elapsed,
                "args": args,
            }
            save_path = f"{outdir}/{args.what}_{samples_elapsed // 1000:06d}.pkl"
            torch.save(checkpoint_dict, save_path)
            logger.info(f"Saved to {save_path}")

        if batches_elapsed % eval_every == 0:
            for m in module_dat:
                m[0] = m[0].eval()

            seen_stuff = [
                seen_notes,
                seen_bombs,
                seen_obstacles,
                seen_history,
                seen_3p,
                seen_cats,
            ]
            unseen_stuff = [
                unseen_notes,
                unseen_bombs,
                unseen_obstacles,
                unseen_history,
                unseen_3p,
                unseen_cats,
            ]
            for name, stuff in [("seen", seen_stuff), ("unseen", unseen_stuff)]:
                notes, bombs, obstacles, history, _3p, cats = stuff

                if args.what == "classy":
                    if args.arch == "cls":
                        with torch.no_grad():
                            z = classy_enc_ema.forward(
                                notes, bombs, obstacles, history, _3p
                            )
                            logits = classy_head_ema.forward(z)
                            cross_entropy_loss = torch.nn.functional.cross_entropy(
                                logits, cats
                            )

                        accuracy = (logits.argmax(dim=-1) == cats).float().mean()
                        top_10_accuracy = (
                            (logits.topk(10, dim=-1).indices == cats[:, None])
                            .any(dim=-1)
                            .float()
                            .mean()
                        )
                        top_100_accuracy = (
                            (logits.topk(100, dim=-1).indices == cats[:, None])
                            .any(dim=-1)
                            .float()
                            .mean()
                        )
                        top_1000_accuracy = (
                            (logits.topk(1000, dim=-1).indices == cats[:, None])
                            .any(dim=-1)
                            .float()
                            .mean()
                        )

                        # Get GPU metrics
                        gpu_metrics = get_gpu_metrics(
                            device.index if device.index is not None else 0
                        )

                        gpu_info_str = (
                            f"VRAM: {gpu_metrics.get('vram_allocated_mb', 0):.0f}MB"
                        )
                        if "gpu_utilization_pct" in gpu_metrics:
                            gpu_info_str += (
                                f" GPU: {gpu_metrics['gpu_utilization_pct']:.1f}%"
                            )
                        if "gpu_memory_utilization_pct" in gpu_metrics:
                            gpu_info_str += f" MemUtil: {gpu_metrics['gpu_memory_utilization_pct']:.1f}%"

                        logger.info(
                            f"Batch {batches_elapsed} ksamples {samples_elapsed // 1000} {name} CrossEntropyLoss: {cross_entropy_loss.item():.2e} Accuracy: {accuracy.item():.2e} Top10: {top_10_accuracy.item():.2e} Top100: {top_100_accuracy.item():.2e} Top1000: {top_1000_accuracy.item():.2e} {gpu_info_str}"
                        )
                        for unit, value in [
                            ["batches", batches_elapsed],
                            ["ksamples", samples_elapsed // 1000],
                        ]:
                            writer.add_scalar(
                                f"{name}/cross_entropy_loss/{unit}",
                                cross_entropy_loss.item(),
                                value,
                            )
                            writer.add_scalar(
                                f"{name}/accuracy/{unit}", accuracy.item(), value
                            )
                            writer.add_scalar(
                                f"{name}/top_10_accuracy/{unit}",
                                top_10_accuracy.item(),
                                value,
                            )
                            writer.add_scalar(
                                f"{name}/top_100_accuracy/{unit}",
                                top_100_accuracy.item(),
                                value,
                            )
                            writer.add_scalar(
                                f"{name}/top_1000_accuracy/{unit}",
                                top_1000_accuracy.item(),
                                value,
                            )

                            # Log GPU metrics
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

                    elif args.arch == "reg":
                        with torch.no_grad():
                            z = classy_enc_ema.forward(
                                notes, bombs, obstacles, history, _3p
                            )
                            pred = classy_head_ema.forward(z)
                            player_embs = classy_emb_ema.forward(
                                torch.tensor(
                                    [id_to_cat[i] for i in d["id"]], device=device
                                )
                            )
                            mse_loss = torch.nn.functional.mse_loss(pred, player_embs)

                        # Get GPU metrics
                        gpu_metrics = get_gpu_metrics(
                            device.index if device.index is not None else 0
                        )

                        gpu_info_str = (
                            f"VRAM: {gpu_metrics.get('vram_allocated_mb', 0):.0f}MB"
                        )
                        if "gpu_utilization_pct" in gpu_metrics:
                            gpu_info_str += (
                                f" GPU: {gpu_metrics['gpu_utilization_pct']:.1f}%"
                            )
                        if "gpu_memory_utilization_pct" in gpu_metrics:
                            gpu_info_str += f" MemUtil: {gpu_metrics['gpu_memory_utilization_pct']:.1f}%"

                        logger.info(
                            f"Batch {batches_elapsed} ksamples {samples_elapsed // 1000} {name} MSELoss: {mse_loss.item():.2e} {gpu_info_str}"
                        )
                        for unit, value in [
                            ["batches", batches_elapsed],
                            ["ksamples", samples_elapsed // 1000],
                        ]:
                            writer.add_scalar(
                                f"{name}/mse_loss/{unit}", mse_loss.item(), value
                            )

                            # Log GPU metrics
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
                else:
                    with torch.no_grad():
                        z_ref, k, z_soft_ref, z_hard_ref, y_ref = gsvae.forward(
                            _3p, n=1
                        )
                        recon_loss = torch.nn.functional.mse_loss(
                            y_ref, _3p[:, None], reduction="none"
                        ).mean()

                        # Get GPU metrics
                        gpu_metrics = get_gpu_metrics(
                            device.index if device.index is not None else 0
                        )

                        gpu_info_str = (
                            f"VRAM: {gpu_metrics.get('vram_allocated_mb', 0):.0f}MB"
                        )
                        if "gpu_utilization_pct" in gpu_metrics:
                            gpu_info_str += (
                                f" GPU: {gpu_metrics['gpu_utilization_pct']:.1f}%"
                            )
                        if "gpu_memory_utilization_pct" in gpu_metrics:
                            gpu_info_str += f" MemUtil: {gpu_metrics['gpu_memory_utilization_pct']:.1f}%"

                        logger.info(
                            f"Batch {batches_elapsed} ksamples {samples_elapsed // 1000} {name} ReconLoss: {recon_loss.item():.2e} {gpu_info_str}"
                        )
                        for unit, value in [
                            ["batches", batches_elapsed],
                            ["ksamples", samples_elapsed // 1000],
                        ]:
                            writer.add_scalar(
                                f"{name}/recon_loss/{unit}", recon_loss.item(), value
                            )

                            # Log GPU metrics
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

        for m in module_dat:
            m[0] = m[0].train()

        # Learning rate decay
        min_lr = 0
        if args.lr_decay_yes:
            if batches_elapsed % save_every < n_warmup_batches:
                lr = min_lr + (args.peak_lr - min_lr) * np.clip(
                    (batches_elapsed % save_every) / n_warmup_batches, 0, 1
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
                            * (batches_elapsed % save_every - n_warmup_batches)
                            / (save_every - n_warmup_batches)
                        )
                    )
                    / 2
                )
        else:
            lr = args.peak_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        writer.add_scalar("train/lr/batches", lr, batches_elapsed)
        writer.add_scalar("train/lr/ksamples", lr, samples_elapsed)

        # Load training data here
        if args.jab:
            d = next(iter(train_dataloader))
        else:
            d = next(train_dataloader_iter)
        if d is not None:
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.pin_memory().to(device=device)

        notes, bombs, obstacles, history, _3p = get_gameplay_tensors(d, args.n_segs)

        if args.what == "classy":
            if args.arch == "cls":
                z = classy_enc.forward(notes, bombs, obstacles, history, _3p)
                logits = classy_head.forward(z)
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor([id_to_cat[i] for i in d["id"]], device=device)
                )
            else:
                z = classy_enc.forward(notes, bombs, obstacles, history, _3p)
                pred = classy_head.forward(z)
                player_embs = classy_emb.forward(
                    torch.tensor([id_to_cat[i] for i in d["id"]], device=device)
                )
                loss = torch.nn.functional.mse_loss(pred, player_embs)
        else:
            z_ref, k, z_soft_ref, z_hard_ref, y_ref = gsvae.forward(_3p, n=1)
            loss = torch.nn.functional.mse_loss(
                y_ref, _3p[:, None], reduction="none"
            ).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        # Update EMA.
        ema_halflife_kimg = 500
        ema_halflife_nimg = ema_halflife_kimg * 1000
        ema_beta = 0.5 ** (_3p.shape[0] / max(ema_halflife_nimg, 1e-8))
        for m in module_dat:
            module, ema, _ = m
            for p_ema, p_net in zip(ema.parameters(), module.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        batches_elapsed += 1
        samples_elapsed += batch_size
        pbar.update(1)

        # Update progress bar with losses and GPU metrics
        postfix_dict = {"loss": loss.item(), "batches": batches_elapsed}
        if (
            batches_elapsed % 10 == 0
        ):  # Update GPU metrics every 10 batches to avoid overhead
            gpu_metrics_pbar = get_gpu_metrics(
                device.index if device.index is not None else 0
            )
            postfix_dict["vram_mb"] = int(gpu_metrics_pbar.get("vram_allocated_mb", 0))
            if "gpu_utilization_pct" in gpu_metrics_pbar:
                postfix_dict["gpu_util%"] = int(gpu_metrics_pbar["gpu_utilization_pct"])
        pbar.set_postfix(postfix_dict)

    pbar.close()
    writer.close()
    train_dataset.msg_q.put(None)
    while not train_dataset.q.empty():
        train_dataset.q.get()
    train_dataset.proc.join()

    unseen_dataset.msg_q.put(None)
    while not unseen_dataset.q.empty():
        unseen_dataset.q.get()
    unseen_dataset.proc.join()
    unseen_dataset.proc.join()

    logger.info(f"Done")


def get_gameplay_tensors(d, n_segs):
    notes = d["notes"]
    bombs = d["bombs"]
    obstacles = d["obstacles"]
    frames = d["t"]
    my_3p = d["my_3p"]
    history = d["history"]

    game_seg_idxs = torch.argsort(
        torch.rand((notes.shape[0], n_segs), device=device), -1
    )  # Shuffle up the order
    # game_seg_idxs = torch.argsort(torch.rand((notes.shape[0], notes.shape[1]), device=device), -1)
    game_notes = torch.take_along_dim(notes, game_seg_idxs[..., None, None], dim=1)
    game_bombs = torch.take_along_dim(bombs, game_seg_idxs[..., None, None], dim=1)
    game_obstacles = torch.take_along_dim(
        obstacles, game_seg_idxs[..., None, None], dim=1
    )
    game_frames = torch.take_along_dim(frames, game_seg_idxs, dim=1)
    game_3p = torch.take_along_dim(my_3p, game_seg_idxs[..., None, None], dim=1)
    game_history = torch.take_along_dim(history, game_seg_idxs[..., None, None], dim=1)
    return game_notes, game_bombs, game_obstacles, game_history, game_3p


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--what",
        type=str,
        choices=["classy", "gsvae", "3pgen_enc", "3pgen_full", "diffusion"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_total_batches", type=int, default=288_000)
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
    parser.add_argument(
        "--matching_loss_type", type=str, default="jsd", choices=["mse", "jsd"]
    )
    parser.add_argument("--matching_weight", type=float, default=1e-4)
    parser.add_argument("--arch", type=str, default="cls", choices=["cls", "reg"])
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
