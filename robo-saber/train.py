"""Train the released generator pair; stream BOXRR-23 and fetch BeatSaver maps."""

import argparse
from collections import OrderedDict
from contextlib import nullcontext
import copy
from itertools import chain
from pathlib import Path
import os
import random
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
import zlib

import bson
from datasets import IterableDataset
import numpy as np
import pandas as pd
import requests
import torch
from urllib3.exceptions import HTTPError as HTTPStreamError
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from beaty_common.torch_nets import CondTransformerGSVAE, GameTensors, ReplayTensors, TransformerGSVAE

revision = "af25be3ef76b176fbee0a094e82d97a611f9c950"
cache_limit = 512 * 1024**2
difficulty_names = {"1": "Easy", "3": "Normal", "5": "Hard", "7": "Expert", "9": "ExpertPlus"}
pred_kw = dict(note_size=8, bomb_size=5, obstacle_size=9, history_size=27,
               hidden_size=1024, embed_size=1024, sentence_length=128, vocab_size=8,
               num_heads=4, num_layers=4)
gsvae_kw = dict(input_size=27, hidden_size=1024, embed_size=128, vocab_size=8,
                sentence_length=128, chunk_length=64, stride=4, num_heads=4, num_layers=4)


def stream_examples(shards, seed, cache_root=None):
    # This generator is the worker entry point required by HF IterableDataset.
    from huggingface_hub import hf_hub_url
    from beaty_common.bsmg_xror_utils import load_cbo_and_3p, open_beatmap_from_bsmg_or_boxrr
    from beaty_common.data_utils import sample_for_training
    from beaty_common.train_utils import nanpad_collate_fn
    from vendor.xror.xror import XROR

    local_maps = Path("data/BeatSaver")
    local_maps.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robo-saber-maps-", dir=cache_root) as cache_dir, requests.Session() as session:
        cache = OrderedDict()
        cache_bytes = 0
        players = list(chain.from_iterable(shards))
        np.random.default_rng(seed).shuffle(players)
        for player, records in players:
            eligible = set(records)
            rng = np.random.default_rng([seed, zlib.crc32(player.encode())])
            sample_rng = torch.Generator().manual_seed(int(rng.integers(2**63)))
            url = hf_hub_url("cschell/boxrr-23", f"users/{player[0]}/{player}.tar",
                             repo_type="dataset", revision=revision)
            block = []
            last_member = -1
            for archive_attempt in range(3):
                try:
                    with session.get(url, stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        response.raw.decode_content = True
                        with tarfile.open(fileobj=response.raw, mode="r|", bufsize=1024**2) as archive:
                            for member_index, member in enumerate(chain(archive, [None])):
                                if member is not None:
                                    if member_index <= last_member:
                                        continue
                                    if not member.isfile() or not member.name.endswith(".xror"):
                                        last_member = member_index
                                        continue
                                    raw = archive.extractfile(member).read()
                                    last_member = member_index
                                    map_path = None
                                    try:
                                        info = bson.decode(raw)["info"]
                                        activity = info["software"]["activity"]
                                        song_hash = str(activity["songHash"]).upper()
                                        difficulty = str(activity["difficulty"])
                                        difficulty = difficulty_names.get(difficulty, difficulty)
                                        left_handed = str(activity.get("leftHanded", False)).lower() == "true"
                                        if "score" not in activity:
                                            continue
                                        key = (song_hash + difficulty, int(activity["score"]),
                                               activity.get("mode", "Standard"),
                                               activity.get("modifiers", ""), left_handed)
                                        if str(info["user"]["id"]) != player or key not in eligible:
                                            continue

                                        map_path = local_maps / f"{song_hash}.zip"
                                        if not map_path.exists():
                                            map_path = local_maps / f"{song_hash.lower()}.zip"
                                        if not map_path.exists():
                                            map_path = Path(cache_dir) / f"{song_hash}.zip"
                                            if song_hash in cache:
                                                cache.move_to_end(song_hash)
                                            else:
                                                partial = map_path.with_suffix(".part")
                                                for attempt in range(3):
                                                    try:
                                                        with session.get(f"https://api.beatsaver.com/maps/hash/{song_hash}",
                                                                         timeout=(15, 60)) as metadata:
                                                            metadata.raise_for_status()
                                                            versions = metadata.json()["versions"]
                                                        version = next((v for v in versions if v["hash"].upper() == song_hash), None)
                                                        if version is None:
                                                            raise FileNotFoundError(f"BeatSaver version {song_hash} is missing")
                                                        with session.get(version["downloadURL"], stream=True,
                                                                         timeout=(15, 60)) as download:
                                                            download.raise_for_status()
                                                            size = 0
                                                            with partial.open("wb") as output:
                                                                for chunk in download.iter_content(1024**2):
                                                                    size += len(chunk)
                                                                    while cache and cache_bytes + size > cache_limit:
                                                                        old_hash, old_size = cache.popitem(last=False)
                                                                        (Path(cache_dir) / f"{old_hash}.zip").unlink()
                                                                        cache_bytes -= old_size
                                                                    output.write(chunk)
                                                        with zipfile.ZipFile(partial) as downloaded_zip:
                                                            if downloaded_zip.testzip() is not None:
                                                                raise zipfile.BadZipFile("CRC failure")
                                                        partial.replace(map_path)
                                                        if size <= cache_limit:
                                                            cache[song_hash] = size
                                                            cache_bytes += size
                                                        break
                                                    except (requests.RequestException, zipfile.BadZipFile) as error:
                                                        partial.unlink(missing_ok=True)
                                                        status = getattr(getattr(error, "response", None), "status_code", None)
                                                        if attempt == 2 or (status is not None and status < 500 and status not in (408, 429)):
                                                            raise
                                                        time.sleep(2**attempt)

                                        beatmap, map_info, duration = open_beatmap_from_bsmg_or_boxrr(str(map_path), None, difficulty)
                                        replay = load_cbo_and_3p(XROR.unpack(raw), beatmap, map_info,
                                                                 left_handed=left_handed, rescale_yes=True)
                                        if len(replay["timestamps"]) < 72 or not np.isfinite(replay["gt_3p_np"]).all():
                                            raise ValueError("Replay has insufficient or nonfinite motion")
                                        collated = nanpad_collate_fn([[replay]])
                                        # The existing sampler uses the global CPU RNG. Isolate it even with --workers 0.
                                        with torch.random.fork_rng(devices=[]):
                                            torch.random.set_rng_state(sample_rng.get_state())
                                            segments = sample_for_training(
                                                collated["notes_np"], collated["bombs_np"], collated["obstacles_np"],
                                                collated["timestamps"], collated["gt_3p_np"], collated["lengths"],
                                                72, 32, 32, 4, 2.0, 20, -0.1)
                                            sample_rng.set_state(torch.random.get_rng_state())
                                        block.append(dict(notes=segments.notes[:, 2].clone(), bombs=segments.bombs[:, 2].clone(),
                                                          obstacles=segments.obstacles[:, 2].clone(), trajectory=segments.trajectory))
                                    except Exception as error:
                                        # Malformed data must not stop a multi-day stream; training errors are not caught here.
                                        print(f"Skipping {member.name}: {type(error).__name__}: {error}", flush=True)
                                    finally:
                                        if map_path is not None and map_path.parent == Path(cache_dir) and song_hash not in cache:
                                            map_path.unlink(missing_ok=True)  # Oversized maps are used once.

                                if block and (len(block) == 64 or member is None):
                                    windows = {name: torch.cat([part[name] for part in block]) for name in block[0]}
                                    block.clear()
                                    for example_index in range(len(windows["trajectory"])):
                                        indices = rng.integers(len(windows["trajectory"]), size=6)
                                        # Randomly choose the target; the remaining five segments are references.
                                        target_index = int(rng.integers(6))
                                        indices[[0, target_index]] = indices[[target_index, 0]]
                                        yield {**{name: value[indices].numpy() for name, value in windows.items()}, "player": player}
                    break
                except (requests.RequestException, HTTPStreamError, tarfile.TarError, OSError) as error:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    if archive_attempt == 2 or (status is not None and status < 500 and status not in (408, 429)):
                        print(f"Skipping archive {player}: {error}", flush=True)
                        break
                    time.sleep(2**archive_attempt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Resume checkpoint.pkl")
    parser.add_argument("--out-dir", type=Path, default=Path("out/train"))
    parser.add_argument("--total-batches", type=int, default=500_000)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(4, len(os.sched_getaffinity(0))))
    args = parser.parse_args()
    if args.total_batches < 0 or not 1 <= args.microbatch_size <= 128 or args.workers < 0:
        parser.error("total-batches and workers must be nonnegative; microbatch-size must be in 1..128")
    assets = ["models/pretrained.pkl", "data/boxrr23_post_qc.csv", "data/heldout_player_maps.csv",
              "data/placeholder_3p_sixd.txt"]
    if any(not Path(path).exists() for path in assets):
        subprocess.run([sys.executable, "robo-saber/prepare.py"], check=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}; effective batch 128, microbatch {args.microbatch_size}", flush=True)

    columns = ["User ID", "Song Hash and Difficulty", "Score", "Mode", "Modifiers", "Left Handed"]
    manifest = pd.read_csv(assets[1], usecols=columns, keep_default_na=False)
    heldout = pd.read_csv(assets[2], usecols=columns[:2]).drop_duplicates()
    train_mask = (~manifest[columns[0]].isin(heldout[columns[0]])
                  & ~manifest[columns[1]].isin(heldout[columns[1]]))
    split_rows = {"train": manifest[train_mask], "validation": manifest.merge(heldout, on=columns[:2])}
    shards = {}
    for split, rows in split_rows.items():
        shards[split] = [(str(player), tuple(group[columns[1:]].itertuples(index=False, name=None)))
                         for player, group in rows.groupby("User ID", sort=True)]
        if not shards[split]:
            raise RuntimeError(f"No eligible {split} replays")
        print(f"{split}: {len(rows):,} eligible replays from {len(shards[split]):,} players", flush=True)
    del manifest, heldout, train_mask, split_rows, rows

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint is not None and (checkpoint["pred_kw"] != pred_kw or checkpoint["gsvae_kw"] != gsvae_kw):
        raise ValueError("Checkpoint architecture differs from the released generator")
    fixed_batches = checkpoint["fixed_batches"] if checkpoint else {}
    # Parent owns the cache root so even interrupted/terminated workers leave no downloads behind.
    cache_root = tempfile.TemporaryDirectory(prefix="robo-saber-training-")
    for split in ("train", "validation"):
        if split not in fixed_batches:
            workers = min(args.workers, len(shards[split]), len(os.sched_getaffinity(0)))
            # One archive group per worker keeps its HTTP session and LRU cache alive across players.
            groups = [shards[split][index::max(1, workers)] for index in range(max(1, workers))]
            fixed_dataset = IterableDataset.from_generator(stream_examples, gen_kwargs={"shards": groups, "seed": 0, "cache_root": cache_root.name})
            # HF 5 otherwise interleaves many archives in threads before yielding its first example.
            fixed_dataset = fixed_dataset.shuffle(seed=0, buffer_size=1024, max_buffer_input_shards=1).with_format("torch")
            fixed_loader = DataLoader(fixed_dataset, batch_size=128, num_workers=workers, pin_memory=device.type == "cuda",
                                      generator=torch.Generator().manual_seed(0),
                                      **(dict(prefetch_factor=2, multiprocessing_context="spawn") if workers else {}))
            fixed_iterator = iter(fixed_loader)
            try:
                fixed_batches[split] = next(fixed_iterator)
            except StopIteration:
                raise RuntimeError(f"No usable {split} examples remain after loading") from None
            finally:
                del fixed_iterator, fixed_loader, fixed_dataset
                for leftover in Path(cache_root.name).iterdir():
                    shutil.rmtree(leftover)
            print(f"Prepared fixed {split} batch", flush=True)

    pred = CondTransformerGSVAE(**pred_kw).to(device)
    gsvae = TransformerGSVAE(**gsvae_kw).to(device)
    if checkpoint:
        pred.load_state_dict(checkpoint["pred_sd"])
        gsvae.load_state_dict(checkpoint["gsvae_sd"])
    else:
        batch = fixed_batches["train"]
        pred.setup(*(batch[name].flatten(0, 1).to(device) for name in ("notes", "bombs", "obstacles")),
                   batch["trajectory"][:, :, :2].flatten(0, 1).to(device))
        gsvae.setup(batch["trajectory"][:, :, 2:].flatten(0, 1).to(device))
    pred_ema = copy.deepcopy(pred).eval().requires_grad_(False)
    gsvae_ema = copy.deepcopy(gsvae).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(chain(pred.parameters(), gsvae.parameters()), lr=5e-5,
                                 betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    steps = 0
    samples = 0
    data_pass = 0
    if checkpoint:
        pred_ema.load_state_dict(checkpoint["pred_ema_sd"])
        gsvae_ema.load_state_dict(checkpoint["gsvae_ema_sd"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps, samples = checkpoint["steps"], checkpoint["samples"]
        data_pass = checkpoint["data_pass"] + 1
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.random.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    released = torch.load(assets[0], map_location="cpu", weights_only=False)
    classifier = {name: value for name, value in released.items() if name.startswith("classy_")}
    del released, checkpoint
    writer = SummaryWriter(str(args.out_dir / "tensorboard"), purge_step=steps if args.checkpoint else None)
    train_iterator = None
    loader_batches = 0
    saved_step = None
    evaluated_step = None
    interrupted = False
    try:
        while True:
            try:
                if (saved_step is None or steps % 25_000 == 0 or steps >= args.total_batches or interrupted) and saved_step != steps:
                    state = dict(pred_kw=pred_kw, gsvae_kw=gsvae_kw, pred_sd=pred.state_dict(), gsvae_sd=gsvae.state_dict(),
                                 pred_ema_sd=pred_ema.state_dict(), gsvae_ema_sd=gsvae_ema.state_dict(),
                                 optimizer=optimizer.state_dict(), steps=steps, samples=samples, data_pass=data_pass,
                                 python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.random.get_rng_state(),
                                 cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                                 fixed_batches=fixed_batches, args=vars(args), revision=revision)
                    temporary = args.out_dir / "checkpoint.tmp"
                    torch.save(state, temporary)
                    temporary.replace(args.out_dir / "checkpoint.pkl")
                    bundle = dict(classifier, pred_kw=pred_kw, gsvae_kw=gsvae_kw, chunk_length=64, history_len=2,
                                  pred_sd={name: value.detach().cpu() for name, value in pred.state_dict().items()},
                                  gsvae_sd={name: value.detach().cpu() for name, value in gsvae.state_dict().items()})
                    temporary = args.out_dir / "pretrained.tmp"
                    torch.save(bundle, temporary)
                    temporary.replace(args.out_dir / "pretrained.pkl")
                    del state, bundle
                    saved_step = steps
                    print(f"Saved step {steps:,} to {args.out_dir}", flush=True)
                if interrupted:
                    break
                phases = []
                if evaluated_step != steps and (evaluated_step is None or steps % 1250 == 0 or steps >= args.total_batches):
                    phases.extend(["train_eval", "validation"])
                if steps < args.total_batches:
                    phases.append("train")
                if not phases:
                    break
                for phase in phases:
                    training = phase == "train"
                    pred.train(training)
                    gsvae.train(training)
                    started = time.perf_counter()
                    if training:
                        while True:
                            if train_iterator is None:
                                workers = min(args.workers, len(shards["train"]), len(os.sched_getaffinity(0)))
                                groups = [shards["train"][index::max(1, workers)] for index in range(max(1, workers))]
                                dataset = IterableDataset.from_generator(stream_examples, gen_kwargs={"shards": groups, "seed": data_pass + 1, "cache_root": cache_root.name})
                                dataset = dataset.shuffle(seed=data_pass + 1, buffer_size=1024, max_buffer_input_shards=1).with_format("torch")
                                loader = DataLoader(dataset, batch_size=128, num_workers=workers, pin_memory=device.type == "cuda",
                                                    generator=torch.Generator().manual_seed(data_pass + 1),
                                                    **(dict(prefetch_factor=2, multiprocessing_context="spawn") if workers else {}))
                                train_iterator = iter(loader)
                                loader_batches = 0
                            try:
                                batch = next(train_iterator)
                                loader_batches += int(len(batch["trajectory"]) == 128)
                                break
                            except StopIteration:
                                if loader_batches == 0:
                                    raise RuntimeError("No complete training batch remains after loading") from None
                                train_iterator = None
                                data_pass += 1
                        # Iterable workers can end with partial batches; keep the optimizer batch exactly 128.
                        if len(batch["trajectory"]) != 128:
                            continue
                        loader_wait = time.perf_counter() - started
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        batch = fixed_batches["train" if phase == "train_eval" else "validation"]
                    totals = torch.zeros(4, device=device)
                    batch_size = len(batch["trajectory"])
                    # Evaluation samples Gumbel noise but must never advance training RNG.
                    with (nullcontext() if training else torch.random.fork_rng()), torch.set_grad_enabled(training):
                        if not training:
                            torch.manual_seed(0)
                        for offset in range(0, batch_size, args.microbatch_size):
                            micro = {name: value[offset:offset + args.microbatch_size].to(device, non_blocking=True).clone()
                                     for name, value in batch.items() if isinstance(value, torch.Tensor)}
                            game = GameTensors(micro["notes"][:, 0], micro["bombs"][:, 0], micro["obstacles"][:, 0], micro["trajectory"][:, 0, :2])
                            refs = ReplayTensors(micro["notes"][:, 1:], micro["bombs"][:, 1:], micro["obstacles"][:, 1:],
                                                 history=micro["trajectory"][:, 1:, :2], trajectory=micro["trajectory"][:, 1:, 2:])
                            target = micro["trajectory"][:, 0, 2:]
                            vae_output = gsvae(target)
                            pred_output = pred(game, refs)
                            recon = (vae_output[4][:, 0] - target).square().mean()
                            p = Categorical(logits=vae_output[0].reshape(-1, 128, 8))
                            q = Categorical(logits=pred_output[0].reshape(-1, 128, 8))
                            mixture = Categorical(probs=(p.probs + q.probs) / 2)
                            jsd = ((kl_divergence(p, mixture) + kl_divergence(q, mixture)) / 2).mean(-1).clamp(0, np.log(2)).mean()
                            loss = recon + 1e-4 * jsd
                            weight = len(target) / batch_size
                            if not torch.isfinite(loss):
                                raise FloatingPointError(f"Nonfinite loss at step {steps}")
                            if training:
                                (loss * weight).backward()
                                prediction_mse = loss.new_zeros(())
                            else:
                                prediction = gsvae_ema.decode(pred_ema(game, refs)[3])[:, 0]
                                prediction_mse = (prediction - target).square().mean()
                            totals += torch.stack([recon.detach(), jsd.detach(), loss.detach(), prediction_mse.detach()]) * weight
                    if training:
                        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1, error_if_nonfinite=True)
                        torch.nn.utils.clip_grad_norm_(gsvae.parameters(), 1, error_if_nonfinite=True)
                        # Finish an update before responding to Ctrl-C so saved optimizer/EMA/counters agree.
                        signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
                        try:
                            optimizer.step()
                            with torch.no_grad():
                                for live, ema in ((pred, pred_ema), (gsvae, gsvae_ema)):
                                    for current, average in zip(live.parameters(), ema.parameters()):
                                        average.lerp_(current, 1 - 0.5 ** (128 / 500_000))
                            steps += 1
                            samples += 128
                        finally:
                            signal.pthread_sigmask(signal.SIG_SETMASK, signal_mask)
                    else:
                        evaluated_step = steps
                    metrics = totals.tolist()  # Also synchronizes GPU work for the throughput measurement.
                    for name, value in zip(("reconstruction_mse", "jsd", "loss", "ema_prediction_mse"), metrics):
                        if name != "ema_prediction_mse" or not training:
                            writer.add_scalar(f"{phase}/{name}", value, steps)
                    if training:
                        writer.add_scalar("train/loader_wait_seconds", loader_wait, steps)
                        writer.add_scalar("train/examples_per_second", 128 / (time.perf_counter() - started), steps)
                    if not training or steps % 100 == 0 or steps == 1:
                        print(f"{phase} step={steps} recon={metrics[0]:.6g} jsd={metrics[1]:.6g} loss={metrics[2]:.6g}"
                              + (f" wait={loader_wait:.3f}s examples/s={128 / (time.perf_counter() - started):.2f}" if training
                                 else f" ema_prediction_mse={metrics[3]:.6g}"), flush=True)
                    pred.train()
                    gsvae.train()
            except KeyboardInterrupt:
                interrupted = True
                print("Interrupted; saving the last completed update.", flush=True)
    finally:
        writer.close()
        del train_iterator
        cache_root.cleanup()
