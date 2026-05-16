import copy
import glob
import io
import os
import tarfile
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import requests
import torch
import xarray as xr
from tqdm import tqdm

import pauli
from beaty_common.bsmg_xror_utils import get_cbo_np, load_cbo_and_3p, open_beatmap_from_bsmg_or_boxrr
from beaty_common.data_utils import SegmentSampler, device as eval_device
from beaty_common.eval_utils import evaluate_3p_on_map
from beaty_common.gen_utils import generate_3p_from_style_embeddings
from beaty_common.train_utils import nanpad_collate_fn
from xror.xror import XROR

device = torch.device("cuda")


def main(args, remaining_args):
    outdir = "out"
    os.makedirs(outdir, exist_ok=True)
    nc_out_path = f"{outdir}/gen3p.nc"

    checkpoint_d = torch.load(args.gen_path, weights_only=False)
    pred_net = pauli.load(checkpoint_d["pred_net_d"])
    pred_net_state_dict = checkpoint_d["pred_net_state_dict"]
    for key in list(pred_net_state_dict.keys()):
        clean_key = key.replace("_orig_mod.", "")
        pred_net_state_dict[clean_key] = pred_net_state_dict.pop(key)
    pred_net.load_state_dict(pred_net_state_dict)
    pred_net.requires_grad_(False)
    pred_net = pred_net.to("cuda").eval()
    print("Loaded pred_net")

    gsvae_net = pauli.load(checkpoint_d["gsvae_net_d"])
    gsvae_net_state_dict = checkpoint_d["gsvae_net_state_dict"]
    for key in list(gsvae_net_state_dict.keys()):
        clean_key = key.replace("_orig_mod.", "")
        gsvae_net_state_dict[clean_key] = gsvae_net_state_dict.pop(key)
    gsvae_net.load_state_dict(gsvae_net_state_dict)
    gsvae_net.requires_grad_(False)
    gsvae_net = gsvae_net.to("cuda").eval()

    print("Loading manifest...")
    boxrr23_manifest = pd.read_csv(args.boxrr23_manifest_path)
    boxrr23_manifest["User ID"] = boxrr23_manifest["User ID"].astype(str)

    globbed = sorted(glob.glob(args.csv_path, recursive=True))
    df = pd.concat([pd.read_csv(filename) for filename in tqdm(globbed)])
    if "User ID" in df.columns:
        df["User ID"] = df["User ID"].astype(str)
    join_cols = [col for col in df.columns if col in boxrr23_manifest.columns]
    if len(join_cols) == 0:
        raise ValueError(f"No shared columns between {args.csv_path} and {args.boxrr23_manifest_path}")
    boxrr23_manifest_join = boxrr23_manifest.drop_duplicates(subset=join_cols)
    df = df.merge(boxrr23_manifest_join, on=join_cols, how="left")

    classy_checkpoint_dict = torch.load(args.classy_path, weights_only=False)
    loaded_mod_dat = classy_checkpoint_dict["mod_dat"]
    module_dat = []
    for i in range(len(loaded_mod_dat)):
        net_state_dict, ema_state_dict, net_d = loaded_mod_dat[i]
        net = pauli.load(net_d)
        for key in list(net_state_dict.keys()):
            clean_key = key.replace("_orig_mod.", "")
            net_state_dict[clean_key] = net_state_dict.pop(key)
        for key in list(ema_state_dict.keys()):
            clean_key = key.replace("_orig_mod.", "")
            ema_state_dict[clean_key] = ema_state_dict.pop(key)
        net.load_state_dict(net_state_dict)
        net = net.to(eval_device).eval().requires_grad_(False)
        ema = copy.deepcopy(net).to(device=eval_device).eval().requires_grad_(False)
        ema.load_state_dict(ema_state_dict)
        module_dat.append([net, ema, net_d])
    classy_enc_ema = module_dat[0][1]
    classy_head_ema = module_dat[1][1]

    chunk_length = checkpoint_d["args"].chunk_length
    history_len = checkpoint_d["args"].history_len

    means = boxrr23_manifest.groupby("User ID")["Normalized Score"].mean()
    if len(means) == 0:
        raise ValueError(f"No users found in {args.boxrr23_manifest_path}")
    p99 = means.quantile(0.99)
    target_player_ids = means[means >= p99].index.astype(str).tolist()
    if len(target_player_ids) == 0:
        raise ValueError("No target_player_ids with mean Normalized Score >= 99th percentile")

    unique_player_ids = boxrr23_manifest["User ID"].unique()
    id_to_cat = {player_id: i for i, player_id in enumerate(unique_player_ids)}
    segment_sampler = SegmentSampler()
    user_tar_cache = {}
    segment_sampler_batch_size = 2048
    segment_length = 72
    stride = 4
    purview_notes = 20
    floor_time = -0.1

    os.makedirs("data/BeatSaver", exist_ok=True)
    n_outputs = min(10, len(df))
    csv_i = 0
    outputs_written = 0
    pbar = tqdm(total=n_outputs)

    while outputs_written < n_outputs and csv_i < len(df):
        np.random.seed(csv_i)
        torch.manual_seed(csv_i)
        torch.cuda.manual_seed_all(csv_i)

        target_song_hash = str(df.iloc[csv_i]["Song Hash"])
        target_difficulty = str(df.iloc[csv_i]["Difficulty Level"])
        target_zip_path = f"data/BeatSaver/{target_song_hash.upper()}.zip"

        if args.target_player_source == "csv":
            if "User ID" not in df.columns:
                raise ValueError("target_player_source=csv requires a 'User ID' column in csv_path")
            raw_target_player_id = df.iloc[csv_i]["User ID"]
            if pd.isna(raw_target_player_id):
                print(f"Skipping csv row {csv_i}: missing User ID for target_player_source=csv")
                csv_i += 1
                continue
            target_player_id = str(raw_target_player_id)
        else:
            target_player_id = str(np.random.choice(target_player_ids))
        if target_player_id not in id_to_cat:
            print(f"Skipping csv row {csv_i}: {target_player_id=} not found in {args.boxrr23_manifest_path}")
            csv_i += 1
            continue
        user_tar = user_tar_cache.get(target_player_id)
        if user_tar is None:
            response = requests.get(
                f"https://huggingface.co/datasets/cschell/boxrr-23/resolve/main/users/{target_player_id[0]}/{target_player_id}.tar",
                timeout=300,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch HF user tar for User ID {target_player_id}: " f"{response.status_code} {response.text}")
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:") as tf:
                member_names = [member.name for member in tf.getmembers() if member.isfile()]
            if len(member_names) == 0:
                raise ValueError(f"No replay files found in HF user tar for User ID {target_player_id}")
            user_tar = {
                "bytes": response.content,
                "member_names": member_names,
            }
            user_tar_cache[target_player_id] = user_tar
        candidate_member_names = np.random.permutation(user_tar["member_names"])
        valid_reference_replays = []
        with tarfile.open(fileobj=io.BytesIO(user_tar["bytes"]), mode="r:") as tf:
            for member_name in candidate_member_names:
                if len(valid_reference_replays) >= 1:
                    break
                member_file = tf.extractfile(str(member_name))
                if member_file is None:
                    raise ValueError(f"Failed to read replay {member_name} from HF user tar for User ID {target_player_id}")
                xror_raw = member_file.read()
                xror_unpacked = XROR.unpack(xror_raw)
                fetched_user_id = str(xror_unpacked.data["info"]["user"]["id"])
                if fetched_user_id != target_player_id:
                    raise ValueError(f"HF user tar mismatch for User ID {target_player_id}: " f"replay {member_name} belongs to {fetched_user_id}")
                try:
                    difficulty = xror_unpacked.data["info"]["software"]["activity"]["difficulty"]
                    if isinstance(difficulty, int):
                        difficulty = {
                            1: "Easy",
                            3: "Normal",
                            5: "Hard",
                            7: "Expert",
                            9: "ExpertPlus",
                        }[difficulty]
                    song_hash = str(xror_unpacked.data["info"]["software"]["activity"]["songHash"]).upper()
                    zip_path = f"data/BeatSaver/{song_hash}.zip"
                    beatmap_ref, map_info_ref, _ = open_beatmap_from_bsmg_or_boxrr(zip_path, None, difficulty)
                    left_handed = xror_unpacked.data["info"]["software"]["activity"].get("leftHanded", False)
                    cbo_and_3p = load_cbo_and_3p(
                        xror_unpacked,
                        beatmap_ref,
                        map_info_ref,
                        left_handed=left_handed,
                        rescale_yes=True,
                    )
                except (FileNotFoundError, KeyError, ValueError):
                    continue
                valid_reference_replays.append(cbo_and_3p)
        if len(valid_reference_replays) == 0:
            print(f"Skipping csv row {csv_i}: found no valid reference replays for {target_player_id=}")
            csv_i += 1
            continue
        sampled_ref_idxs = np.random.choice(len(valid_reference_replays), size=5, replace=True)
        cbo_and_3p_for_set = [valid_reference_replays[idx] for idx in sampled_ref_idxs]

        d = nanpad_collate_fn([cbo_and_3p_for_set])
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.to(device=device)
        res = []
        for j in range(d["notes_np"].shape[0]):
            seen_game_segments, seen_movement_segments = segment_sampler.sample_for_training(
                d["notes_np"][[j]],
                d["bombs_np"][[j]],
                d["obstacles_np"][[j]],
                d["timestamps"][[j]],
                d["gt_3p_np"][[j]],
                d["lengths"][[j]],
                segment_length,
                1,
                segment_sampler_batch_size,
                stride,
                2.0,
                purview_notes,
                floor_time,
            )
            notes_ = seen_game_segments.notes[:, 2]
            bombs_ = seen_game_segments.bombs[:, 2]
            obstacles_ = seen_game_segments.obstacles[:, 2]
            my_3p_ = seen_movement_segments.three_p[:, 2:]
            history_ = seen_movement_segments.three_p[:, :2]
            res.append((notes_, bombs_, obstacles_, my_3p_, history_))
        sample_dict = {
            "notes": torch.cat([x[0] for x in res], dim=0).detach().cpu(),
            "bombs": torch.cat([x[1] for x in res], dim=0).detach().cpu(),
            "obstacles": torch.cat([x[2] for x in res], dim=0).detach().cpu(),
            "my_3p": torch.cat([x[3] for x in res], dim=0).detach().cpu(),
            "history": torch.cat([x[4] for x in res], dim=0).detach().cpu(),
        }

        notes = sample_dict["notes"].to("cuda")
        bombs = sample_dict["bombs"].to("cuda")
        obstacles = sample_dict["obstacles"].to("cuda")
        my_3p = sample_dict["my_3p"].to("cuda")
        history = sample_dict["history"].to("cuda")
        style_embs, style_masks = pred_net.encode_style(notes[None], bombs[None], obstacles[None], history[None], my_3p[None])

        segment_3p, cand_3p = generate_3p_from_style_embeddings(
            style_embs,
            style_masks,
            target_song_hash,
            target_difficulty,
            pred_net,
            gsvae_net,
            device,
            chunk_length,
            history_len,
        )
        generated_3p = segment_3p.detach().cpu()
        generated_cands = cand_3p.detach().cpu()

        x = {
            "Song Hash": target_song_hash,
            "Difficulty Level": target_difficulty,
            "User ID": target_player_id,
            "seed": csv_i,
        }
        player_cat = torch.as_tensor([id_to_cat[target_player_id]], device=eval_device)
        my_3p = generated_3p.to(device=eval_device, dtype=torch.float32)
        song_hash = str(x["Song Hash"])
        difficulty = str(x["Difficulty Level"])
        characteristic = "Standard"

        try:
            beatmap, map_info, song_duration = open_beatmap_from_bsmg_or_boxrr(
                target_zip_path,
                None,
                target_difficulty,
            )
        except FileNotFoundError:
            print(f"Skipping csv row {csv_i}: target map unavailable for {target_song_hash=} {target_difficulty=}")
            csv_i += 1
            continue

        notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
        timestamps = np.arange(0, song_duration, 1 / 60)
        min_len = min(my_3p.shape[2], timestamps.shape[0])
        timestamps = timestamps[:min_len]
        my_3p = my_3p[:, :, :min_len]
        d = {
            "notes_np": notes_np,
            "bombs_np": bombs_np,
            "obstacles_np": obstacles_np,
            "timestamps": timestamps,
            "3p": my_3p[0, 0].flatten(-2, -1).detach().cpu().numpy(),
        }
        d = nanpad_collate_fn([[d]])
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.to(device=eval_device)
        game_segments, movement_segments = segment_sampler.sample_for_training(
            d["notes_np"],
            d["bombs_np"],
            d["obstacles_np"],
            d["timestamps"],
            d["3p"],
            d["lengths"],
            72,
            36,
            512,
            4,
            2.0,
            20,
            -0.1,
        )
        notes = game_segments.notes[:, 2]
        bombs = game_segments.bombs[:, 2]
        obstacles = game_segments.obstacles[:, 2]
        history = movement_segments.three_p[:, :2]
        future_3p = movement_segments.three_p[:, 2:]
        with torch.no_grad():
            z = classy_enc_ema.forward(
                notes.unflatten(0, (-1, 6)),
                bombs.unflatten(0, (-1, 6)),
                obstacles.unflatten(0, (-1, 6)),
                history.unflatten(0, (-1, 6)),
                future_3p.unflatten(0, (-1, 6)),
            )
            logits = classy_head_ema.forward(z)
            cross_entropy_loss = torch.nn.functional.cross_entropy(logits, player_cat.repeat_interleave(6), reduction="none")

        accuracy = (logits.argmax(dim=-1) == player_cat).float().mean()
        top_10_accuracy = (logits.topk(10, dim=-1).indices == player_cat[:, None]).any(dim=-1).float().mean()
        top_100_accuracy = (logits.topk(100, dim=-1).indices == player_cat[:, None]).any(dim=-1).float().mean()
        top_1000_accuracy = (logits.topk(1000, dim=-1).indices == player_cat[:, None]).any(dim=-1).float().mean()

        ts, n_opportunities, n_goods, n_hits, n_misses = evaluate_3p_on_map(my_3p, difficulty, characteristic, beatmap, map_info, song_duration)
        ts = ts.item()
        n_opportunities = n_opportunities.item()
        n_goods = n_goods.item()
        n_hits = n_hits.item()
        n_misses = n_misses.item()
        print(f"{target_song_hash=} {target_difficulty=} {target_player_id=} {csv_i=}")
        print(f"{cross_entropy_loss=} {accuracy=} {top_10_accuracy=} " f"{top_100_accuracy=} {top_1000_accuracy=}")
        print(f"{ts=} {n_opportunities=} {n_hits=} {n_misses=} {n_goods=}")

        three_p_arr = generated_3p.numpy()
        cands_arr = generated_cands.numpy()
        three_p_dims = tuple(f"three_p_dim_{j}" for j in range(three_p_arr.ndim))
        cands_dims = tuple(f"cands_dim_{j}" for j in range(cands_arr.ndim))
        ds = xr.Dataset(
            {
                "3p": (three_p_dims, three_p_arr),
                "cands": (cands_dims, cands_arr),
                "TS": ts,
                "n_opportunities": n_opportunities,
                "n_hits": n_hits,
                "n_misses": n_misses,
                "n_goods": n_goods,
                "PR_GC": n_goods / n_opportunities,
                "CE": cross_entropy_loss.mean().item(),
                "Top-1 Accuracy": accuracy.item(),
                "Top-10 Accuracy": top_10_accuracy.item(),
                "Top-100 Accuracy": top_100_accuracy.item(),
                "Top-1000 Accuracy": top_1000_accuracy.item(),
            },
            attrs={
                "Song Hash": x["Song Hash"],
                "Difficulty Level": x["Difficulty Level"],
                "User ID": x["User ID"],
                "seed": int(x["seed"]),
            },
        )
        mode = "w" if outputs_written == 0 else "a"
        ds.to_netcdf(nc_out_path, mode=mode, group=str(outputs_written), engine="h5netcdf")
        outputs_written += 1
        csv_i += 1
        pbar.update(1)

    pbar.close()
    if outputs_written == 0:
        print("No rows were generated.")
        return
    print(f"Saved to {nc_out_path}")
    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--csv_path", type=str, default="data/for_video.csv")
    parser.add_argument("--gen_path", type=str, default="models/ccm.pkl")
    parser.add_argument("--classy_path", type=str, default="models/classy.pkl")
    parser.add_argument("--boxrr23_manifest_path", type=str, default="data/boxrr23_post_qc.csv")
    parser.add_argument("--target_player_source", type=str, default="strong_random", choices=["strong_random", "csv"])
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
