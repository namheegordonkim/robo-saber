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
from beaty_common.torch_nets2 import CondTransformerGSVAE, GameplayEncoder, SentinelPredictor, TransformerGSVAE
from xror.xror import XROR

device = torch.device("cuda")
user_tar_cache = {}


def load_models(args):
    checkpoint_d = torch.load(args.gen_path, weights_only=False)

    pred_net = pauli.load(checkpoint_d["pred_net_d"])
    pred_net = CondTransformerGSVAE(
        note_size=pred_net.note_size,
        bomb_size=pred_net.bomb_size,
        obstacle_size=pred_net.obstacle_size,
        history_size=pred_net.threep_size,
        hidden_size=pred_net.hidden_size,
        embed_size=pred_net.embed_size,
        sentence_length=pred_net.sentence_length,
        vocab_size=pred_net.vocab_size,
        num_heads=pred_net.num_heads,
        num_layers=pred_net.num_layers,
    )
    pred_net_state_dict = checkpoint_d["pred_net_state_dict"]
    for key in list(pred_net_state_dict.keys()):
        clean_key = key.replace("_orig_mod.", "")
        pred_net_state_dict[clean_key] = pred_net_state_dict.pop(key)
    pred_net.load_state_dict(pred_net_state_dict)
    pred_net = pred_net.to(device).eval().requires_grad_(False)

    gsvae_net = pauli.load(checkpoint_d["gsvae_net_d"])
    gsvae_net = TransformerGSVAE(
        input_size=gsvae_net.input_size,
        hidden_size=gsvae_net.hidden_size,
        embed_size=gsvae_net.embed_size,
        vocab_size=gsvae_net.vocab_size,
        sentence_length=gsvae_net.sentence_length,
        chunk_length=gsvae_net.chunk_length,
        stride=gsvae_net.stride,
        num_heads=gsvae_net.num_heads,
        num_layers=gsvae_net.num_layers,
    )
    gsvae_net_state_dict = checkpoint_d["gsvae_net_state_dict"]
    for key in list(gsvae_net_state_dict.keys()):
        clean_key = key.replace("_orig_mod.", "")
        gsvae_net_state_dict[clean_key] = gsvae_net_state_dict.pop(key)
    gsvae_net.load_state_dict(gsvae_net_state_dict)
    gsvae_net = gsvae_net.to(device).eval().requires_grad_(False)

    classy_checkpoint_dict = torch.load(args.classy_path, weights_only=False)
    emas = []
    for net_state_dict, ema_state_dict, net_d in classy_checkpoint_dict["mod_dat"]:
        net = pauli.load(net_d)
        if type(net).__name__ == "GameplayEncoder":
            net = GameplayEncoder(
                note_size=net.note_size,
                bomb_size=net.bomb_size,
                obstacle_size=net.obstacle_size,
                history_size=net.threep_size,
                hidden_size=net.hidden_size,
                embed_size=net.embed_size,
                num_heads=net.num_heads,
                num_layers=net.num_layers,
            )
        elif type(net).__name__ == "SentinelPredictor":
            net = SentinelPredictor(
                input_size=net.input_size,
                output_size=net.output_size,
                hidden_size=net.hidden_size,
                num_heads=net.num_heads,
                num_layers=net.num_layers,
            )
        else:
            raise TypeError(f"Unexpected classy module type: {type(net).__name__}")
        for key in list(ema_state_dict.keys()):
            clean_key = key.replace("_orig_mod.", "")
            ema_state_dict[clean_key] = ema_state_dict.pop(key)
        net.load_state_dict(ema_state_dict)
        emas.append(net.to(eval_device).eval().requires_grad_(False))
    classy_enc_ema, classy_head_ema = emas

    return {
        "pred_net": pred_net,
        "gsvae_net": gsvae_net,
        "classy_enc_ema": classy_enc_ema,
        "classy_head_ema": classy_head_ema,
        "chunk_length": checkpoint_d["args"].chunk_length,
        "history_len": checkpoint_d["args"].history_len,
    }


def load_data(args):
    boxrr23_manifest = pd.read_csv(args.boxrr23_manifest_path)
    boxrr23_manifest["User ID"] = boxrr23_manifest["User ID"].astype(str)

    df = pd.concat([pd.read_csv(filename) for filename in tqdm(sorted(glob.glob(args.csv_path, recursive=True)))])
    if "User ID" in df.columns:
        df["User ID"] = df["User ID"].astype(str)
    join_cols = [col for col in df.columns if col in boxrr23_manifest.columns]
    if len(join_cols) == 0:
        raise ValueError(f"No shared columns between {args.csv_path} and {args.boxrr23_manifest_path}")
    df = df.merge(boxrr23_manifest.drop_duplicates(subset=join_cols), on=join_cols, how="left")

    means = boxrr23_manifest.groupby("User ID")["Normalized Score"].mean()
    target_player_ids = means[means >= means.quantile(0.99)].index.tolist()
    id_to_class_idx = {pid: i for i, pid in enumerate(boxrr23_manifest["User ID"].unique())}

    return df, target_player_ids, id_to_class_idx


def generate(models, target_player_id, target_song_hash, target_difficulty):
    user_tar = user_tar_cache.get(target_player_id)
    if user_tar is None:
        response = requests.get(
            f"https://huggingface.co/datasets/cschell/boxrr-23/resolve/main/users/{target_player_id[0]}/{target_player_id}.tar",
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch HF user tar for User ID {target_player_id}: {response.status_code} {response.text}")
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:") as tf:
            member_names = [member.name for member in tf.getmembers() if member.isfile()]
        if len(member_names) == 0:
            raise ValueError(f"No replay files found in HF user tar for User ID {target_player_id}")
        user_tar = {"bytes": response.content, "member_names": member_names}
        user_tar_cache[target_player_id] = user_tar

    candidate_member_names = np.random.permutation(user_tar["member_names"])
    valid_reference_replays = []
    with tarfile.open(fileobj=io.BytesIO(user_tar["bytes"]), mode="r:") as tf:
        for member_name in candidate_member_names:
            member_file = tf.extractfile(str(member_name))
            if member_file is None:
                raise ValueError(f"Failed to read replay {member_name} from HF user tar for User ID {target_player_id}")
            xror_unpacked = XROR.unpack(member_file.read())
            fetched_user_id = str(xror_unpacked.data["info"]["user"]["id"])
            if fetched_user_id != target_player_id:
                raise ValueError(f"HF user tar mismatch for User ID {target_player_id}: replay {member_name} belongs to {fetched_user_id}")
            try:
                activity = xror_unpacked.data["info"]["software"]["activity"]
                difficulty = activity["difficulty"]
                if isinstance(difficulty, int):
                    difficulty = {1: "Easy", 3: "Normal", 5: "Hard", 7: "Expert", 9: "ExpertPlus"}[difficulty]
                song_hash = str(activity["songHash"]).upper()
                beatmap_ref, map_info_ref, _ = open_beatmap_from_bsmg_or_boxrr(f"data/BeatSaver/{song_hash}.zip", None, difficulty)
                left_handed = activity.get("leftHanded", False)
                cbo_and_3p = load_cbo_and_3p(xror_unpacked, beatmap_ref, map_info_ref, left_handed=left_handed, rescale_yes=True)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            valid_reference_replays.append(cbo_and_3p)
            break
    if len(valid_reference_replays) == 0:
        return None

    sampled_ref_idxs = np.random.choice(len(valid_reference_replays), size=5, replace=True)
    cbo_and_3p_for_set = [valid_reference_replays[idx] for idx in sampled_ref_idxs]

    d = nanpad_collate_fn([cbo_and_3p_for_set])
    d = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    res = []
    for j in range(d["notes_np"].shape[0]):
        seen_game_segments, seen_movement_segments = SegmentSampler().sample_for_training(
            d["notes_np"][[j]], d["bombs_np"][[j]], d["obstacles_np"][[j]],
            d["timestamps"][[j]], d["gt_3p_np"][[j]], d["lengths"][[j]],
            72, 1, 2048, 4, 2.0, 20, -0.1,
        )
        res.append((
            seen_game_segments.notes[:, 2],
            seen_game_segments.bombs[:, 2],
            seen_game_segments.obstacles[:, 2],
            seen_movement_segments.three_p[:, 2:],
            seen_movement_segments.three_p[:, :2],
        ))
    notes, bombs, obstacles, my_3p, history = (torch.cat(parts, dim=0) for parts in zip(*res))

    style_embs, style_masks = models["pred_net"].encode_style(
        notes[None], bombs[None], obstacles[None], history[None], my_3p[None]
    )
    segment_3p, cand_3p = generate_3p_from_style_embeddings(
        style_embs, style_masks, target_song_hash, target_difficulty,
        models["pred_net"], models["gsvae_net"], device,
        models["chunk_length"], models["history_len"],
    )
    return segment_3p.detach().cpu(), cand_3p.detach().cpu()


def evaluate(models, generated_3p, target_zip_path, target_difficulty, player_cat):
    try:
        beatmap, map_info, song_duration = open_beatmap_from_bsmg_or_boxrr(target_zip_path, None, target_difficulty)
    except FileNotFoundError:
        return None

    my_3p = generated_3p.to(device=eval_device, dtype=torch.float32)

    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
    timestamps = np.arange(0, song_duration, 1 / 60)
    min_len = min(my_3p.shape[2], timestamps.shape[0])
    timestamps = timestamps[:min_len]
    my_3p = my_3p[:, :, :min_len]
    d = nanpad_collate_fn([[{
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "timestamps": timestamps,
        "3p": my_3p[0, 0].flatten(-2, -1).detach().cpu().numpy(),
    }]])
    d = {k: v.to(device=eval_device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    game_segments, movement_segments = SegmentSampler().sample_for_training(
        d["notes_np"], d["bombs_np"], d["obstacles_np"],
        d["timestamps"], d["3p"], d["lengths"],
        72, 36, 512, 4, 2.0, 20, -0.1,
    )
    with torch.no_grad():
        z = models["classy_enc_ema"].forward(
            game_segments.notes[:, 2].unflatten(0, (-1, 6)),
            game_segments.bombs[:, 2].unflatten(0, (-1, 6)),
            game_segments.obstacles[:, 2].unflatten(0, (-1, 6)),
            movement_segments.three_p[:, :2].unflatten(0, (-1, 6)),
            movement_segments.three_p[:, 2:].unflatten(0, (-1, 6)),
        )
        logits = models["classy_head_ema"].forward(z)
        cross_entropy_loss = torch.nn.functional.cross_entropy(logits, player_cat.repeat_interleave(6), reduction="none")

    ts, n_opportunities, n_goods, n_hits, n_misses = (
        t.item() for t in evaluate_3p_on_map(my_3p, target_difficulty, "Standard", beatmap, map_info, song_duration)
    )

    return {
        "TS": ts,
        "n_opportunities": n_opportunities,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "n_goods": n_goods,
        "PR_GC": n_goods / n_opportunities,
        "CE": cross_entropy_loss.mean().item(),
        **{
            f"Top-{k} Accuracy": (logits.topk(k, dim=-1).indices == player_cat[:, None]).any(dim=-1).float().mean().item()
            for k in (1, 10, 100, 1000)
        },
    }


def write_record(nc_out_path, outputs_written, generated_3p, generated_cands, metrics,
                 target_song_hash, target_difficulty, target_player_id, csv_i):
    three_p_arr = generated_3p.numpy()
    cands_arr = generated_cands.numpy()
    three_p_dims = tuple(f"three_p_dim_{j}" for j in range(three_p_arr.ndim))
    cands_dims = tuple(f"cands_dim_{j}" for j in range(cands_arr.ndim))
    ds = xr.Dataset(
        {
            "3p": (three_p_dims, three_p_arr),
            "cands": (cands_dims, cands_arr),
            **metrics,
        },
        attrs={
            "Song Hash": target_song_hash,
            "Difficulty Level": target_difficulty,
            "User ID": target_player_id,
            "seed": int(csv_i),
        },
    )
    mode = "w" if outputs_written == 0 else "a"
    ds.to_netcdf(nc_out_path, mode=mode, group=str(outputs_written), engine="h5netcdf")


def main(args, remaining_args):
    outdir = "out"
    os.makedirs(outdir, exist_ok=True)
    nc_out_path = f"{outdir}/gen3p.nc"

    models = load_models(args)
    print("Loaded models")
    print("Loading manifest...")
    df, target_player_ids, id_to_class_idx = load_data(args)

    os.makedirs("data/BeatSaver", exist_ok=True)
    n_outputs = min(10, len(df))
    outputs_written = 0
    pbar = tqdm(total=n_outputs)

    for i in range(len(df)):
        if outputs_written >= n_outputs:
            break

        np.random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)

        row = df.iloc[i]
        target_song_hash = str(row["Song Hash"])
        target_difficulty = str(row["Difficulty Level"])
        target_zip_path = f"data/BeatSaver/{target_song_hash.upper()}.zip"

        if args.target_player_source == "csv":
            if "User ID" not in df.columns:
                raise ValueError("target_player_source=csv requires a 'User ID' column in csv_path")
            raw_target_player_id = row["User ID"]
            if pd.isna(raw_target_player_id):
                print(f"Skipping csv row {i}: missing User ID for target_player_source=csv")
                continue
            target_player_id = str(raw_target_player_id)
        else:
            target_player_id = str(np.random.choice(target_player_ids))

        if target_player_id not in id_to_class_idx:
            print(f"Skipping csv row {i}: {target_player_id=} not found in {args.boxrr23_manifest_path}")
            continue

        out = generate(models, target_player_id, target_song_hash, target_difficulty)
        if out is None:
            print(f"Skipping csv row {i}: found no valid reference replays for {target_player_id=}")
            continue
        generated_3p, generated_cands = out

        player_cat = torch.as_tensor([id_to_class_idx[target_player_id]], device=eval_device)
        metrics = evaluate(models, generated_3p, target_zip_path, target_difficulty, player_cat)
        if metrics is None:
            print(f"Skipping csv row {i}: target map unavailable for {target_song_hash=} {target_difficulty=}")
            continue

        print(f"{i=} {target_song_hash=} {target_difficulty=} {target_player_id=}")
        print(metrics)
        write_record(nc_out_path, outputs_written, generated_3p, generated_cands, metrics, target_song_hash, target_difficulty, target_player_id, i)
        outputs_written += 1
        pbar.update(1)

    pbar.close()
    if outputs_written == 0:
        print("No rows were generated.")
        return
    print(f"Saved to {nc_out_path}")
    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--gen_path", type=str, default="models/ccm.pkl")
    parser.add_argument("--classy_path", type=str, default="models/classy.pkl")
    parser.add_argument("--boxrr23_manifest_path", type=str, default="data/boxrr23_post_qc.csv")
    parser.add_argument("--target_player_source", type=str, default="strong_random", choices=["strong_random", "csv"])
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
