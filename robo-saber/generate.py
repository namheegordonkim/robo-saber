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

from beaty_common.bsmg_xror_utils import get_cbo_np, load_cbo_and_3p, open_beatmap_from_bsmg_or_boxrr
from beaty_common.data_utils import device as eval_device, sample_for_training
from beaty_common.eval_utils import evaluate_3p_on_map
from beaty_common.gen_utils import generate_3p_from_style_embeddings
from beaty_common.torch_nets import CondTransformerGSVAE, GameplayEncoder, ReplayTensors, SentinelPredictor, TransformerGSVAE
from beaty_common.train_utils import nanpad_collate_fn
from vendor.xror.xror import XROR

MODEL_DEVICE = torch.device("cuda")
PLAYER_HEIGHT = 1.5044
BEATSAVER_DIR = "data/BeatSaver"
STANDARD_CHARACTERISTIC = "Standard"
DEFAULT_CLEAN_SNAPSHOT_PATH = os.path.normpath("models/pretrained.pkl")
MAX_OUTPUT_ROWS = 10
STRONG_PLAYER_QUANTILE = 0.99
REFERENCE_REPLAYS_PER_REQUEST = 5
REFERENCE_SEGMENT_LENGTH = 72
REFERENCE_SEGMENT_COUNT = 1
REFERENCE_SEGMENT_MINIBATCH = 2048
REFERENCE_SEGMENT_STRIDE = 4
REFERENCE_LOOKAHEAD_SECONDS = 2.0
REFERENCE_PURVIEW_NOTES = 20
REFERENCE_FLOOR_TIME = -0.1
REFERENCE_HISTORY_FRAMES = 2
CLASSIFIER_SEGMENT_LENGTH = 72
CLASSIFIER_SEGMENT_COUNT = 36
CLASSIFIER_SEGMENT_MINIBATCH = 512
CLASSIFIER_SEGMENT_STRIDE = 4
CLASSIFIER_LOOKAHEAD_SECONDS = 2.0
CLASSIFIER_PURVIEW_NOTES = 20
CLASSIFIER_FLOOR_TIME = -0.1
DIFFICULTY_NAME_BY_CODE = {
    1: "Easy",
    3: "Normal",
    5: "Hard",
    7: "Expert",
    9: "ExpertPlus",
}

player_tar_cache = {}


def load_models(bundle_path):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

    style_predictor = CondTransformerGSVAE(**bundle["pred_kw"])
    style_predictor.load_state_dict(bundle["pred_sd"], strict=True)
    style_predictor = style_predictor.to(MODEL_DEVICE).eval().requires_grad_(False)

    trajectory_decoder = TransformerGSVAE(**bundle["gsvae_kw"])
    trajectory_decoder.load_state_dict(bundle["gsvae_sd"], strict=True)
    trajectory_decoder = trajectory_decoder.to(MODEL_DEVICE).eval().requires_grad_(False)

    player_encoder = GameplayEncoder(**bundle["classy_enc_kw"])
    player_encoder.load_state_dict(bundle["classy_enc_sd"], strict=True)
    player_encoder = player_encoder.to(eval_device).eval().requires_grad_(False)

    player_classifier = SentinelPredictor(**bundle["classy_head_kw"])
    player_classifier.load_state_dict(bundle["classy_head_sd"], strict=True)
    player_classifier = player_classifier.to(eval_device).eval().requires_grad_(False)

    return {
        "style_predictor": style_predictor,
        "trajectory_decoder": trajectory_decoder,
        "player_encoder": player_encoder,
        "player_classifier": player_classifier,
        "chunk_length": int(bundle["chunk_length"]),
        "history_length": int(bundle["history_len"]),
    }


def load_request_rows(args):
    boxrr23_manifest = pd.read_csv(args.boxrr23_manifest_path)
    boxrr23_manifest["User ID"] = boxrr23_manifest["User ID"].astype(str)

    request_rows = pd.concat([pd.read_csv(path) for path in tqdm(sorted(glob.glob(args.csv_path, recursive=True)))])
    if "User ID" in request_rows.columns:
        request_rows["User ID"] = request_rows["User ID"].astype(str)

    join_columns = [column for column in request_rows.columns if column in boxrr23_manifest.columns]
    if len(join_columns) == 0:
        raise ValueError(f"No shared columns between {args.csv_path} and {args.boxrr23_manifest_path}")
    request_rows = request_rows.merge(
        boxrr23_manifest.drop_duplicates(subset=join_columns),
        on=join_columns,
        how="left",
    )

    mean_scores = boxrr23_manifest.groupby("User ID")["Normalized Score"].mean()
    strong_player_ids = mean_scores[mean_scores >= mean_scores.quantile(STRONG_PLAYER_QUANTILE)].index.tolist()
    player_id_to_class_index = {player_id: index for index, player_id in enumerate(boxrr23_manifest["User ID"].unique())}
    return request_rows, strong_player_ids, player_id_to_class_index


def generate(models, target_player_id, target_song_hash, target_difficulty):
    player_tar = player_tar_cache.get(target_player_id)
    if player_tar is None:
        response = requests.get(
            f"https://huggingface.co/datasets/cschell/boxrr-23/resolve/main/users/{target_player_id[0]}/{target_player_id}.tar",
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch HF user tar for User ID {target_player_id}: {response.status_code} {response.text}"
            )
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:") as tar_file:
            member_names = [member.name for member in tar_file.getmembers() if member.isfile()]
        if len(member_names) == 0:
            raise ValueError(f"No replay files found in HF user tar for User ID {target_player_id}")
        player_tar = {"bytes": response.content, "member_names": member_names}
        player_tar_cache[target_player_id] = player_tar

    reference_replay = None
    with tarfile.open(fileobj=io.BytesIO(player_tar["bytes"]), mode="r:") as tar_file:
        for member_name in np.random.permutation(player_tar["member_names"]):
            member_file = tar_file.extractfile(str(member_name))
            if member_file is None:
                raise ValueError(f"Failed to read replay {member_name} from HF user tar for User ID {target_player_id}")

            unpacked_xror = XROR.unpack(member_file.read())
            fetched_user_id = str(unpacked_xror.data["info"]["user"]["id"])
            if fetched_user_id != target_player_id:
                raise ValueError(
                    f"HF user tar mismatch for User ID {target_player_id}: replay {member_name} belongs to {fetched_user_id}"
                )

            try:
                activity = unpacked_xror.data["info"]["software"]["activity"]
                replay_difficulty = activity["difficulty"]
                if isinstance(replay_difficulty, int):
                    replay_difficulty = DIFFICULTY_NAME_BY_CODE[replay_difficulty]
                replay_song_hash = str(activity["songHash"]).upper()
                beatmap, map_info = open_beatmap_from_bsmg_or_boxrr(
                    f"{BEATSAVER_DIR}/{replay_song_hash}.zip",
                    None,
                    replay_difficulty,
                )[:2]
                left_handed = activity.get("leftHanded", False)
                reference_replay = load_cbo_and_3p(
                    unpacked_xror,
                    beatmap,
                    map_info,
                    left_handed=left_handed,
                    rescale_yes=True,
                )
            except (FileNotFoundError, KeyError, ValueError):
                continue
            break

    if reference_replay is None:
        return None

    collated = nanpad_collate_fn([[reference_replay for reference_index in range(REFERENCE_REPLAYS_PER_REQUEST)]])
    collated = {
        name: value.to(device=MODEL_DEVICE) if isinstance(value, torch.Tensor) else value
        for name, value in collated.items()
    }

    reference_segments = []
    for reference_index in range(collated["notes_np"].shape[0]):
        sampled_replay = sample_for_training(
            collated["notes_np"][[reference_index]],
            collated["bombs_np"][[reference_index]],
            collated["obstacles_np"][[reference_index]],
            collated["timestamps"][[reference_index]],
            collated["gt_3p_np"][[reference_index]],
            collated["lengths"][[reference_index]],
            REFERENCE_SEGMENT_LENGTH,
            REFERENCE_SEGMENT_COUNT,
            REFERENCE_SEGMENT_MINIBATCH,
            REFERENCE_SEGMENT_STRIDE,
            REFERENCE_LOOKAHEAD_SECONDS,
            REFERENCE_PURVIEW_NOTES,
            REFERENCE_FLOOR_TIME,
        )
        reference_segments.append(
            (
                sampled_replay.notes[:, REFERENCE_HISTORY_FRAMES],
                sampled_replay.bombs[:, REFERENCE_HISTORY_FRAMES],
                sampled_replay.obstacles[:, REFERENCE_HISTORY_FRAMES],
                sampled_replay.trajectory[:, REFERENCE_HISTORY_FRAMES:],
                sampled_replay.trajectory[:, :REFERENCE_HISTORY_FRAMES],
            )
        )

    reference_notes, reference_bombs, reference_obstacles, reference_trajectory, reference_history = (
        torch.cat(parts, dim=0) for parts in zip(*reference_segments)
    )
    playstyle_tokens, playstyle_mask = models["style_predictor"].encode_style(
        ReplayTensors(
            reference_notes[None],
            reference_bombs[None],
            reference_obstacles[None],
            history=reference_history[None],
            trajectory=reference_trajectory[None],
        )
    )
    generated_3p, generated_candidates = generate_3p_from_style_embeddings(
        playstyle_tokens,
        playstyle_mask,
        target_song_hash,
        target_difficulty,
        models["style_predictor"],
        models["trajectory_decoder"],
        MODEL_DEVICE,
        models["chunk_length"],
        models["history_length"],
    )
    return generated_3p.detach().cpu(), generated_candidates.detach().cpu()


def evaluate(models, generated_3p, target_zip_path, target_difficulty, player_category):
    try:
        beatmap, map_info, song_duration = open_beatmap_from_bsmg_or_boxrr(target_zip_path, None, target_difficulty)
    except FileNotFoundError:
        return None

    generated_3p = generated_3p.to(device=eval_device, dtype=torch.float32)
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
    timestamps = np.arange(0, song_duration, 1 / 60)
    min_length = min(generated_3p.shape[2], timestamps.shape[0])
    timestamps = timestamps[:min_length]
    generated_3p = generated_3p[:, :, :min_length]

    collated = nanpad_collate_fn(
        [[{
            "notes_np": notes_np,
            "bombs_np": bombs_np,
            "obstacles_np": obstacles_np,
            "timestamps": timestamps,
            "3p": generated_3p[0, 0].flatten(-2, -1).detach().cpu().numpy(),
        }]]
    )
    collated = {
        name: value.to(device=eval_device) if isinstance(value, torch.Tensor) else value
        for name, value in collated.items()
    }

    sampled_replay = sample_for_training(
        collated["notes_np"],
        collated["bombs_np"],
        collated["obstacles_np"],
        collated["timestamps"],
        collated["3p"],
        collated["lengths"],
        CLASSIFIER_SEGMENT_LENGTH,
        CLASSIFIER_SEGMENT_COUNT,
        CLASSIFIER_SEGMENT_MINIBATCH,
        CLASSIFIER_SEGMENT_STRIDE,
        CLASSIFIER_LOOKAHEAD_SECONDS,
        CLASSIFIER_PURVIEW_NOTES,
        CLASSIFIER_FLOOR_TIME,
    )
    with torch.no_grad():
        player_embedding = models["player_encoder"].forward(
            ReplayTensors(
                sampled_replay.notes[:, REFERENCE_HISTORY_FRAMES].unflatten(0, (-1, 6)),
                sampled_replay.bombs[:, REFERENCE_HISTORY_FRAMES].unflatten(0, (-1, 6)),
                sampled_replay.obstacles[:, REFERENCE_HISTORY_FRAMES].unflatten(0, (-1, 6)),
                history=sampled_replay.trajectory[:, :REFERENCE_HISTORY_FRAMES].unflatten(0, (-1, 6)),
                trajectory=sampled_replay.trajectory[:, REFERENCE_HISTORY_FRAMES:].unflatten(0, (-1, 6)),
            )
        )
        classifier_logits = models["player_classifier"].forward(player_embedding)
        cross_entropy = torch.nn.functional.cross_entropy(
            classifier_logits,
            player_category.repeat_interleave(6),
            reduction="none",
        )

    normalized_score, n_opportunities, n_goods, n_hits, n_misses = (
        value.item()
        for value in evaluate_3p_on_map(
            generated_3p,
            target_difficulty,
            STANDARD_CHARACTERISTIC,
            beatmap,
            map_info,
            song_duration,
        )
    )
    return {
        "TS": normalized_score,
        "n_opportunities": n_opportunities,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "n_goods": n_goods,
        "PR_GC": n_goods / n_opportunities,
        "CE": cross_entropy.mean().item(),
        **{
            f"Top-{k} Accuracy": (
                classifier_logits.topk(k, dim=-1).indices == player_category[:, None]
            ).any(dim=-1).float().mean().item()
            for k in (1, 10, 100, 1000)
        },
    }


def write_record(
    nc_out_path,
    outputs_written,
    generated_3p,
    generated_candidates,
    metrics,
    target_song_hash,
    target_difficulty,
    target_player_id,
    request_index,
):
    three_p_array = generated_3p.numpy()
    candidate_array = generated_candidates.numpy()
    three_p_dims = tuple(f"three_p_dim_{dim_index}" for dim_index in range(three_p_array.ndim))
    candidate_dims = tuple(f"cands_dim_{dim_index}" for dim_index in range(candidate_array.ndim))
    dataset = xr.Dataset(
        {
            "3p": (three_p_dims, three_p_array),
            "cands": (candidate_dims, candidate_array),
            **metrics,
        },
        attrs={
            "Song Hash": target_song_hash,
            "Difficulty Level": target_difficulty,
            "User ID": target_player_id,
            "seed": int(request_index),
        },
    )
    dataset.to_netcdf(
        nc_out_path,
        mode="w" if outputs_written == 0 else "a",
        group=str(outputs_written),
        engine="h5netcdf",
    )


def main(args):
    os.makedirs("out", exist_ok=True)
    output_path = args.nc_out_path
    output_parent = os.path.dirname(os.path.abspath(output_path))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    models = load_models(args.clean_models_bundle)
    print(f"Loaded models from inference snapshot {args.clean_models_bundle!r}")
    print("Loading manifest...")
    request_rows, strong_player_ids, player_id_to_class_index = load_request_rows(args)

    os.makedirs(BEATSAVER_DIR, exist_ok=True)
    n_outputs = min(MAX_OUTPUT_ROWS, len(request_rows))
    outputs_written = 0
    progress = tqdm(total=n_outputs)

    for request_index in range(len(request_rows)):
        if outputs_written >= n_outputs:
            break

        np.random.seed(request_index)
        torch.manual_seed(request_index)
        torch.cuda.manual_seed_all(request_index)

        request_row = request_rows.iloc[request_index]
        target_song_hash = str(request_row["Song Hash"])
        target_difficulty = str(request_row["Difficulty Level"])
        target_zip_path = f"{BEATSAVER_DIR}/{target_song_hash.upper()}.zip"

        if args.target_player_source == "csv":
            if "User ID" not in request_rows.columns:
                raise ValueError("target_player_source=csv requires a 'User ID' column in csv_path")
            raw_target_player_id = request_row["User ID"]
            if pd.isna(raw_target_player_id):
                print(f"Skipping csv row {request_index}: missing User ID for target_player_source=csv")
                continue
            target_player_id = str(raw_target_player_id)
        else:
            target_player_id = str(np.random.choice(strong_player_ids))

        if target_player_id not in player_id_to_class_index:
            print(f"Skipping csv row {request_index}: target_player_id={target_player_id!r} not found in {args.boxrr23_manifest_path}")
            continue

        generated = generate(models, target_player_id, target_song_hash, target_difficulty)
        if generated is None:
            print(f"Skipping csv row {request_index}: found no valid reference replays for target_player_id={target_player_id!r}")
            continue
        generated_3p, generated_candidates = generated

        player_category = torch.as_tensor([player_id_to_class_index[target_player_id]], device=eval_device)
        metrics = evaluate(models, generated_3p, target_zip_path, target_difficulty, player_category)
        if metrics is None:
            print(
                f"Skipping csv row {request_index}: target map unavailable for target_song_hash={target_song_hash!r} "
                f"target_difficulty={target_difficulty!r}"
            )
            continue

        print(
            f"request_index={request_index} target_song_hash={target_song_hash!r} "
            f"target_difficulty={target_difficulty!r} target_player_id={target_player_id!r}"
        )
        print(metrics)
        write_record(
            output_path,
            outputs_written,
            generated_3p,
            generated_candidates,
            metrics,
            target_song_hash,
            target_difficulty,
            target_player_id,
            request_index,
        )
        outputs_written += 1
        progress.update(1)

    progress.close()
    if outputs_written == 0:
        print("No rows were generated.")
        return
    print(f"Saved to {output_path}")
    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--boxrr23_manifest_path", type=str, default="data/boxrr23_post_qc.csv")
    parser.add_argument("--target_player_source", type=str, default="strong_random", choices=["strong_random", "csv"])
    parser.add_argument(
        "--clean_models_bundle",
        type=str,
        default=DEFAULT_CLEAN_SNAPSHOT_PATH,
        metavar="PATH",
        help=(
            "Single-file inference bundle (typically ``prepare.py`` -> models/pretrained.pkl). "
            f"Default: {DEFAULT_CLEAN_SNAPSHOT_PATH!r}."
        ),
    )
    parser.add_argument(
        "--nc_out_path",
        type=str,
        default="out/gen3p.nc",
        help="NetCDF output path (default: out/gen3p.nc).",
    )
    main(parser.parse_known_args()[0])
