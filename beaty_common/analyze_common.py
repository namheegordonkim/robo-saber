import glob
from functools import reduce

import numpy as np
import torch
from datasets import Dataset

from beaty_common.bsmg_xror_utils import (
    open_beatmap_from_unpacked_xror,
    load_cbo_and_3p,
)
from beaty_common.data_utils import SegmentSampler
from beaty_common.train_utils import nanpad_collate_fn
from proj_utils.dirs import proj_dir
from xror.xror import XROR


def expand_objects_with_time(objects, num_frames, fps=60):
    """
    Expand objects tensor to include time dimension with adjusted timestamps.

    Args:
        objects: Tensor of shape [B1, B2, N, F] or [B, N, F] where objects[..., 0] contains timestamps in seconds
        num_frames: Number of time frames to expand to (e.g., 16)
        fps: Frames per second (default 15)

    Returns:
        Tensor of shape [B1, B2, T, N, F] or [B, T, N, F] where timestamps are adjusted for each frame
    """
    if objects.ndim == 3:
        # Handle 3D input: [B, N, F] -> [B, T, N, F]
        b, n, f = objects.shape
        expanded = objects.unsqueeze(1).expand(b, num_frames, n, f).clone()
        time_offsets = (
            torch.arange(num_frames, device=objects.device, dtype=objects.dtype) / fps
        )
        time_offsets = time_offsets.view(1, num_frames, 1)
        expanded[..., 0] = expanded[..., 0] - time_offsets
    elif objects.ndim == 4:
        # Handle 4D input: [B1, B2, N, F] -> [B1, B2, T, N, F]
        b1, b2, n, f = objects.shape
        expanded = objects.unsqueeze(2).expand(b1, b2, num_frames, n, f).clone()
        time_offsets = (
            torch.arange(num_frames, device=objects.device, dtype=objects.dtype) / fps
        )
        time_offsets = time_offsets.view(1, 1, num_frames, 1)
        expanded[..., 0] = expanded[..., 0] - time_offsets
    else:
        raise ValueError(
            f"Expected 3D or 4D tensor, got {objects.ndim}D tensor with shape {objects.shape}"
        )

    return expanded


def collect_samples_from_index_pairs(index_pair_set):
    """
    Collect game samples from index pairs.

    Args:
        index_pair_set: Array of (shard_idx, datapoint_idx) pairs

    Returns:
        Dictionary containing notes, bombs, obstacles, t, my_3p, history, and metadata lists
        (user_ids, song_hashes, difficulties, song_hash_and_difficulty - one entry per sample)
    """
    # Hardcoded configuration
    segment_sampler = SegmentSampler()
    segment_length = 72
    stride = 4
    purview_notes = 20
    purview_sec = 2.0
    floor_time = -0.1
    segment_sampler_batch_size = 2048
    arrow_files = np.array(
        sorted(
            glob.glob(
                f"{proj_dir}/datasets/boxrr-23/cschell___boxrr-23/default/0.0.0/af25be3ef76b176fbee0a094e82d97a611f9c950/*.arrow"
            )
        )
    )

    cbo_and_3p_for_set = []
    user_ids = []
    song_hashes = []
    difficulties = []
    shard_idxs, datapoint_idxs = index_pair_set.T

    for shard_idx, datapoint_idx in zip(shard_idxs, datapoint_idxs):
        shard_file = str(arrow_files[int(shard_idx)])
        ds = Dataset.from_file(shard_file)
        xror_unpacked = XROR.unpack(ds[int(datapoint_idx)]["xror"])
        beatmap, map_info = open_beatmap_from_unpacked_xror(xror_unpacked)
        left_handed = xror_unpacked.data["info"]["software"]["activity"].get(
            "leftHanded", False
        )
        cbo_and_3p = load_cbo_and_3p(
            xror_unpacked, beatmap, map_info, left_handed=left_handed, rescale_yes=True
        )
        cbo_and_3p_for_set.append(cbo_and_3p)

        # Collect metadata for each sample
        user_ids.append(xror_unpacked.data["info"]["user"]["id"])
        song_hashes.append(
            xror_unpacked.data["info"]["software"]["activity"]["songHash"]
        )
        difficulties.append(
            xror_unpacked.data["info"]["software"]["activity"]["difficulty"]
        )

    d = nanpad_collate_fn([cbo_and_3p_for_set])
    n = d["notes_np"].shape[0]
    res = []
    for i in range(n):
        seen_game_segments, seen_movement_segments = (
            segment_sampler.sample_for_training(
                d["notes_np"][[i]],
                d["bombs_np"][[i]],
                d["obstacles_np"][[i]],
                d["timestamps"][[i]],
                d["gt_3p_np"][[i]],
                d["lengths"][[i]],
                segment_length,
                1,
                segment_sampler_batch_size,
                stride,
                purview_sec,
                purview_notes,
                floor_time,
                firsts_only=False,
            )
        )
        notes_ = seen_game_segments.notes[:, 2]
        bombs_ = seen_game_segments.bombs[:, 2]
        obstacles_ = seen_game_segments.obstacles[:, 2]
        t_ = seen_game_segments.frames[:, 2]
        my_3p_ = seen_movement_segments.three_p[:, 2:]
        history_ = seen_movement_segments.three_p[:, :2]
        res.append((notes_, bombs_, obstacles_, t_, my_3p_, history_))

    samples = list(
        reduce(
            lambda acc, rr: [
                torch.cat([a, r], dim=0) if isinstance(a, torch.Tensor) else a
                for a, r in zip(acc, rr)
            ],
            res,
        )
    )

    # Create song_hash_and_difficulty list from collected metadata
    song_hash_and_difficulty = [f"{sh}{d}" for sh, d in zip(song_hashes, difficulties)]

    notes, bombs, obstacles, t, my_3p, history = samples

    return {
        "user_ids": user_ids,
        "song_hashes": song_hashes,
        "difficulties": difficulties,
        "song_hash_and_difficulty": song_hash_and_difficulty,
        "notes": notes,
        "bombs": bombs,
        "obstacles": obstacles,
        "t": t,
        "time": t / 60,
        "my_3p": my_3p,
        "history": history,
    }


def collect_for_user_id(all_k_df_original, user_id, n_samples=5):
    """
    Collect samples for a specific User ID.

    Args:
        all_k_df_original: DataFrame containing User ID, Shard Index, and Datapoint Index columns
        user_id: The User ID to collect samples for
        n_samples: Number of rows to sample (default: 5)

    Returns:
        sample_dict from collect_samples_from_index_pairs
    """
    good_rows = all_k_df_original[all_k_df_original["User ID"] == user_id]
    chosen_rows = good_rows.sample(n=n_samples, replace=True)
    index_pair_set = chosen_rows[["Shard Index", "Datapoint Index"]].values
    sample_dict = collect_samples_from_index_pairs(index_pair_set)
    return sample_dict


def collect_for_user_id_and_song(
    all_k_df_original, user_id, song_hash_and_difficulty, n_samples=5
):
    """
    Collect samples for a specific User ID and Song Hash and Difficulty.

    Args:
        all_k_df_original: DataFrame containing User ID, Song Hash and Difficulty, Shard Index, and Datapoint Index columns
        user_id: The User ID to collect samples for
        song_hash_and_difficulty: The Song Hash and Difficulty to collect samples for
        n_samples: Number of rows to sample (default: 5)

    Returns:
        sample_dict from collect_samples_from_index_pairs
    """
    good_rows = all_k_df_original[
        (all_k_df_original["User ID"] == user_id)
        & (all_k_df_original["Song Hash and Difficulty"] == song_hash_and_difficulty)
    ]
    chosen_rows = good_rows.sample(n=n_samples, replace=True)
    index_pair_set = chosen_rows[["Shard Index", "Datapoint Index"]].values
    sample_dict = collect_samples_from_index_pairs(index_pair_set)
    return sample_dict
