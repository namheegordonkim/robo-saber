import json
import os
import zipfile
from typing import Any

import numpy as np
import requests
import torch
from mutagen import MutagenError
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.stats import gaussian_kde

from vendor.xror.xror import XROR

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_3p(xror: Any, rescale_yes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    frames_np = np.array(xror.data["frames"])
    if frames_np.shape[0] < 2048:
        raise ValueError("XROR file has less than 2048 frames")

    trajectory_quat, trajectory_sixd, timestamps = extract_3p_with_60fps(frames_np)
    trajectory_quat = trajectory_quat.reshape(-1, 3, 7)
    if rescale_yes:
        height_shift = 1.5044 - np.nanmedian(trajectory_quat[:, 0, 1])
    else:
        height_shift = 0

    trajectory_sixd = trajectory_sixd.reshape(-1, 3, 9)
    trajectory_sixd[..., 1] += height_shift
    trajectory_quat[..., 1] += height_shift

    median_head_xy = np.nanmedian(trajectory_sixd[:, [0], [0, 2]], axis=0)
    trajectory_sixd[..., [0, 2]] -= median_head_xy[None]
    trajectory_quat[..., [0, 2]] -= median_head_xy[None]

    for axis_index in [2]:
        try:
            kde = gaussian_kde(trajectory_sixd[..., 0, axis_index])
            grid = np.linspace(-1, 1, 1000)
            density = kde.evaluate(grid)
            peak_indices = find_peaks(density, distance=1000)[0]
            peaks = grid[peak_indices]
            peak_diffs = np.abs(peaks[None] - trajectory_sixd[..., 0, axis_index][:, None])
            closest_peak_indices = np.argmin(peak_diffs, axis=-1)
            trajectory_sixd[..., axis_index] -= peaks[closest_peak_indices][..., None]
        except Exception:
            pass

    return trajectory_sixd.reshape(-1, 27), timestamps


def load_cbo_and_3p(
    xror: Any,
    beatmap: dict[str, Any],
    map_info: dict[str, Any],
    left_handed: bool = False,
    rescale_yes: bool = False,
) -> dict[str, Any]:
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info, left_handed)
    trajectory_sixd, timestamps = load_3p(xror, rescale_yes=rescale_yes)
    song_duration = timestamps.max()
    notes_np = notes_np[notes_np[:, 0] <= song_duration]
    bombs_np = bombs_np[bombs_np[:, 0] <= song_duration]
    obstacles_np = obstacles_np[obstacles_np[:, 0] <= song_duration]
    return {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "gt_3p_np": trajectory_sixd,
        "timestamps": timestamps,
        "xror_info": xror.data["info"],
    }


def open_bsmg(filename: str, difficulty: str) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    beatmap, map_info, song_duration = open_beatmap_from_bsmg_or_boxrr(filename, None, difficulty)
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
    timestamps = np.arange(0, song_duration, 1 / 60)
    return {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "timestamps": timestamps,
    }, beatmap, map_info


def open_beatmap_from_bsmg_or_boxrr(
    zip_path: str,
    xror_path: str | None,
    difficulty: str,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    if xror_path is not None:
        with open(xror_path, "rb") as f:
            file = f.read()
        return open_beatmap_from_raw_xror(file)
    level_filename = f"{difficulty}Standard.dat"
    if not os.path.exists(zip_path):
        song_hash = os.path.splitext(os.path.basename(zip_path))[0]
        print(f"Song hash {song_hash} not found locally. Downloading from BeatSaver...")
        download_url = get_beatsaver_data([song_hash])[song_hash]["versions"][0]["downloadURL"]
        response = requests.get(download_url)
        if response.status_code != 200:
            print("Error:", response.status_code)
            raise FileNotFoundError
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully as {zip_path}")
    info_filename = "Info.dat"
    with zipfile.ZipFile(zip_path) as zipf:
        try:
            with zipf.open(info_filename) as f:
                map_info = json.load(f)
        except KeyError:
            info_filename = "info.dat"
            with zipf.open(info_filename) as f:
                map_info = json.load(f)

        song_filename = map_info["_songFilename"]
        try:
            with zipf.open(song_filename) as f:
                try:
                    audio = OggVorbis(f)
                except MutagenError:
                    audio = OggOpus(f)
            song_duration = audio.info.length
        except KeyError:
            try:
                song_filenames = [fl.filename for fl in zipf.filelist if "song" in fl.filename.lower()]
                if len(song_filenames) == 0:
                    raise KeyError
                song_filename = song_filenames[0]

                with zipf.open(song_filename) as f:
                    try:
                        audio = OggVorbis(f)
                    except MutagenError:
                        audio = OggOpus(f)
                song_duration = audio.info.length
            except KeyError:
                raise
        try:
            with zipf.open(level_filename) as f:
                beatmap = json.load(f)
        except KeyError:
            level_filename = level_filename.replace("Standard", "")
            with zipf.open(level_filename) as f:
                beatmap = json.load(f)

    return beatmap, map_info, song_duration


def open_beatmap_from_raw_xror(xror_raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    xror = XROR.unpack(xror_raw)
    beatmap, map_info = open_beatmap_from_unpacked_xror(xror)
    return beatmap, map_info


def open_beatmap_from_unpacked_xror(xror: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        difficulty = xror.data["info"]["software"]["activity"]["difficulty"]
    except KeyError:
        raise KeyError("Difficulty not found in XROR file")

    if type(difficulty) == int:
        difficulty = {
            1: "Easy",
            3: "Normal",
            5: "Hard",
            7: "Expert",
            9: "ExpertPlus",
        }[difficulty]
    song_hash = xror.data["info"]["software"]["activity"]["songHash"]
    zip_path = os.path.join("data", "BeatSaver", f"{song_hash}.zip")
    return open_beatmap_from_bsmg_or_boxrr(zip_path, None, difficulty)[:2]


def get_beatsaver_data(song_hashes: list[str]) -> dict[str, Any]:
    song_hashes = [str(song_hash) for song_hash in song_hashes]
    if len(song_hashes) == 1:
        url = f"https://api.beatsaver.com/maps/hash/{song_hashes[0]}"
    else:
        url = f"https://api.beatsaver.com/maps/hash/{','.join(song_hashes)}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        beatsaver_data = response.json()
    else:
        print("Error:", response.status_code)
        raise FileNotFoundError
    if len(song_hashes) == 1:
        return {song_hashes[0]: beatsaver_data}
    return beatsaver_data


def get_cbo_np(
    beatmap: dict[str, Any],
    map_info: dict[str, Any],
    left_handed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_bpm = map_info["_beatsPerMinute"]
    bpm_changes = [(0, base_bpm)]
    if "bpmEvents" in beatmap.keys():
        for bm in beatmap["bpmEvents"]:
            b = bm["b"]
            m = bm["m"]
            if m == 0:
                m = base_bpm
            bpm_changes.append((b, m))
    elif "_events" in beatmap.keys():
        for event in beatmap["_events"]:
            if event["_type"] == 100:
                b = event["_time"]
                m = event["_floatValue"]
                if m == 0:
                    m = base_bpm
                bpm_changes.append((b, m))

    if len(bpm_changes) == 1:
        bpm_changes.append((0.0, base_bpm))
    bpm_changes = np.array(bpm_changes, dtype=float)
    sec_per_beat = 60.0 / bpm_changes[..., 1]
    cumulative_sec = np.zeros_like(bpm_changes[..., 0], dtype=float)
    durations_beats = bpm_changes[1:, 0] - bpm_changes[:-1, 0]
    durations_sec = durations_beats * sec_per_beat[:-1]
    cumulative_sec[1:] = np.cumsum(durations_sec)

    notes_np = []
    bombs_np = []
    if "colorNotes" in beatmap:
        for i in range(len(beatmap["colorNotes"])):
            b = beatmap["colorNotes"][i].get("b", 10)
            if "colorNotesData" in beatmap:
                note_data = beatmap["colorNotesData"][i]
            else:
                note_data = beatmap["colorNotes"][i]
            r = note_data.get("r", 0)
            i = note_data.get("i", 0)
            x = note_data.get("x", 1)
            y = note_data.get("y", 0)
            c = note_data.get("c", 0)
            d = note_data.get("d", 1)
            a = note_data.get("a", 0)
            note_np = np.array([b, r, i, x, y, c, d, a], dtype=float)
            notes_np.append(note_np)
    else:
        for note in beatmap["_notes"]:
            b = note.get("_time", 10)
            r = 0
            i = 0
            x = note.get("_lineIndex", 1)
            y = note.get("_lineLayer", 0)
            c = note.get("_type", 0)
            d = note.get("_cutDirection", 1)
            a = 0
            if c == 3:
                bomb_np = np.array([b, r, i, x, y], dtype=float)
                bombs_np.append(bomb_np)
            else:
                note_np = np.array([b, r, i, x, y, c, d, a], dtype=float)
                notes_np.append(note_np)
    if len(notes_np) > 0:
        notes_np = np.stack(notes_np, axis=0)
        beat_idx = np.searchsorted(bpm_changes[..., 0], notes_np[..., 0], side="left") - 1
        intercept = cumulative_sec[beat_idx]
        slope = sec_per_beat[beat_idx]
        notes_np[:, 0] = intercept + slope * (notes_np[:, 0] - bpm_changes[beat_idx, 0])
    else:
        notes_np = np.empty((0, 8))
    if "bombNotes" in beatmap:
        for i in range(len(beatmap["bombNotes"])):
            b = beatmap["bombNotes"][i].get("b", 10)
            if "colorNotesData" in beatmap:
                bomb_data = beatmap["bombNotesData"][i]
            else:
                bomb_data = beatmap["bombNotes"][i]
            r = bomb_data.get("r", 0)
            i = bomb_data.get("i", 0)
            x = bomb_data.get("x", 1)
            y = bomb_data.get("y", 0)
            bomb_np = np.array([b, r, i, x, y], dtype=float)
            bombs_np.append(bomb_np)
    if len(bombs_np) > 0:
        bombs_np = np.stack(bombs_np, axis=0)
        beat_idx = np.searchsorted(bpm_changes[..., 0], bombs_np[..., 0], side="left") - 1
        intercept = cumulative_sec[beat_idx]
        slope = sec_per_beat[beat_idx]
        bombs_np[:, 0] = intercept + slope * (bombs_np[:, 0] - bpm_changes[beat_idx, 0])
    else:
        bombs_np = np.empty((0, 5))
    obstacles_np = []
    if "obstacles" in beatmap:
        for i in range(len(beatmap["obstacles"])):
            b = beatmap["obstacles"][i].get("b", 10)
            if "obstaclesData" in beatmap:
                obstacle_data = beatmap["obstaclesData"][i]
            else:
                obstacle_data = beatmap["obstacles"][i]
            r = obstacle_data.get("r", 0)
            i = obstacle_data.get("i", 0)
            t = 2
            d = obstacle_data.get("d", 5)
            x = obstacle_data.get("x", 1)
            y = obstacle_data.get("y", 0)
            w = obstacle_data.get("w", 1)
            h = obstacle_data.get("h", 5)
            obstacle_np = np.array([b, r, i, t, d, x, y, w, h], dtype=float)
            obstacles_np.append(obstacle_np)
    else:
        for obstacle in beatmap["_obstacles"]:
            b = obstacle.get("_time", 10)
            r = 0
            i = 0
            t = obstacle.get("_type", 2)
            d = obstacle.get("_duration", 5)
            x = obstacle.get("_lineIndex", 1)
            y = obstacle.get("_lineLayer", 0)
            w = obstacle.get("_width", 1)
            h = obstacle.get("_height", 5)
            if t == 1:
                y = 2
                h = 3
            obstacle_np = np.array([b, r, i, t, d, x, y, w, h], dtype=float)
            obstacles_np.append(obstacle_np)
    if len(obstacles_np) > 0:
        obstacles_np = np.stack(obstacles_np, axis=0)
        beat_idx = np.searchsorted(bpm_changes[..., 0], obstacles_np[..., 0], side="left") - 1
        intercept = cumulative_sec[beat_idx]
        slope = sec_per_beat[beat_idx]
        obstacles_np[:, 0] = intercept + slope * (obstacles_np[:, 0] - bpm_changes[beat_idx, 0])
        obstacles_np[:, 4] = intercept + slope * (obstacles_np[:, 4] - bpm_changes[beat_idx, 0])
    else:
        obstacles_np = np.empty((0, 9))

    if left_handed:
        notes_np[:, 3] = 3 - notes_np[:, 3]
        notes_np[:, 5] = 1 - notes_np[:, 5]
        left_directions = (notes_np[:, 6] == 2) | (notes_np[:, 6] == 4) | (notes_np[:, 6] == 6)
        right_directions = (notes_np[:, 6] == 3) | (notes_np[:, 6] == 5) | (notes_np[:, 6] == 7)
        notes_np[left_directions, 6] += 1
        notes_np[right_directions, 6] -= 1
        bombs_np[:, 3] = 3 - bombs_np[:, 3]
        obstacles_np[:, 5] = 3 - obstacles_np[:, 5]

    return notes_np, bombs_np, obstacles_np


def extract_3p_with_60fps(frames_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    my_pos_quat, my_pos_sixd, timestamps = get_pos_sixd(frames_np)
    last_zero_idx = np.where(timestamps < 1e-7)[0][-1]
    timestamps = timestamps[last_zero_idx:]
    my_pos_quat = my_pos_quat[last_zero_idx:]
    my_pos_sixd = my_pos_sixd[last_zero_idx:]
    sixty_fps_timestamps = np.arange(0, np.max(timestamps), 1 / 60)

    left_idxs = np.clip(np.searchsorted(timestamps, sixty_fps_timestamps, side="left") - 1, 0, timestamps.shape[0] - 1)
    right_idxs = np.clip(left_idxs + 1, 0, timestamps.shape[0] - 1)

    left_stamps = timestamps[left_idxs]
    center_stamps = sixty_fps_timestamps
    right_stamps = timestamps[right_idxs]

    all_3p = my_pos_sixd
    left_3p = all_3p[left_idxs]
    right_3p = all_3p[right_idxs]
    slope = (right_3p - left_3p) / (right_stamps - left_stamps)[:, None]
    interpolated = left_3p + slope * (center_stamps - left_stamps)[:, None]

    denom = right_stamps - left_stamps
    t = np.where(denom > 1e-7, (center_stamps - left_stamps) / denom, 0.0)

    left_pq = my_pos_quat[left_idxs]
    right_pq = my_pos_quat[right_idxs]

    left_pos = left_pq[..., :3]
    right_pos = right_pq[..., :3]
    interp_pos = left_pos + t[:, None, None] * (right_pos - left_pos)

    left_quat = left_pq[..., 3:]
    right_quat = right_pq[..., 3:]
    left_quat_flat = left_quat.reshape(-1, 4)
    right_quat_flat = right_quat.reshape(-1, 4)
    t_flat = np.repeat(t, 3)
    interp_quat_flat = slerp_quaternions(left_quat_flat, right_quat_flat, t_flat)
    interp_quat = interp_quat_flat.reshape(-1, 3, 4)

    interp_pos_quat = np.concatenate([interp_pos, interp_quat], axis=-1)
    interp_pos_quat = interp_pos_quat.reshape(-1, 21)

    return interp_pos_quat, interpolated, sixty_fps_timestamps


def get_pos_sixd(frames_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = frames_np[1:, ..., 0]
    part_data = frames_np[1:, ..., 1:]
    part_data = part_data.reshape(part_data.shape[0], 3, -1)
    my_pos = part_data[..., :3]
    my_quat = part_data[..., 3:]
    my_rot = Rotation.from_quat(my_quat.reshape(-1, 4))
    my_sixd = my_rot.as_matrix()[..., :2].swapaxes(-2, -1).reshape(*my_quat.shape[:-1], 6)
    my_pos_sixd = np.concatenate([my_pos, my_sixd], axis=-1)
    my_pos_sixd = my_pos_sixd.reshape(-1, 27)
    my_pos_quat = np.concatenate([my_pos, my_quat], axis=-1)
    return my_pos_quat, my_pos_sixd, timestamps


def slerp_quaternions(q0: np.ndarray, q1: np.ndarray, t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Vectorized SLERP for quaternions with order (x, y, z, w).
    Handles shortest-path interpolation by flipping q1 when dot(q0, q1) < 0.
    Falls back to normalized lerp for very small angles.

    Args:
        q0: Start quaternions, shape (..., 4)
        q1: End quaternions, shape (..., 4)
        t: Interpolation weights in [0, 1], shape (...) or broadcastable
        eps: Threshold below which to use linear interpolation

    Returns:
        Interpolated quaternions, shape (..., 4), normalized
    """
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)

    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    dot = np.clip(dot, -1.0, 1.0)

    theta = np.arccos(dot)

    t_expanded = t[..., None] if t.ndim < q0.ndim else t

    sin_theta = np.sin(theta)
    use_slerp = sin_theta > eps

    s0 = np.where(use_slerp, np.sin((1.0 - t_expanded) * theta) / sin_theta, 1.0 - t_expanded)
    s1 = np.where(use_slerp, np.sin(t_expanded * theta) / sin_theta, t_expanded)

    result = s0 * q0 + s1 * q1

    result = result / np.linalg.norm(result, axis=-1, keepdims=True)
    return result
