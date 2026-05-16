import json
import os
import zipfile

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


def load_3p(xror, rescale_yes=False):
    # XROR processing
    # beatmap, map_info = open_beatmap_from_unpacked_xror(xror)
    frames_np = np.array(xror.data["frames"])
    if frames_np.shape[0] < 2048:
        raise ValueError("XROR file has less than 2048 frames")

    my_pos_quat, my_pos_sixd, timestamps = extract_3p_with_60fps(frames_np)
    my_pos_quat = my_pos_quat.reshape(-1, 3, 7)
    if rescale_yes:
        ratio = 1.5044 / np.nanmedian(my_pos_quat[:, 0, 1])  # head height
        difference = 1.5044 - np.nanmedian(my_pos_quat[:, 0, 1])  # head height
    else:
        ratio = 1
        difference = 0

    my_pos_sixd = my_pos_sixd.reshape(-1, 3, 9)
    my_pos_sixd[..., 1] += difference
    my_pos_quat[..., 1] += difference

    median_xys = np.nanmedian(my_pos_sixd[:, [0], [0, 2]], axis=0)
    my_pos_sixd[..., [0, 2]] -= median_xys[None]
    my_pos_quat[..., [0, 2]] -= median_xys[None]

    # correct for strange jumps
    for i in [2]:
        try:
            kde = gaussian_kde(my_pos_sixd[..., 0, i])
            ls = np.linspace(-1, 1, 1000)
            dens = kde.evaluate(ls)
            peak_idxs = find_peaks(dens, distance=1000)[0]
            peaks = ls[peak_idxs]
            peak_diffs = np.abs(peaks[None] - my_pos_sixd[..., 0, i][:, None])
            relevant_peak_idxs = np.argmin(peak_diffs, axis=-1)
            my_pos_sixd[..., i] -= peaks[relevant_peak_idxs][..., None]
            my_pos_quat[..., i] -= peaks[relevant_peak_idxs][..., None]
        except Exception as e:
            pass

    my_pos_sixd = my_pos_sixd.reshape(-1, 27)
    my_pos_quat = my_pos_quat.reshape(-1, 21)

    return my_pos_quat, my_pos_sixd, timestamps


def load_cbo_and_3p(xror, beatmap, map_info, left_handed=False, rescale_yes=False):
    # XROR processing
    left_handed_yes = left_handed
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info, left_handed_yes)
    xyzquats, my_pos_sixd, timestamps = load_3p(xror, rescale_yes=rescale_yes)
    # In case XROR 3p trajectory is truncated, cut CBO to match
    song_duration = timestamps.max()
    notes_np = notes_np[notes_np[:, 0] <= song_duration]
    bombs_np = bombs_np[bombs_np[:, 0] <= song_duration]
    obstacles_np = obstacles_np[obstacles_np[:, 0] <= song_duration]
    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "xyzquats": xyzquats,
        "gt_3p_np": my_pos_sixd,
        "timestamps": timestamps,
        "xror_info": xror.data["info"],
    }
    return d


def open_xror(filename):
    # XROR processing
    (
        beatmap,
        song_info,
        song_duration,
    ) = open_beatmap_from_bsmg_or_boxrr(None, filename)
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, song_info)
    with open(filename, "rb") as f:
        file = f.read()
    xror = XROR.unpack(file)
    frames_np = np.array(xror.data["frames"])
    my_pos_quat, my_pos_sixd, timestamps = extract_3p_with_60fps(frames_np)
    # quick static scaling. eventually needs refactoring
    my_pos_quat = my_pos_quat.reshape(-1, 3, 7)
    # xz = np.nanmedian(my_pos_expm[:100, 0, [0, 2]], axis=0)
    # my_pos_expm[..., [0, 2]] -= xz
    ratio = 1.5044 / np.nanmedian(my_pos_quat[-100:, 0, 1])  # head height
    # Project to head & residual space for scaling
    my_pos_quat[..., 1:, :3] -= my_pos_quat[..., [0], :3]
    my_pos_quat[..., 0, 1] *= ratio
    my_pos_quat[..., 1:, :3] *= ratio
    my_pos_quat[..., 1:, :3] += my_pos_quat[..., [0], :3]
    my_pos_quat = my_pos_quat.reshape(-1, 21)

    my_pos_sixd = my_pos_sixd.reshape(-1, 3, 9)
    my_pos_sixd[..., 1:, :3] -= my_pos_sixd[..., [0], :3]
    my_pos_sixd[..., 0, 1] *= ratio
    my_pos_sixd[..., 1:, :3] *= ratio
    my_pos_sixd[..., 1:, :3] += my_pos_sixd[..., [0], :3]
    my_pos_sixd = my_pos_sixd.reshape(-1, 27)

    # nan_yes = np.isnan(my_pos_expm)
    # my_pos_expm[nan_yes] = 0

    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        # "gt_3p_np": my_pos_expm,
        "gt_3p_np": my_pos_sixd,
        "timestamps": timestamps,
    }
    return d


def open_bsmg(filename, difficulty):
    # BSMG processing
    beatmap, map_info, song_duration = open_beatmap_from_bsmg_or_boxrr(filename, None, difficulty)
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
    timestamps = np.arange(0, song_duration, 1 / 60)
    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "timestamps": timestamps,
    }
    return d, beatmap, map_info


def open_beatmap_from_bsmg_or_boxrr(zip_path: str, xror_path: str, difficulty: str):
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
        # Load map info
        try:
            with zipf.open(info_filename) as f:
                map_info = json.load(f)
        except KeyError:
            info_filename = "info.dat"
            try:
                with zipf.open(info_filename) as f:
                    map_info = json.load(f)
            except KeyError as e:
                raise e

        song_filename = map_info["_songFilename"]
        try:
            with zipf.open(song_filename) as f:
                try:
                    audio = OggVorbis(f)
                except MutagenError:
                    audio = OggOpus(f)
            song_duration = audio.info.length
        except KeyError as e:
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
            except KeyError as e:
                # In case the song doesn't exist, use a heuristic, i.e. the last note/bomb/obstacle time
                # pass
                raise e
        try:
            with zipf.open(level_filename) as f:
                beatmap = json.load(f)
        except KeyError:
            level_filename = level_filename.replace("Standard", "")
            try:
                with zipf.open(level_filename) as f:
                    beatmap = json.load(f)
            except KeyError as e:
                raise e

    return beatmap, map_info, song_duration


def open_beatmap_from_raw_xror(xror_raw):
    xror = XROR.unpack(xror_raw)
    beatmap, map_info = open_beatmap_from_unpacked_xror(xror)
    return beatmap, map_info


def open_beatmap_from_unpacked_xror(xror):
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


def get_beatsaver_data(song_hashes):
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


def get_cbo_np(beatmap, map_info, left_handed_yes=False):
    # For tempo conversions
    # Parse bpm events, whether given as _events or bpmEvents

    base_bpm = map_info["_beatsPerMinute"]
    bpm_changes = [(0, base_bpm)]
    # print(beatmap.keys())
    if "bpmEvents" in beatmap.keys():
        for bm in beatmap["bpmEvents"]:
            b = bm["b"]
            m = bm["m"]
            if m == 0:
                m = base_bpm
            bpm_changes.append((b, m))
    # elif "_BPMChanges" in beatmap.keys():
    #     for bm in beatmap["_BPMChanges"]:
    #         b = bm["_time"]
    #         m = bm["_BPM"]
    #         if m == 0:
    #             m = base_bpm
    #         bpm_changes.append((b, m))
    elif "_events" in beatmap.keys():
        for event in beatmap["_events"]:
            if event["_type"] == 100:
                b = event["_time"]
                m = event["_floatValue"]
                if m == 0:
                    m = base_bpm
                bpm_changes.append((b, m))

    if len(bpm_changes) == 1:
        # bpm_changes.append((0.0, map_info["_beatsPerMinute"]))
        bpm_changes.append((0.0, base_bpm))
    bpm_changes = np.array(bpm_changes, dtype=float)
    sec_per_beat = 60.0 / bpm_changes[..., 1]
    # cumulative seconds up to each tempo change
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
        # bombs_np[:, 0] *= 60 / map_info["_beatsPerMinute"]
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
        # obstacles_np[:, 0] *= 60 / map_info["_beatsPerMinute"]
        # obstacles_np[:, 4] *= 60 / map_info["_beatsPerMinute"]
        obstacles_np = np.stack(obstacles_np, axis=0)
        beat_idx = np.searchsorted(bpm_changes[..., 0], obstacles_np[..., 0], side="left") - 1
        intercept = cumulative_sec[beat_idx]
        slope = sec_per_beat[beat_idx]
        obstacles_np[:, 0] = intercept + slope * (obstacles_np[:, 0] - bpm_changes[beat_idx, 0])
        obstacles_np[:, 4] = intercept + slope * (obstacles_np[:, 4] - bpm_changes[beat_idx, 0])

    else:
        obstacles_np = np.empty((0, 9))

    # Handle left-handedness
    if left_handed_yes:
        # x position
        notes_np[:, 3] = 3 - notes_np[:, 3]
        # color
        notes_np[:, 5] = 1 - notes_np[:, 5]
        # direction
        left_yes = (notes_np[:, 6] == 2) | (notes_np[:, 6] == 4) | (notes_np[:, 6] == 6)
        right_yes = (notes_np[:, 6] == 3) | (notes_np[:, 6] == 5) | (notes_np[:, 6] == 7)
        notes_np[left_yes, 6] += 1
        notes_np[right_yes, 6] -= 1
        # bomb x position
        bombs_np[:, 3] = 3 - bombs_np[:, 3]
        # obstacle x position
        obstacles_np[:, 5] = 3 - obstacles_np[:, 5]

    return notes_np, bombs_np, obstacles_np


def extract_3p_with_60fps(frames_np):
    my_pos_quat, my_pos_sixd, timestamps = get_pos_expm(frames_np)
    # timestamp quality control
    last_zero_idx = np.where(timestamps < 1e-7)[0][-1]
    timestamps = timestamps[last_zero_idx:]
    my_pos_quat = my_pos_quat[last_zero_idx:]
    my_pos_sixd = my_pos_sixd[last_zero_idx:]
    # Currently warping takes way too long....

    # Linear spline reparameterization
    sixty_fps_timestamps = np.arange(0, np.max(timestamps), 1 / 60)

    # Use bucketization to snap XROR timestamps to the 60fps version
    left_idxs = np.clip(np.searchsorted(timestamps, sixty_fps_timestamps, side="left") - 1, 0, timestamps.shape[0] - 1)
    right_idxs = np.clip(left_idxs + 1, 0, timestamps.shape[0] - 1)

    left_stamps = timestamps[left_idxs]
    center_stamps = sixty_fps_timestamps
    right_stamps = timestamps[right_idxs]

    # Linear interpolation for my_pos_sixd (unchanged)
    all_3p = my_pos_sixd
    left_3p = all_3p[left_idxs]
    right_3p = all_3p[right_idxs]
    slope = (right_3p - left_3p) / (right_stamps - left_stamps)[:, None]
    interpolated = left_3p + slope * (center_stamps - left_stamps)[:, None]

    # Interpolate my_pos_quat: linear for xyz, SLERP for quaternion
    # my_pos_quat has shape (T, 3, 7) = [pos(3), quat(4)] per part
    denom = right_stamps - left_stamps
    t = np.where(denom > 1e-7, (center_stamps - left_stamps) / denom, 0.0)

    left_pq = my_pos_quat[left_idxs]   # (T_60fps, 3, 7)
    right_pq = my_pos_quat[right_idxs]  # (T_60fps, 3, 7)

    # Linear interpolation for position (xyz)
    left_pos = left_pq[..., :3]
    right_pos = right_pq[..., :3]
    interp_pos = left_pos + t[:, None, None] * (right_pos - left_pos)

    # SLERP for quaternion (x, y, z, w)
    left_quat = left_pq[..., 3:]   # (T_60fps, 3, 4)
    right_quat = right_pq[..., 3:]  # (T_60fps, 3, 4)
    # Reshape for slerp: (T_60fps * 3, 4)
    left_quat_flat = left_quat.reshape(-1, 4)
    right_quat_flat = right_quat.reshape(-1, 4)
    t_flat = np.repeat(t, 3)  # repeat t for each of the 3 parts
    interp_quat_flat = slerp_quaternions(left_quat_flat, right_quat_flat, t_flat)
    interp_quat = interp_quat_flat.reshape(-1, 3, 4)

    # Combine interpolated position and quaternion
    interp_pos_quat = np.concatenate([interp_pos, interp_quat], axis=-1)  # (T_60fps, 3, 7)
    interp_pos_quat = interp_pos_quat.reshape(-1, 21)  # flatten to (T_60fps, 21)

    return interp_pos_quat, interpolated, sixty_fps_timestamps

    #
    # batch_size = 1024
    # idxs = np.arange(sixty_fps_timestamps.shape[0])
    # n_batches = idxs.shape[0] // batch_size
    # if n_batches > 0:
    #     batch_idxs = np.array_split(idxs, n_batches)
    # else:
    #     batch_idxs = [idxs]
    # batch_pos_expms = []
    # batch_pos_sixds = []
    # for batch_i in batch_idxs:
    #     batch_sixties = sixty_fps_timestamps[batch_i]
    #     # timestamp_yes = np.logical_and(batch_sixties.min() - 1 <= timestamps, timestamps <= batch_sixties.max() + 1)
    #     # timestamp_yes = batch_sixties.min() - 1 <= timestamps
    #     # timestamp_yes = timestamps <= batch_sixties.max()
    #
    #     # For each timestamp, identify the left and right points
    #     # good_timestamps = timestamps[timestamp_yes]
    #     good_timestamps = timestamps
    #     for my_3p, res in [[my_pos_expm, batch_pos_expms], [my_pos_sixd, batch_pos_sixds]]:
    #         # good_3p = my_3p[timestamp_yes]
    #         good_3p = my_3p
    #
    #         a = good_timestamps[None] - batch_sixties[:, None]
    #         # Last occurrence of negative
    #         left_idxs = a.shape[-1] - np.argmax(a[..., ::-1] <= 0, axis=-1) - 1
    #         # First occurrence of positive
    #         right_idxs = np.argmax(a >= 0, axis=-1)
    #         left = good_3p[left_idxs]
    #         right = good_3p[right_idxs]
    #         left_times = good_timestamps[left_idxs]
    #         right_times = good_timestamps[right_idxs]
    #         denom = np.clip(right_times - left_times, 1e-7, None)
    #         slopes = (right - left) / denom[..., None]
    #         interpolated = left + slopes * (batch_sixties - left_times)[..., None]
    #         res.append(interpolated)
    #
    # my_pos_expm = np.concatenate(batch_pos_expms, axis=0)
    # my_pos_sixd = np.concatenate(batch_pos_sixds, axis=0)
    # timestamps = sixty_fps_timestamps

    # return my_pos_expm, my_pos_sixd, timestamps


def get_pos_expm(frames_np):
    timestamps = frames_np[1:, ..., 0]
    part_data = frames_np[1:, ..., 1:]
    part_data = part_data.reshape(part_data.shape[0], 3, -1)
    my_pos = part_data[..., :3]
    my_quat = part_data[..., 3:]
    my_rot = Rotation.from_quat(my_quat.reshape(-1, 4))
    my_sixd = my_rot.as_matrix()[..., :2].swapaxes(-2, -1).reshape(*my_quat.shape[:-1], 6)
    my_pos_sixd = np.concatenate([my_pos, my_sixd], axis=-1)
    my_pos_sixd = my_pos_sixd.reshape(-1, 27)
    my_expm = quat_to_expm(my_quat)
    my_pos_expm = np.concatenate([my_pos, my_expm], axis=-1)
    my_pos_expm = my_pos_expm.reshape(-1, 18)
    my_pos_quat = np.concatenate([my_pos, my_quat], axis=-1)
    return my_pos_quat, my_pos_sixd, timestamps


def quat_to_expm(quat: np.ndarray, eps: float = 1e-8):
    """
    Quaternion is (x, y, z, w)
    """
    im = quat[..., :3]
    im_norm = np.linalg.norm(im, axis=-1)
    half_angle = np.arctan2(im_norm, quat[..., 3])
    expm = np.where(
        im_norm[..., None] < eps,
        im,
        half_angle[..., None] * (im / im_norm[..., None]),
    )
    return expm


def slerp_quaternions(q0: np.ndarray, q1: np.ndarray, t: np.ndarray, eps: float = 1e-6):
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
    # Normalize inputs
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)

    # Compute dot product
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)

    # Flip q1 where dot < 0 to ensure shortest path
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    # Clamp dot to valid range for arccos
    dot = np.clip(dot, -1.0, 1.0)

    # Compute angle
    theta = np.arccos(dot)

    # Expand t for broadcasting with quaternion components
    t_expanded = t[..., None] if t.ndim < q0.ndim else t

    # Use slerp where angle is large enough, else use normalized lerp
    sin_theta = np.sin(theta)
    use_slerp = sin_theta > eps

    # SLERP formula: (sin((1-t)*theta)/sin(theta)) * q0 + (sin(t*theta)/sin(theta)) * q1
    s0 = np.where(use_slerp, np.sin((1.0 - t_expanded) * theta) / sin_theta, 1.0 - t_expanded)
    s1 = np.where(use_slerp, np.sin(t_expanded * theta) / sin_theta, t_expanded)

    result = s0 * q0 + s1 * q1

    # Normalize the result
    result = result / np.linalg.norm(result, axis=-1, keepdims=True)
    return result
