import json
import os
import zipfile

import numpy as np
import requests
from mutagen import MutagenError
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from scipy.spatial.transform import Rotation

from xror.xror import XROR


def open_xror(filename, logger):
    # XROR processing
    beatmap, song_info = open_bsmg_or_boxrr(None, filename, logger)
    notes_np, bombs_np, obstacles_np = get_xbo_np(beatmap, song_info)
    with open(filename, "rb") as f:
        file = f.read()
    xror = XROR.unpack(file)
    frames_np = np.array(xror.data["frames"])
    my_pos_expm, my_pos_sixd, timestamps = extract_3p_with_60fps(frames_np)
    # quick static scaling. eventually needs refactoring
    my_pos_expm = my_pos_expm.reshape(-1, 3, 6)
    # xz = np.nanmedian(my_pos_expm[:100, 0, [0, 2]], axis=0)
    # my_pos_expm[..., [0, 2]] -= xz
    ratio = 1.5044 / np.nanmedian(my_pos_expm[:100, 0, 1])  # head height
    # Project to head & residual space for scaling
    my_pos_expm[..., 1:, :3] -= my_pos_expm[..., [0], :3]
    my_pos_expm[..., 0, 1] *= ratio
    my_pos_expm[..., 1:, :3] *= ratio
    my_pos_expm[..., 1:, :3] += my_pos_expm[..., [0], :3]
    my_pos_expm = my_pos_expm.reshape(-1, 18)
    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "gt_3p_np": my_pos_expm,
        "timestamps": timestamps,
    }
    return d


def open_bsmg(filename, logger):
    # BSMG processing
    beatmap, song_info, song_duration = open_bsmg_or_boxrr(filename, None, logger)
    notes_np, bombs_np, obstacles_np = get_xbo_np(beatmap, song_info)
    timestamps = np.arange(0, song_duration, 1 / 60)
    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "timestamps": timestamps,
    }
    return d


def open_bsmg_or_boxrr(zip_path: str, xror_path: str):
    if xror_path is not None:
        with open(xror_path, "rb") as f:
            file = f.read()
        xror = XROR.unpack(file)
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
        level_filename = f'{difficulty}{xror.data["info"]["software"]["activity"]["mode"]}.dat'
        song_hash = xror.data["info"]["software"]["activity"]["songHash"]
        zip_name = f"{song_hash}.zip"
        data_dir = os.path.join("data", "bsmg")
        zip_path = os.path.join(data_dir, zip_name)
        os.makedirs(data_dir, exist_ok=True)

        if not os.path.exists(zip_path):
            print(f"Song hash {song_hash} not found locally. Downloading from BeatSaver...")
            beatsaver_data = get_beatsaver_json_by_hash(song_hash)
            download_url = beatsaver_data["versions"][0]["downloadURL"]
            response = requests.get(download_url)
            if response.status_code == 200:
                with open(zip_path, "wb") as file:
                    file.write(response.content)
                print(f"File downloaded successfully as {zip_path}")
            else:
                print("Error:", response.status_code)

    else:
        # level_filename = "NormalStandard.dat"
        level_filename = "ExpertPlus.dat"
    info_filename = "Info.dat"
    with zipfile.ZipFile(zip_path) as zipf:
        try:
            with zipf.open(level_filename) as f:
                song = json.load(f)
        except KeyError:
            level_filename = level_filename.replace("Standard", "")
            try:
                with zipf.open(level_filename) as f:
                    song = json.load(f)
            except KeyError as e:
                raise e

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


    return song, map_info, song_duration


def get_beatsaver_json_by_hash(song_hash):
    url = f"https://api.beatsaver.com/maps/hash/{song_hash}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        beatsaver_data = response.json()
    else:
        print("Error:", response.status_code)
        raise FileNotFoundError
    return beatsaver_data


def get_xbo_np(beatmap, song_info):
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
        notes_np[:, 0] *= 60 / song_info["_beatsPerMinute"]
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
        bombs_np[:, 0] *= 60 / song_info["_beatsPerMinute"]
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
        obstacles_np[:, 0] *= 60 / song_info["_beatsPerMinute"]
        obstacles_np[:, 4] *= 60 / song_info["_beatsPerMinute"]
    else:
        obstacles_np = np.empty((0, 9))
    return notes_np, bombs_np, obstacles_np


def extract_3p_with_60fps(frames_np):
    my_pos_expm, my_pos_sixd, timestamps = get_pos_expm(frames_np)
    # timestamp quality control
    last_zero_idx = np.where(timestamps < 1e-7)[0][-1]
    timestamps = timestamps[last_zero_idx:]
    my_pos_expm = my_pos_expm[last_zero_idx:]
    my_pos_sixd = my_pos_sixd[last_zero_idx:]
    # Linear spline reparameterization
    sixty_fps_timestamps = np.arange(0, np.max(timestamps), 1 / 60)
    # For each timestamp, identify the left and right points
    a = timestamps[None] - sixty_fps_timestamps[:, None]
    # Last occurrence of negative
    left_idxs = a.shape[-1] - np.argmax(a[..., ::-1] <= 0, axis=-1) - 1
    # First occurrence of positive
    right_idxs = np.argmax(a >= 0, axis=-1)
    left_pos_expms = my_pos_expm[left_idxs]
    right_pos_expms = my_pos_expm[right_idxs]
    left_times = timestamps[left_idxs]
    right_times = timestamps[right_idxs]
    denom = np.clip(right_times - left_times, 1e-7, None)
    slopes = (right_pos_expms - left_pos_expms) / denom[..., None]
    interpolated = left_pos_expms + slopes * (sixty_fps_timestamps - left_times)[..., None]
    my_pos_expm = interpolated
    timestamps = sixty_fps_timestamps
    return my_pos_expm, my_pos_sixd, timestamps


def get_pos_expm(frames_np):
    timestamps = frames_np[1:, ..., 0]
    part_data = frames_np[1:, ..., 1:]
    part_data = part_data.reshape(part_data.shape[0], 3, -1)
    my_pos = part_data[..., :3]
    my_quat = part_data[..., 3:]
    my_rot = Rotation.from_quat(my_quat.reshape(-1, 4))
    my_sixd = my_rot.as_matrix()[..., :2].reshape(*my_quat.shape[:-1], 6)
    my_pos_sixd = np.concatenate([my_pos, my_sixd], axis=-1)
    my_pos_sixd = my_pos_sixd.reshape(-1, 27)
    my_expm = quat_to_expm(my_quat)
    my_pos_expm = np.concatenate([my_pos, my_expm], axis=-1)
    my_pos_expm = my_pos_expm.reshape(-1, 18)
    return my_pos_expm, my_pos_sixd, timestamps


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
