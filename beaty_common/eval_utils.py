import numpy as np
import torch

from beaty_common.bsmg_xror_utils import device, get_cbo_np
from beaty_common.data_utils import sample_for_evaluation
from beaty_common.train_utils import nanpad_collate_fn
from vendor.torch_saber import TorchSaber

EVALUATION_SEGMENT_LENGTH = 72
EVALUATION_SAMPLE_COUNT = 1
EVALUATION_MINIBATCH_SIZE = 512
EVALUATION_SEGMENT_STRIDE = 1
EVALUATION_LOOKAHEAD_SECONDS = 4.0
EVALUATION_PURVIEW_NOTES = 80
EVALUATION_FLOOR_TIME = -0.5
SIMULATION_BATCH_SIZE = 1024
PLAYER_HEIGHT = 1.5044


def evaluate_3p_on_map(trajectory_3p, difficulty, characteristic, beatmap, map_info, song_duration, left_handed=False):
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
    timestamps = np.arange(0, song_duration, 1 / 60)
    min_length = min(trajectory_3p.shape[2], timestamps.shape[0])
    timestamps = timestamps[:min_length]
    trajectory_3p = trajectory_3p[:, :, :min_length]
    collated = nanpad_collate_fn(
        [[{
            "notes_np": notes_np,
            "bombs_np": bombs_np,
            "obstacles_np": obstacles_np,
            "timestamps": timestamps,
        }]]
    )
    collated = {
        name: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for name, value in collated.items()
    }
    sampled_map = sample_for_evaluation(
        collated["notes_np"],
        collated["bombs_np"],
        collated["obstacles_np"],
        collated["timestamps"],
        collated["lengths"],
        collated["timestamps"].shape[1],
        EVALUATION_SAMPLE_COUNT,
        EVALUATION_MINIBATCH_SIZE,
        EVALUATION_SEGMENT_STRIDE,
        EVALUATION_LOOKAHEAD_SECONDS,
        EVALUATION_PURVIEW_NOTES,
        EVALUATION_FLOOR_TIME,
    )
    note_jump_speed = get_njs(map_info, difficulty, characteristic)

    (
        note_appeared_mask,
        bomb_appeared_mask,
        bomb_collided_mask,
        head_obstacle_signed_distances,
        good_cut_mask,
        bad_cut_mask,
        cut_angles,
    ) = TorchSaber.simulate(
        trajectory_3p[:, :, [0]],
        trajectory_3p,
        sampled_map.notes[:, None],
        sampled_map.bombs[:, None],
        sampled_map.obstacles[:, None],
        sampled_map.note_ids[:, None],
        sampled_map.bomb_ids[:, None],
        sampled_map.obstacle_ids[:, None],
        collated["notes_np"],
        collated["bombs_np"],
        collated["obstacles_np"],
        PLAYER_HEIGHT,
        note_jump_speed,
        SIMULATION_BATCH_SIZE,
    )
    note_alive_mask = torch.where(
        torch.isnan(collated["notes_np"][..., 0]),
        torch.as_tensor(False),
        torch.as_tensor(True),
    )[..., None, :]
    (
        normalized_score,
        n_opportunities,
        n_hits,
        n_misses,
        n_goods,
        collided_note_mask,
        bomb_penalty,
        obstacle_penalty,
    ) = TorchSaber.evaluate(
        note_alive_mask.to(device),
        note_appeared_mask.to(device),
        bomb_appeared_mask.to(device),
        bomb_collided_mask.to(device),
        head_obstacle_signed_distances.to(device),
        good_cut_mask.to(device),
        bad_cut_mask.to(device),
        cut_angles.to(device),
        batch_size=SIMULATION_BATCH_SIZE,
    )
    return normalized_score, n_opportunities, n_goods, n_hits, n_misses


def get_njs(map_info, difficulty, characteristic):
    found = False
    njs = 18
    for difficulty_beatmap_set in map_info["_difficultyBeatmapSets"]:
        if difficulty_beatmap_set["_beatmapCharacteristicName"] == characteristic:
            for difficulty_beatmap in difficulty_beatmap_set["_difficultyBeatmaps"]:
                if difficulty_beatmap["_difficulty"] == difficulty:
                    found = True
                    njs = difficulty_beatmap["_noteJumpMovementSpeed"]
                    break
    assert found
    if njs == 0:
        njs = map_info["_beatsPerMinute"] / 10
    return njs
