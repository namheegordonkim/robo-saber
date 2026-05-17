import numpy as np
import torch
from typing import Any

from beaty_common.bsmg_xror_utils import device, get_cbo_np
from beaty_common.data_utils import sample_for_evaluation
from beaty_common.train_utils import nanpad_collate_fn
from beaty_common.torch_nets import MapTensors, ReplayTensors
from vendor.torch_saber import PlayerMotion, SaberSimulation, TorchSaber

EVALUATION_SEGMENT_LENGTH = 72
EVALUATION_SAMPLE_COUNT = 1
EVALUATION_MINIBATCH_SIZE = 512
EVALUATION_SEGMENT_STRIDE = 1
EVALUATION_LOOKAHEAD_SECONDS = 4.0
EVALUATION_PURVIEW_NOTES = 80
EVALUATION_FLOOR_TIME = -0.5
SIMULATION_BATCH_SIZE = 1024


def evaluate_3p_on_map(
    trajectory_3p: torch.Tensor,
    difficulty: str,
    characteristic: str,
    beatmap: dict[str, Any],
    map_info: dict[str, Any],
    song_duration: float,
    left_handed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    simulation_replay = ReplayTensors(
        sampled_map.notes[:, None],
        sampled_map.bombs[:, None],
        sampled_map.obstacles[:, None],
        note_ids=sampled_map.note_ids[:, None],
        bomb_ids=sampled_map.bomb_ids[:, None],
        obstacle_ids=sampled_map.obstacle_ids[:, None],
    )

    masks = TorchSaber.simulate(
        SaberSimulation(
            PlayerMotion(trajectory_3p[:, :, [0]], trajectory_3p),
            simulation_replay,
            MapTensors(collated["notes_np"], collated["bombs_np"], collated["obstacles_np"]),
            note_jump_speed,
        ),
        batch_size=SIMULATION_BATCH_SIZE,
    )
    note_alive_mask = torch.where(
        torch.isnan(collated["notes_np"][..., 0]),
        torch.as_tensor(False),
        torch.as_tensor(True),
    )[..., None, :]
    evaluation = TorchSaber.evaluate(
        note_alive_mask.to(device),
        masks,
        batch_size=SIMULATION_BATCH_SIZE,
    )
    score = evaluation.score
    return score.normalized_score, score.n_opportunities, score.n_goods, score.n_hits, score.n_misses


def get_njs(map_info: dict[str, Any], difficulty: str, characteristic: str) -> float:
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
