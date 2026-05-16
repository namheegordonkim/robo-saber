import numpy as np
import torch

from beaty_common.bsmg_xror_utils import open_beatmap_from_unpacked_xror, load_cbo_and_3p, device, get_cbo_np
from beaty_common.data_utils import SegmentSampler
from beaty_common.train_utils import nanpad_collate_fn
from vendor.torch_saber import TorchSaber


def evaluate_3p_on_map(_3p, difficulty, characteristic, beatmap, map_info, song_duration, left_handed=False):
    segment_sampler = SegmentSampler()
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
    # song_duration = notes_np[:, 0].max()
    timestamps = np.arange(0, song_duration, 1 / 60)
    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "timestamps": timestamps,
    }
    d = nanpad_collate_fn([[d]])
    length = d["timestamps"].shape[1]
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    game_segments = segment_sampler.sample_for_evaluation(
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        d["timestamps"],
        d["lengths"],
        length,
        1,
        512,
        1,
        4.0,
        80,
        -0.5,
    )
    # Get NJS
    njs = get_njs(map_info, difficulty, characteristic)
    note_alive_mask = torch.where(torch.isnan(d["notes_np"][..., 0]), torch.as_tensor(False), torch.as_tensor(True))[..., None, :]
    (
        note_appeared_yes_mask,
        bomb_appeared_yes_mask,
        obstacle_appeared_mask,
        bomb_collided_yes_mask,
        obstacle_collided_yes_mask,
        gc_mask,
        bc_mask,
        ms_mask,
        offset_vels,
    ) = TorchSaber.simulate(
        _3p[:, :, [0]],
        _3p,
        game_segments.notes[:, None],
        game_segments.bombs[:, None],
        game_segments.obstacles[:, None],
        game_segments.note_ids[:, None],
        game_segments.bomb_ids[:, None],
        game_segments.obstacle_ids[:, None],
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        1.5044,
        njs,
        1024,
    )
    (
        ts,
        n_opportunities,
        n_hits,
        n_misses,
        n_goods,
        collided_note_masks,
        bomb_penalty,
        obstacle_penalty,
        scores_at_collisions,
        final_good_yes,
    ) = TorchSaber.evaluate(
        note_alive_mask.to(device),
        note_appeared_yes_mask.to(device),
        bomb_appeared_yes_mask.to(device),
        bomb_collided_yes_mask.to(device),
        obstacle_collided_yes_mask.to(device),
        gc_mask.to(device),
        bc_mask.to(device),
        offset_vels.to(device),
        batch_size=1024,
    )
    return ts, n_opportunities, n_goods, n_hits, n_misses


def get_njs(map_info, difficulty, characteristic):
    found = False
    njs = 18
    for dbs in map_info["_difficultyBeatmapSets"]:
        if dbs["_beatmapCharacteristicName"] == characteristic:
            for db in dbs["_difficultyBeatmaps"]:
                if db["_difficulty"] == difficulty:
                    found = True
                    njs = db["_noteJumpMovementSpeed"]
                    break
    assert found
    if njs == 0:
        njs = map_info["_beatsPerMinute"] / 10
    return njs


def evaluate_xror_on_map(xror_unpacked, beatmap, map_info, left_handed=False, njs_offset=0.0):
    segment_sampler = SegmentSampler()
    difficulty = xror_unpacked.data["info"]["software"]["activity"]["difficulty"]
    characteristic = xror_unpacked.data["info"]["software"]["activity"]["mode"]
    d = load_cbo_and_3p(xror_unpacked, beatmap, map_info, left_handed=left_handed, rescale_yes=True)
    found = False
    njs = get_njs(map_info, difficulty, characteristic) + njs_offset
    d = nanpad_collate_fn([[d]])
    length = d["timestamps"].shape[1]
    # length = n_frames
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    # d["lengths"][:] = n_frames # cap it at here
    # d["gt_3p_np"] = d["gt_3p_np"][:, :n_frames]
    game_segments = segment_sampler.sample_for_evaluation(
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        d["timestamps"],
        d["lengths"],
        length,
        1,
        512,
        1,
        10.0,
        200,
        -0.1,
    )
    gt_3p_np = d["gt_3p_np"]
    gt_3p_np = gt_3p_np.reshape((*gt_3p_np.shape[:-1], 3, -1))
    # height = torch.nanmedian(gt_3p_np[:, -1000:, 0, 1]).item()
    # height = np.nanmedian(np.array(xror_unpacked.data['events'][5]['floatData'])[:, -1]).item()
    note_alive_mask = torch.where(torch.isnan(d["notes_np"][..., 0]), torch.as_tensor(False), torch.as_tensor(True))[..., None, :]
    # njs = x["njs"]
    # njs = 10
    n_notes = d["notes_np"].shape[1]
    # print(f"{n_notes=}")
    (
        note_appeared_yes_mask,
        bomb_appeared_yes_mask,
        obstacle_appeared_mask,
        bomb_collided_yes_mask,
        obstacle_collided_yes_mask,
        gc_mask,
        bc_mask,
        ms_mask,
        offset_vels,
    ) = TorchSaber.simulate(
        gt_3p_np[:, None, [0]],
        gt_3p_np[:, None],
        game_segments.notes[:, None],
        game_segments.bombs[:, None],
        game_segments.obstacles[:, None],
        game_segments.note_ids[:, None],
        game_segments.bomb_ids[:, None],
        game_segments.obstacle_ids[:, None],
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        1.5044,
        njs,
        512,
    )
    (
        ts,
        n_opportunities,
        n_hits,
        n_misses,
        n_goods,
        collided_note_masks,
        bomb_penalty,
        obstacle_penalty,
        scores_at_collisions,
        final_good_yes,
    ) = TorchSaber.evaluate(
        note_alive_mask.to(device),
        bomb_appeared_yes_mask.to(device),
        bomb_collided_yes_mask.to(device),
        obstacle_collided_yes_mask.to(device),
        gc_mask.to(device),
        bc_mask.to(device),
        offset_vels.to(device),
        batch_size=1024,
    )
    return ts, n_opportunities, n_goods, n_hits, n_misses
