import os

import torch

from beaty_common.bsmg_xror_utils import open_bsmg
from beaty_common.data_utils import sample_for_evaluation
from beaty_common.eval_utils import get_njs
from beaty_common.pose_utils import interpolate_xyzsixd
from beaty_common.train_utils import nanpad_collate_fn, placeholder_3p_sixd
from beaty_common.torch_nets import GameTensors, MapTensors, ReplayTensors
from vendor.torch_saber import TorchSaber

PLAYER_HEIGHT = 1.5044
GENERATION_SAMPLE_COUNT = 1
GENERATION_MINIBATCH_SIZE = 512
GENERATION_SEGMENT_STRIDE = 1
DEFAULT_CHARACTERISTIC = "Standard"


def generate_3p_work(
    device,
    execution_horizon,
    sampled_song,
    trajectory_decoder,
    history_length,
    length,
    style_predictor,
    stride,
    map_profiles,
    argmax_yes,
    note_jump_speed,
    playstyle_tokens: torch.Tensor,
    playstyle_mask: torch.Tensor,
    n_cands: int,
    temperature: float,
    topk: int,
):
    generated_windows = [torch.as_tensor(placeholder_3p_sixd[None, None], device=device) for frame_index in range(history_length)]
    note_alive_mask = torch.ones(
        (sampled_song.notes.shape[0], n_cands, map_profiles.notes.shape[1]),
        dtype=torch.bool,
        device=device,
    )
    generated_candidates = [
        torch.as_tensor(placeholder_3p_sixd[None, None, None], device=device).unflatten(-1, (3, -1)).repeat_interleave(n_cands, 1)
        for frame_index in range(history_length)
    ]

    with torch.no_grad():
        for frame_index in range(length):
            history_window = torch.cat(generated_windows[-history_length:], dim=1)
            current_notes = sampled_song.notes[:, frame_index].to(device=device)
            current_bombs = sampled_song.bombs[:, frame_index].to(device=device)
            current_obstacles = sampled_song.obstacles[:, frame_index].to(device=device)

            if frame_index % (execution_horizon * stride) != 0:
                continue

            game_object_embeddings, game_object_mask, game_history_embeddings, game_history_mask = style_predictor.encode_game(
                GameTensors(current_notes, current_bombs, current_obstacles, history_window)
            )
            logits = style_predictor.predict_logits_from_embeds(
                game_object_embeddings,
                game_object_mask,
                game_history_embeddings,
                game_history_mask,
                playstyle_tokens * 1,
                playstyle_mask,
            )
            _, decoded_tokens, _, hard_samples, _ = style_predictor.sample_from_z(
                logits,
                n=n_cands if not argmax_yes else 1,
                temperature=temperature,
                topk=topk,
            )
            if argmax_yes:
                decoded_windows = trajectory_decoder.decode(decoded_tokens)
            else:
                decoded_windows = trajectory_decoder.decode(hard_samples)

            candidate_keypoints = torch.cat(
                [history_window[:, [-1], None].repeat_interleave(decoded_windows.shape[1], 1), decoded_windows],
                dim=2,
            )
            candidate_trajectories = interpolate_xyzsixd(candidate_keypoints, stride)
            candidate_trajectories = candidate_trajectories.reshape((*candidate_trajectories.shape[:-1], 3, -1))

            stop_frame = min(frame_index + candidate_trajectories.shape[-3], length)
            carry_state = history_window[:, None, [-1]].unflatten(-1, (3, -1))
            carry_state = carry_state.repeat(1, candidate_trajectories.shape[1], 1, 1, 1)
            candidate_replay = ReplayTensors(
                sampled_song.notes[:, frame_index:stop_frame][:, None].repeat_interleave(n_cands, 1),
                sampled_song.bombs[:, frame_index:stop_frame][:, None].repeat_interleave(n_cands, 1),
                sampled_song.obstacles[:, frame_index:stop_frame][:, None].repeat_interleave(n_cands, 1),
                note_ids=sampled_song.note_ids[:, frame_index:stop_frame][:, None].repeat_interleave(n_cands, 1),
                bomb_ids=sampled_song.bomb_ids[:, frame_index:stop_frame][:, None].repeat_interleave(n_cands, 1),
                obstacle_ids=sampled_song.obstacle_ids[:, frame_index:stop_frame][:, None].repeat_interleave(n_cands, 1),
            )
            (
                normalized_score,
                n_opportunities,
                n_hits,
                n_misses,
                n_goods,
                collided_note_mask,
                bomb_penalty,
                obstacle_penalty,
            ) = TorchSaber.evaluate_and_simulate(
                carry_state,
                candidate_trajectories[:, :, : stop_frame - frame_index],
                candidate_replay,
                map_profiles,
                note_alive_mask,
                PLAYER_HEIGHT,
                note_jump_speed,
            )

            candidate_score = 1.0 + normalized_score[0] - 10.0 * bomb_penalty[0] + 10.0 * obstacle_penalty[0]
            best_candidate_index = torch.argmax(candidate_score, dim=-1)

            selected_collision_mask = collided_note_mask[
                torch.arange(collided_note_mask.shape[0]),
                best_candidate_index,
            ][:, None]
            note_alive_mask = note_alive_mask & ~selected_collision_mask
            best_decoded_window = decoded_windows[torch.arange(decoded_windows.shape[0]), best_candidate_index]

            generated_candidates.append(decoded_windows.reshape(*decoded_windows.shape[:-1], 3, -1))
            for step_index in range(execution_horizon):
                if step_index >= best_decoded_window.shape[1]:
                    break
                generated_windows.append(best_decoded_window[:, None, step_index])

    generated_3p = torch.cat(generated_windows[history_length:], dim=1)
    generated_candidates = torch.cat(generated_candidates[history_length:], dim=2)
    return generated_3p, generated_candidates


def generate_3p_from_style_embeddings(
    playstyle_tokens: torch.Tensor,
    playstyle_mask: torch.Tensor,
    song_hash: str,
    difficulty: str,
    style_predictor,
    trajectory_decoder,
    device,
    chunk_length: int,
    history_length: int,
    lookahead: float = 2.0,
    n_cands: int = 32,
    temperature: float = 1.0,
    topk: int = 0,
    argmax_yes: bool = False,
    characteristic: str = DEFAULT_CHARACTERISTIC,
    purview_notes: int = 20,
    floor_time: float = -0.1,
):
    os.makedirs("data/BeatSaver", exist_ok=True)
    collated, beatmap, map_info = open_bsmg(f"data/BeatSaver/{song_hash.upper()}.zip", difficulty)
    note_jump_speed = get_njs(map_info, difficulty, characteristic)

    collated = nanpad_collate_fn([[collated]])
    length = collated["timestamps"].shape[1]
    for name, value in collated.items():
        collated[name] = value.to(device=device)

    map_profiles = MapTensors(collated["notes_np"], collated["bombs_np"], collated["obstacles_np"])
    sampled_song = sample_for_evaluation(
        map_profiles.notes,
        map_profiles.bombs,
        map_profiles.obstacles,
        collated["timestamps"],
        collated["lengths"],
        length,
        GENERATION_SAMPLE_COUNT,
        GENERATION_MINIBATCH_SIZE,
        GENERATION_SEGMENT_STRIDE,
        lookahead,
        purview_notes,
        floor_time,
    )

    stride = trajectory_decoder.stride
    execution_horizon = chunk_length // stride
    generated_3p, candidate_3p = generate_3p_work(
        device,
        execution_horizon,
        sampled_song,
        trajectory_decoder,
        history_length,
        length,
        style_predictor,
        stride,
        map_profiles,
        argmax_yes,
        note_jump_speed,
        playstyle_tokens,
        playstyle_mask,
        n_cands,
        temperature,
        topk,
    )

    generated_3p = interpolate_xyzsixd(generated_3p[:, None], stride)[:, :, :length]
    generated_3p = generated_3p.reshape((*generated_3p.shape[:-1], 3, -1))

    candidate_3p = interpolate_xyzsixd(candidate_3p, stride)[:, :, :length]
    candidate_3p = candidate_3p.reshape((*candidate_3p.shape[:-1], 3, -1))
    return generated_3p, candidate_3p
