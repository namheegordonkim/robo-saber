import os
from dataclasses import dataclass
from typing import Any

import torch

from beaty_common.bsmg_xror_utils import open_bsmg
from beaty_common.data_utils import sample_for_evaluation
from beaty_common.eval_utils import get_njs
from beaty_common.pose_utils import interpolate_xyzsixd
from beaty_common.train_utils import nanpad_collate_fn, placeholder_3p_sixd
from beaty_common.torch_nets import GameTensors, MapTensors, ReplayTensors
from vendor.torch_saber import PlayerMotion, SaberSimulation, TorchSaber

GENERATION_SAMPLE_COUNT = 1
GENERATION_MINIBATCH_SIZE = 512
GENERATION_SEGMENT_STRIDE = 1
DEFAULT_CHARACTERISTIC = "Standard"


@dataclass(slots=True)
class PlaystyleTensors:
    tokens: torch.Tensor
    mask: torch.Tensor


@dataclass(slots=True)
class GenerationSettings:
    device: torch.device
    execution_horizon: int
    history_length: int
    length: int
    stride: int
    argmax_yes: bool
    note_jump_speed: float
    n_cands: int
    temperature: float
    topk: int


def generate_3p_work(
    settings: GenerationSettings,
    sampled_song: ReplayTensors,
    map_profiles: MapTensors,
    playstyle: PlaystyleTensors,
    style_predictor: Any,
    trajectory_decoder: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    generated_windows = [
        torch.as_tensor(placeholder_3p_sixd[None, None], device=settings.device)
        for frame_index in range(settings.history_length)
    ]
    note_alive_mask = torch.ones(
        (sampled_song.notes.shape[0], settings.n_cands, map_profiles.notes.shape[1]),
        dtype=torch.bool,
        device=settings.device,
    )
    generated_candidates = [
        torch.as_tensor(placeholder_3p_sixd[None, None, None], device=settings.device)
        .unflatten(-1, (3, -1))
        .repeat_interleave(settings.n_cands, 1)
        for frame_index in range(settings.history_length)
    ]

    with torch.no_grad():
        for frame_index in range(settings.length):
            history_window = torch.cat(generated_windows[-settings.history_length:], dim=1)
            current_notes = sampled_song.notes[:, frame_index].to(device=settings.device)
            current_bombs = sampled_song.bombs[:, frame_index].to(device=settings.device)
            current_obstacles = sampled_song.obstacles[:, frame_index].to(device=settings.device)

            if frame_index % (settings.execution_horizon * settings.stride) != 0:
                continue

            game_object_embeddings, game_object_mask, game_history_embeddings, game_history_mask = style_predictor.encode_game(
                GameTensors(current_notes, current_bombs, current_obstacles, history_window)
            )
            logits = style_predictor.predict_logits_from_embeds(
                game_object_embeddings,
                game_object_mask,
                game_history_embeddings,
                game_history_mask,
                playstyle.tokens * 1,
                playstyle.mask,
            )
            _, decoded_tokens, _, hard_samples, _ = style_predictor.sample_from_z(
                logits,
                n=settings.n_cands if not settings.argmax_yes else 1,
                temperature=settings.temperature,
                topk=settings.topk,
            )
            if settings.argmax_yes:
                decoded_windows = trajectory_decoder.decode(decoded_tokens)
            else:
                decoded_windows = trajectory_decoder.decode(hard_samples)

            candidate_keypoints = torch.cat(
                [history_window[:, [-1], None].repeat_interleave(decoded_windows.shape[1], 1), decoded_windows],
                dim=2,
            )
            candidate_trajectories = interpolate_xyzsixd(candidate_keypoints, settings.stride)
            candidate_trajectories = candidate_trajectories.reshape((*candidate_trajectories.shape[:-1], 3, -1))

            stop_frame = min(frame_index + candidate_trajectories.shape[-3], settings.length)
            carry_state = history_window[:, None, [-1]].unflatten(-1, (3, -1))
            carry_state = carry_state.repeat(1, candidate_trajectories.shape[1], 1, 1, 1)
            candidate_replay = ReplayTensors(
                sampled_song.notes[:, frame_index:stop_frame][:, None].repeat_interleave(settings.n_cands, 1),
                sampled_song.bombs[:, frame_index:stop_frame][:, None].repeat_interleave(settings.n_cands, 1),
                sampled_song.obstacles[:, frame_index:stop_frame][:, None].repeat_interleave(settings.n_cands, 1),
                note_ids=sampled_song.note_ids[:, frame_index:stop_frame][:, None].repeat_interleave(settings.n_cands, 1),
                bomb_ids=sampled_song.bomb_ids[:, frame_index:stop_frame][:, None].repeat_interleave(settings.n_cands, 1),
                obstacle_ids=sampled_song.obstacle_ids[:, frame_index:stop_frame][:, None].repeat_interleave(settings.n_cands, 1),
            )
            feedback = TorchSaber.evaluate_and_simulate(
                SaberSimulation(
                    PlayerMotion(carry_state, candidate_trajectories[:, :, : stop_frame - frame_index]),
                    candidate_replay,
                    map_profiles,
                    settings.note_jump_speed,
                ),
                note_alive_mask,
            )

            candidate_score = (
                1.0
                + feedback.normalized_score[0]
                - 10.0 * feedback.bomb_penalty[0]
                + 10.0 * feedback.obstacle_penalty[0]
            )
            best_candidate_index = torch.argmax(candidate_score, dim=-1)

            selected_hit_mask = feedback.hit_note_mask[
                torch.arange(feedback.hit_note_mask.shape[0]),
                best_candidate_index,
            ][:, None]
            note_alive_mask = note_alive_mask & ~selected_hit_mask
            best_decoded_window = decoded_windows[torch.arange(decoded_windows.shape[0]), best_candidate_index]

            generated_candidates.append(decoded_windows.reshape(*decoded_windows.shape[:-1], 3, -1))
            for step_index in range(settings.execution_horizon):
                if step_index >= best_decoded_window.shape[1]:
                    break
                generated_windows.append(best_decoded_window[:, None, step_index])

    generated_3p = torch.cat(generated_windows[settings.history_length:], dim=1)
    generated_candidates = torch.cat(generated_candidates[settings.history_length:], dim=2)
    return generated_3p, generated_candidates


def generate_3p_from_style_embeddings(
    playstyle: PlaystyleTensors,
    song_hash: str,
    difficulty: str,
    style_predictor: Any,
    trajectory_decoder: Any,
    device: torch.device,
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    settings = GenerationSettings(
        device=device,
        execution_horizon=chunk_length // stride,
        history_length=history_length,
        length=length,
        stride=stride,
        argmax_yes=argmax_yes,
        note_jump_speed=note_jump_speed,
        n_cands=n_cands,
        temperature=temperature,
        topk=topk,
    )
    generated_3p, candidate_3p = generate_3p_work(
        settings,
        sampled_song,
        map_profiles,
        playstyle,
        style_predictor,
        trajectory_decoder,
    )

    generated_3p = interpolate_xyzsixd(generated_3p[:, None], stride)[:, :, :length]
    generated_3p = generated_3p.reshape((*generated_3p.shape[:-1], 3, -1))

    candidate_3p = interpolate_xyzsixd(candidate_3p, stride)[:, :, :length]
    candidate_3p = candidate_3p.reshape((*candidate_3p.shape[:-1], 3, -1))
    return generated_3p, candidate_3p
