import os
import numpy as np
import torch

from beaty_common.pose_utils import interpolate_xyzsixd
from torch_saber import TorchSaber
from beaty_common.train_utils import placeholder_3p_sixd, nanpad_collate_fn
from beaty_common.bsmg_xror_utils import open_bsmg
from beaty_common.eval_utils import get_njs
from beaty_common.data_utils import SegmentSampler
from proj_utils.dirs import proj_dir


def generate_3p_work(
    device,
    execution_horizon,
    game_segments,
    gsvae_net,
    history_len,
    length,
    pred_net,
    stride,
    note_profiles,
    bomb_profiles,
    obstacle_profiles,
    argmax_yes,
    gt_3p,
    njs,
    playstyle_tokens: torch.Tensor,
    playstyle_mask: torch.Tensor,
    n_cands: int,
    temperature: float,
    topk: int,
):
    w_ins = [torch.as_tensor(placeholder_3p_sixd[None, None], device=device) for t in range(history_len)]
    n_notes = note_profiles.shape[1]
    note_alive_mask = torch.ones((game_segments.notes.shape[0], n_cands, n_notes), dtype=torch.bool, device=device)
    generated_cands = [torch.as_tensor(placeholder_3p_sixd[None, None, None], device=device).unflatten(-1, (3, -1)).repeat_interleave(n_cands, 1) for t in range(history_len)]

    with torch.no_grad():
        for t in range(0, length):
            w_in = torch.cat(w_ins[-history_len:], dim=1)  # history
            notes_in = game_segments.notes[:, t].to(device=device)
            bombs_in = game_segments.bombs[:, t].to(device=device)
            obstacles_in = game_segments.obstacles[:, t].to(device=device)

            if t % (execution_horizon * stride) == 0:
                # Query the model
                game_obj_embeds, game_obj_mask, game_hist_embeds, game_hist_mask = pred_net.encode_game(
                    notes_in,
                    bombs_in,
                    obstacles_in,
                    w_in,
                )
                z = pred_net.predict_logits_from_embeds(
                    game_obj_embeds,
                    game_obj_mask,
                    game_hist_embeds,
                    game_hist_mask,
                    playstyle_tokens * 1,
                    playstyle_mask,
                )
                z, k, z_soft, z_hard, _ = pred_net.sample_from_z(
                    z,
                    n=n_cands if not argmax_yes else 1,
                    temperature=temperature,
                    topk=topk,
                )
                if argmax_yes:
                    decoded = gsvae_net.decode(k)
                else:
                    decoded = gsvae_net.decode(z_hard)

                keypoints = torch.cat([w_in[:, [-1], None].repeat_interleave(decoded.shape[1], 1), decoded], dim=2)

                # Interpolate the model output for evaluation
                y_cands = interpolate_xyzsixd(keypoints, stride)
                y_cands_ = y_cands.reshape((*y_cands.shape[:-1], 3, -1))

                # Evaluate candidate trajectories
                hi = np.min([t + y_cands.shape[-2], length])
                carry = w_in[:, None, [-1]].unflatten(-1, (3, -1))
                carry_shape = [
                    1,
                ] * len(carry.shape)
                carry_shape[1] = y_cands_.shape[1]
                carry = carry.repeat(carry_shape)
                res = TorchSaber.evaluate_and_simulate(
                    carry,
                    y_cands_[:, ..., : hi - t, :, :],
                    game_segments.notes[:, t:hi][:, None].repeat_interleave(n_cands, 1),
                    game_segments.bombs[:, t:hi][:, None].repeat_interleave(n_cands, 1),
                    game_segments.obstacles[:, t:hi][:, None].repeat_interleave(n_cands, 1),
                    game_segments.note_ids[:, t:hi][:, None].repeat_interleave(n_cands, 1),
                    game_segments.bomb_ids[:, t:hi][:, None].repeat_interleave(n_cands, 1),
                    game_segments.obstacle_ids[:, t:hi][:, None].repeat_interleave(n_cands, 1),
                    note_profiles,
                    bomb_profiles,
                    obstacle_profiles,
                    note_alive_mask,
                    1.5044,
                    njs,
                )

                deltas = decoded[:, :, 0] - w_in[:, [-1]]
                dists = torch.norm(deltas, dim=-1)
                pos_continuity_score = torch.exp(-(dists**2))
                pos_continuity_score[pos_continuity_score.isnan()] = 0

                history_vel = w_in[:, [-1]] - w_in[:, [-2]]
                # mean_vel = torch.mean(keypoints[..., 1:, :] - keypoints[..., :-1, :], dim=-2)
                vel_diff = torch.norm(history_vel - deltas, dim=-1)
                # vel_diff = torch.norm(history_vel - mean_vel, dim=-1)
                vel_continuity_score = torch.exp(-(vel_diff**2))
                vel_continuity_score[vel_continuity_score.isnan()] = 0

                diffs = y_cands[:, :, 1:] - y_cands[:, :, :-1]
                diffdiffs = diffs[:, :, 1:] - diffs[:, :, :-1]
                diffdiffdiffs = diffdiffs[:, :, 1:] - diffdiffs[:, :, :-1]

                head_xy_mags = torch.norm(y_cands_[..., 0, :1], dim=-1).mean(-1)
                ref_head_xyz = torch.tensor([0, 1.5044, 0], device=device).repeat(*y_cands_[..., 0, :3].shape[:-1], 1)
                head_deviation = torch.norm(y_cands_[..., 0, :3] - ref_head_xyz, dim=-1).mean(-1)
                hand_xyz_mags = torch.norm(y_cands_[..., 1:, :3], dim=-1).mean(-1).mean(-1)
                vels = torch.norm(diffs, dim=-1).mean(-1)
                accs = torch.norm(diffdiffs, dim=-1).mean(-1)
                jerks = torch.norm(diffdiffdiffs, dim=-1).mean(-1)

                head_xy_score = torch.exp(-(head_xy_mags**2))
                hand_xyz_score = torch.exp(-(hand_xyz_mags**2))
                head_deviation_score = torch.exp(-(head_deviation**2))
                vel_score = torch.exp(-(vels**2))
                acc_score = torch.exp(-(accs**2))
                jerk_score = torch.exp(-(jerks**2))

                # best_idx = (
                #     torch.argmax(
                #         1e0 * res[0][0],
                #         dim=-1,
                #         keepdim=False,
                #     )
                #     * 1
                # )
                final_score = 1e0 + res[0][0] - 1e1 * res[-2][0] + 1e1 * res[-1][0]
                best_idx = torch.argmax(final_score, dim=-1, keepdim=False) * 1
                # print(res[0][0])
                # print(final_score)
                # print(best_idx)

                note_alive_mask = note_alive_mask & ~res[5][:, [best_idx]]
                y_out = decoded[torch.arange(decoded.shape[0]), best_idx]
                # if gt_3p is not None and t < 500:
                #     y_out = gt_3p[:, t:hi:stride]

                generated_cands.append(decoded.reshape(*decoded.shape[:-1], 3, -1))
                # note_alive_mask = note_alive_mask & (~res[-3][torch.arange(decoded.shape[0]), best_idx])
                for i in range(execution_horizon):
                    if i >= y_out.shape[1]:
                        break
                    w_ins.append(y_out[:, None, i])
    # generated_3p = torch.cat(w_ins, dim=1)
    generated_3p = torch.cat(w_ins[history_len:], dim=1)
    # generated_cands = torch.cat(generated_cands, dim=2)
    generated_cands = torch.cat(generated_cands[history_len:], dim=2)
    return generated_3p, generated_cands


def generate_3p(
    device,
    execution_horizon,
    game_segments,
    playstyle_notes: torch.Tensor,
    playstyle_bombs: torch.Tensor,
    playstyle_obstacles: torch.Tensor,
    playstyle_history: torch.Tensor,
    playstyle_3p: torch.Tensor,
    gsvae_net,
    history_len,
    length,
    pred_net,
    stride,
    note_profiles,
    bomb_profiles,
    obstacle_profiles,
    argmax_yes,
    gt_3p,
    njs,
    n_cands: int,
    temperature: float,
):
    playstyle_tokens, playstyle_mask = pred_net.encode_style(
        playstyle_notes,
        playstyle_bombs,
        playstyle_obstacles,
        playstyle_history,
        playstyle_3p,
    )
    return generate_3p_work(
        device,
        execution_horizon,
        game_segments,
        gsvae_net,
        history_len,
        length,
        pred_net,
        stride,
        note_profiles,
        bomb_profiles,
        obstacle_profiles,
        argmax_yes,
        gt_3p,
        njs,
        playstyle_tokens,
        playstyle_mask,
        n_cands,
        temperature,
    )


def generate_3p_from_style_embeddings(
    playstyle_tokens: torch.Tensor,
    playstyle_mask: torch.Tensor,
    song_hash: str,
    difficulty: str,
    pred_net,
    gsvae_net,
    device,
    chunk_length: int,
    history_len: int,
    lookahead: float = 2.0,
    n_cands: int = 32,
    temperature: float = 1.0,
    topk: int = 0,
    argmax_yes: bool = False,
    characteristic: str = "Standard",
    purview_notes: int = 20,
    floor_time: float = -0.1,
):
    """
    Generate segment_3p and cand_3p from playstyle embeddings.

    Args:
        playstyle_tokens: Encoded playstyle tokens from pred_net.encode_style()
        playstyle_mask: Playstyle mask from pred_net.encode_style()
        song_hash: Song hash identifier
        difficulty: Difficulty level (e.g., "Expert", "ExpertPlus")
        pred_net: Prediction network model
        gsvae_net: GSVAE network model
        device: Torch device
        lookahead: Lookahead parameter for segment sampling (default 2.0)
        n_cands: Number of candidates for generation (default 32)
        temperature: Temperature for sampling (default 1.0)
        argmax_yes: Whether to use argmax sampling (default False)
        characteristic: Beatmap characteristic (default "Standard")
        purview_notes: Number of notes in purview (default 20)
        floor_time: Floor time for segment sampling (default -0.1)

    Returns:
        segment_3p: Interpolated and processed 3P trajectory
        cand_3p: Interpolated and processed candidate trajectories
    """
    zip_dir = "data/BeatSaver"
    os.makedirs(zip_dir, exist_ok=True)
    zip_path = f"{zip_dir}/{song_hash.upper()}.zip"
    d, beatmap, map_info = open_bsmg(zip_path, difficulty)
    njs = get_njs(map_info, difficulty, characteristic)

    d = nanpad_collate_fn([[d]])
    length = d["timestamps"].shape[1]
    for k, v in d.items():
        d[k] = v.to(device=device)
    note_profiles = d["notes_np"]
    bomb_profiles = d["bombs_np"]
    obstacle_profiles = d["obstacles_np"]
    length = d["timestamps"].shape[1]

    segment_sampler = SegmentSampler()
    game_segments_ = segment_sampler.sample_for_evaluation(
        note_profiles,
        bomb_profiles,
        obstacle_profiles,
        d["timestamps"],
        d["lengths"],
        length,
        1,
        512,
        1,
        lookahead,
        purview_notes,
        floor_time,
    )

    stride = gsvae_net.stride

    execution_horizon = (chunk_length // stride) // 1

    generated_3p, cand_3p = generate_3p_work(
        device,
        execution_horizon,
        game_segments_,
        gsvae_net,
        history_len,
        length,
        pred_net,
        stride,
        note_profiles,
        bomb_profiles,
        obstacle_profiles,
        argmax_yes,
        None,
        njs,
        playstyle_tokens,
        playstyle_mask,
        n_cands,
        temperature,
        topk,
    )

    interpolated_3p = interpolate_xyzsixd(generated_3p[:, None], stride)
    segment_3p = interpolated_3p[:, :, :length]
    segment_3p = segment_3p.reshape(*(segment_3p.shape[:-1]), 3, -1)
    # segment_3p = segment_3p[:, :, history_len:]

    interpolated_cands = interpolate_xyzsixd(cand_3p, stride)
    cand_3p = interpolated_cands[:, :, :length]
    cand_3p = cand_3p.reshape(*(cand_3p.shape[:-1]), 3, -1)
    # cand_3p = cand_3p[:, :, history_len:]

    return segment_3p, cand_3p
