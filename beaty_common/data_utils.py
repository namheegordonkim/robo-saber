import torch

from beaty_common.torch_nets import ReplayTensors

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def process_object_bag(
    object_bags: torch.Tensor,
    segment_indices: torch.Tensor,
    segment_timestamps: torch.Tensor,
    purview_seconds: float,
    purview_count: int,
    floor_time: float,
    use_max_object_count: bool = False,
    is_obstacle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_object_count = max(object_bags.shape[1], purview_count) if use_max_object_count else object_bags.shape[1]
    objects_for_segments = object_bags[segment_indices[..., 0]]

    if is_obstacle:
        start_offsets = objects_for_segments[..., 0] - segment_timestamps[..., None]
        end_offsets = objects_for_segments[..., 0] + objects_for_segments[..., 4] - segment_timestamps[..., None]
        object_in_purview = torch.logical_and(start_offsets < purview_seconds, end_offsets > floor_time)
    else:
        time_offsets = objects_for_segments[..., 0] - segment_timestamps[..., None]
        object_in_purview = torch.logical_and(floor_time < time_offsets, time_offsets < purview_seconds)

    object_indices = torch.arange(max_object_count, dtype=torch.long, device=object_bags.device)[None, None]
    object_indices = object_indices.repeat_interleave(object_in_purview.shape[0], 0)
    object_indices = object_indices.repeat_interleave(object_in_purview.shape[1], 1)
    object_indices += object_in_purview.max(-1)[-1][..., None]
    object_indices %= max_object_count
    object_indices = object_indices[..., :purview_count]

    receive_mask = torch.take_along_dim(object_in_purview, object_indices, 2)
    segment_objects = torch.take_along_dim(objects_for_segments, object_indices[..., None], 2)
    segment_objects[~receive_mask] = torch.nan
    segment_objects[..., 0] -= segment_timestamps[..., None]

    segment_object_ids = object_indices.clone()
    segment_object_ids[~receive_mask] = -1
    return segment_objects, segment_object_ids


def sample_for_training(
    notes_np: torch.Tensor,
    bombs_np: torch.Tensor,
    obstacles_np: torch.Tensor,
    timestamps: torch.Tensor,
    three_p: torch.Tensor,
    lengths: torch.Tensor,
    segment_length: int,
    n_samples: int,
    minibatch_size: int,
    stride: int,
    purview_sec: float,
    purview_notes: int,
    floor_time: float,
    firsts_only: bool = False,
) -> ReplayTensors:
    local_device = notes_np.device
    segment_indices = get_segment_indices(lengths, n_samples, segment_length, stride, firsts_only).to(device=local_device)

    batch_indices = torch.split(torch.arange(segment_indices.shape[0], device=local_device), minibatch_size)
    segment_batches = []
    for batch_index in batch_indices:
        batch_segment_indices = segment_indices[batch_index]
        segment_three_p = three_p[batch_segment_indices[..., 0], batch_segment_indices[..., 1]]
        segment_timestamps = timestamps[batch_segment_indices[..., 0], batch_segment_indices[..., 1]]

        segment_notes, segment_note_ids = process_object_bag(
            notes_np,
            batch_segment_indices,
            segment_timestamps,
            purview_sec,
            purview_notes,
            floor_time,
        )
        segment_bombs, segment_bomb_ids = process_object_bag(
            bombs_np,
            batch_segment_indices,
            segment_timestamps,
            purview_sec,
            purview_notes,
            floor_time,
        )
        segment_obstacles, segment_obstacle_ids = process_object_bag(
            obstacles_np,
            batch_segment_indices,
            segment_timestamps,
            purview_sec,
            purview_notes,
            floor_time,
            is_obstacle=True,
        )

        segment_batches.append(
            (
                segment_three_p,
                segment_notes,
                segment_bombs,
                segment_obstacles,
                segment_note_ids,
                segment_bomb_ids,
                segment_obstacle_ids,
            )
        )

    (
        segment_three_p,
        segment_notes,
        segment_bombs,
        segment_obstacles,
        segment_note_ids,
        segment_bomb_ids,
        segment_obstacle_ids,
    ) = [torch.cat(parts, dim=0) for parts in zip(*segment_batches)]

    return ReplayTensors(
        notes=segment_notes,
        bombs=segment_bombs,
        obstacles=segment_obstacles,
        trajectory=segment_three_p,
        note_ids=segment_note_ids,
        bomb_ids=segment_bomb_ids,
        obstacle_ids=segment_obstacle_ids,
    )


def sample_for_evaluation(
    notes_np: torch.Tensor,
    bombs_np: torch.Tensor,
    obstacles_np: torch.Tensor,
    timestamps: torch.Tensor,
    lengths: torch.Tensor,
    segment_length: int,
    n_samples: int,
    minibatch_size: int,
    stride: int,
    purview_sec: float,
    purview_notes: int,
    floor_time: float,
    firsts_only: bool = False,
) -> ReplayTensors:
    local_device = notes_np.device
    segment_indices = get_segment_indices(lengths, n_samples, segment_length, stride, firsts_only).to(device=local_device)

    batch_indices = torch.split(torch.arange(segment_indices.shape[1], device=local_device), minibatch_size)
    segment_batches = []
    for batch_index in batch_indices:
        batch_segment_indices = segment_indices[:, batch_index]
        segment_timestamps = timestamps[batch_segment_indices[..., 0], batch_segment_indices[..., 1]]

        segment_notes, segment_note_ids = process_object_bag(
            notes_np,
            batch_segment_indices,
            segment_timestamps,
            purview_sec,
            purview_notes,
            floor_time,
            use_max_object_count=True,
        )
        segment_bombs, segment_bomb_ids = process_object_bag(
            bombs_np,
            batch_segment_indices,
            segment_timestamps,
            purview_sec,
            purview_notes,
            floor_time,
            use_max_object_count=True,
        )
        segment_obstacles, segment_obstacle_ids = process_object_bag(
            obstacles_np,
            batch_segment_indices,
            segment_timestamps,
            purview_sec,
            purview_notes,
            floor_time,
            use_max_object_count=True,
            is_obstacle=True,
        )

        segment_batches.append(
            (
                segment_notes,
                segment_bombs,
                segment_obstacles,
                segment_note_ids,
                segment_bomb_ids,
                segment_obstacle_ids,
            )
        )

    (
        segment_notes,
        segment_bombs,
        segment_obstacles,
        segment_note_ids,
        segment_bomb_ids,
        segment_obstacle_ids,
    ) = [torch.cat(parts, dim=1) for parts in zip(*segment_batches)]

    return ReplayTensors(
        notes=segment_notes,
        bombs=segment_bombs,
        obstacles=segment_obstacles,
        note_ids=segment_note_ids,
        bomb_ids=segment_bomb_ids,
        obstacle_ids=segment_obstacle_ids,
    )


def get_segment_indices(
    lengths: torch.Tensor,
    n_samples: int,
    segment_length: int,
    stride: int,
    firsts_only: bool,
) -> torch.Tensor:
    local_device = lengths.device
    song_indices = (torch.rand(n_samples) * lengths.shape[0]).to(dtype=torch.long, device=local_device)
    start_frames = (torch.rand(n_samples, device=local_device) * (lengths[song_indices] - segment_length)).to(
        dtype=torch.long,
        device=local_device,
    )
    if firsts_only:
        start_frames = torch.ones_like(start_frames) * (lengths[song_indices] - segment_length - 1)

    frame_indices = torch.arange(0, segment_length, stride, device=local_device) + start_frames[:, None]
    return torch.cat(
        [
            song_indices[:, None, None].repeat_interleave(frame_indices.shape[-1], 1),
            frame_indices[:, :, None],
        ],
        dim=2,
    )
