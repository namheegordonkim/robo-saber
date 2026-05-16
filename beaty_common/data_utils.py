from dataclasses import dataclass
from functools import reduce
from typing import List

import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class Segment:

    def __post_init__(self):
        for k, v in self.__dict__.items():
            assert isinstance(v, torch.Tensor), f"Expected {k} to be a torch.Tensor"

    def __getitem__(self, item):
        return self.__class__(**{k: v[item] for k, v in self.__dict__.items()})

    def __mul__(self, other):
        return self.__class__(**{k: v * other for k, v in self.__dict__.items()})

    def __set__(self, instance, value):
        for k, v in self.__dict__.items():
            k[:] = value

    @classmethod
    def cat(cls, segments: List["Segment"], dim=0):
        return cls(**{k: torch.cat([getattr(s, k) for s in segments], dim=dim) for k in segments[0].__dict__.keys()})


@dataclass
class GameSegment(Segment):
    notes: torch.Tensor  # note bag sequence, (B, T, 20, 8) where T is the segment length
    bombs: torch.Tensor  # bomb bag sequence, (B, T, 20, 5)
    obstacles: torch.Tensor  # obstacle bag sequence, (B, T, 20, 9)
    note_ids: torch.Tensor  # note ids in the bag, (B, T, 20)
    bomb_ids: torch.Tensor  # bomb ids in the bag, (B, T, 20)
    obstacle_ids: torch.Tensor  # obstacle ids in the bag, (B, T, 20)
    idxs: torch.Tensor  # song idxs (B, T)
    frames: torch.Tensor  # frame idxs (B, T)

    def __len__(self):
        return self.notes.shape[1]

    @property
    def shape(self):
        return self.notes.shape[0], self.notes.shape[1]


@dataclass
class MovementSegment(Segment):
    three_p: torch.Tensor  # 3p sequence, (B, T, 18) or (B, T, 3, 6)
    idxs: torch.Tensor  # song idxs (B, T)
    frames: torch.Tensor  # frame idxs (B, T)

    def __len__(self):
        return self.three_p.shape[1]


class SegmentSampler:

    def _process_object_bag(
        self,
        object_np: torch.Tensor,
        batch_segment_its: torch.Tensor,
        segment_timestamps: torch.Tensor,
        purview_sec: float,
        purview_notes: int,
        floor_time: float,
        use_max_n_notes: bool = False,
        is_obstacle: bool = False,
    ):
        """
        Helper method to process notes, bombs, or obstacles.

        Args:
            object_np: The input tensor (notes_np, bombs_np, or obstacles_np)
            batch_segment_its: Batch segment iteration indices
            segment_timestamps: Timestamps for segments
            purview_sec: Time window in seconds
            purview_notes: Maximum number of objects to include
            floor_time: Minimum time threshold
            use_max_n_notes: If True, use max(object_np.shape[1], purview_notes)
            is_obstacle: If True, use obstacle-specific time delta logic

        Returns:
            Tuple of (segment_objects, segment_object_ids)
        """
        n_notes = max(object_np.shape[1], purview_notes) if use_max_n_notes else object_np.shape[1]

        # Get the bipartite mapping between regular-intervaled timestamps and objects within a map
        all_note_bags = object_np[batch_segment_its[..., 0]]

        # Calculate time deltas and determine which objects are in purview
        if is_obstacle:
            # Obstacles have duration, so check both start and end times
            time_delta_start = all_note_bags[..., 0] - segment_timestamps[..., None]
            time_delta_end = (all_note_bags[..., 0] + all_note_bags[..., 4]) - segment_timestamps[..., None]
            in_purview_yes = torch.logical_and(time_delta_start < purview_sec, time_delta_end > floor_time)
        else:
            # Notes and bombs are point events
            time_deltas = all_note_bags[..., 0] - segment_timestamps[..., None]
            in_purview_yes = torch.logical_and(floor_time < time_deltas, time_deltas < purview_sec)

        # Populate the "purview" information in object bag
        # i.e. closest object to the right of the current timestamp should be the first one in the array
        idxs = torch.arange(n_notes, dtype=torch.long, device=object_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
        idxs += in_purview_yes.max(-1)[-1][..., None]
        idxs %= n_notes
        idxs = idxs[..., :purview_notes]
        right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
        segment_objects = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
        segment_objects[~right_recv_mask] = torch.nan
        segment_objects[..., 0] -= segment_timestamps[..., None]

        segment_object_ids = idxs * 1
        segment_object_ids[~right_recv_mask] = -1  # integers don't like nans

        return segment_objects, segment_object_ids

    def sample_for_training(
        self,
        notes_np: torch.Tensor,
        bombs_np: torch.Tensor,
        obstacles_np: torch.Tensor,
        timestamps: torch.Tensor,
        my_pos_expm: torch.Tensor,
        lengths: torch.Tensor,
        segment_length: int,
        n_samples: int,
        minibatch_size: int,
        stride: int,
        purview_sec: float,
        purview_notes: int,
        floor_time: float,
        firsts_only: bool = False,
    ) -> (GameSegment, MovementSegment):
        """
        Produce variable-dim input and output samples based on specified purview in seconds
        """
        device = notes_np.device
        segment_its = get_segment_its(lengths, n_samples, segment_length, stride, firsts_only).to(device=device)
        # segment_its = get_segment_its_favoring_obstacles(lengths, n_samples, segment_length, obstacles_np[..., 0], timestamps).to(device=device)

        idxs = torch.arange(segment_its.shape[0], device=device)
        batch_idxs = torch.split(idxs, minibatch_size)
        batch_reses = []
        for batch_i in batch_idxs:
            batch_segment_its = segment_its[batch_i]

            segment_3ps = my_pos_expm[batch_segment_its[..., 0], batch_segment_its[..., 1]]
            segment_timestamps = timestamps[batch_segment_its[..., 0], batch_segment_its[..., 1]]

            # Process notes
            segment_notes, segment_note_ids = self._process_object_bag(
                notes_np,
                batch_segment_its,
                segment_timestamps,
                purview_sec,
                purview_notes,
                floor_time,
                use_max_n_notes=False,
                is_obstacle=False,
            )

            # Process bombs
            segment_bombs, segment_bomb_ids = self._process_object_bag(
                bombs_np,
                batch_segment_its,
                segment_timestamps,
                purview_sec,
                purview_notes,
                floor_time,
                use_max_n_notes=False,
                is_obstacle=False,
            )

            # Process obstacles
            segment_obstacles, segment_obstacle_ids = self._process_object_bag(
                obstacles_np,
                batch_segment_its,
                segment_timestamps,
                purview_sec,
                purview_notes,
                floor_time,
                use_max_n_notes=False,
                is_obstacle=True,
            )

            batch_reses.append(
                (
                    segment_3ps,
                    segment_notes,
                    segment_bombs,
                    segment_obstacles,
                    segment_note_ids,
                    segment_bomb_ids,
                    segment_obstacle_ids,
                )
            )

        (
            segment_3ps,
            segment_notes,
            segment_bombs,
            segment_obstacles,
            segment_note_ids,
            segment_bomb_ids,
            segment_obstacle_ids,
        ) = list(
            reduce(
                lambda acc, res: [torch.cat([a, r], dim=0) for a, r in zip(acc, res)],
                batch_reses,
            )
        )

        game_segment = GameSegment(
            notes=segment_notes,
            bombs=segment_bombs,
            obstacles=segment_obstacles,
            note_ids=segment_note_ids,
            bomb_ids=segment_bomb_ids,
            obstacle_ids=segment_obstacle_ids,
            idxs=segment_its[..., 0],
            frames=segment_its[..., 1],
        )
        movement_segment = MovementSegment(three_p=segment_3ps, idxs=segment_its[..., 0], frames=segment_its[..., 1])

        return game_segment, movement_segment

    def sample_for_evaluation(
        self,
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
    ) -> GameSegment:
        """
        Produce variable-dim input and output samples based on specified purview in seconds
        """
        # purview_sec = 2.0
        # purview_sec = 0.25
        # purview_notes = 20

        device = notes_np.device
        segment_its = get_segment_its(lengths, n_samples, segment_length, stride, firsts_only).to(device=device)

        idxs = torch.arange(segment_its.shape[1], device=device)
        batch_idxs = torch.split(idxs, minibatch_size)
        batch_reses = []
        for batch_i in batch_idxs:
            batch_segment_its = segment_its[:, batch_i]

            segment_timestamps = timestamps[batch_segment_its[..., 0], batch_segment_its[..., 1]]

            # Process notes
            segment_notes, segment_note_ids = self._process_object_bag(
                notes_np,
                batch_segment_its,
                segment_timestamps,
                purview_sec,
                purview_notes,
                floor_time,
                use_max_n_notes=True,
                is_obstacle=False,
            )

            # Process bombs
            segment_bombs, segment_bomb_ids = self._process_object_bag(
                bombs_np,
                batch_segment_its,
                segment_timestamps,
                purview_sec,
                purview_notes,
                floor_time,
                use_max_n_notes=True,
                is_obstacle=False,
            )

            # Process obstacles
            segment_obstacles, segment_obstacle_ids = self._process_object_bag(
                obstacles_np,
                batch_segment_its,
                segment_timestamps,
                purview_sec,
                purview_notes,
                floor_time,
                use_max_n_notes=True,
                is_obstacle=True,
            )

            batch_reses.append(
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
        ) = list(
            reduce(
                lambda acc, res: [torch.cat([a, r], dim=1) for a, r in zip(acc, res)],
                batch_reses,
            )
        )

        game_segment = GameSegment(
            notes=segment_notes,
            bombs=segment_bombs,
            obstacles=segment_obstacles,
            note_ids=segment_note_ids,
            bomb_ids=segment_bomb_ids,
            obstacle_ids=segment_obstacle_ids,
            idxs=segment_its[..., 0],
            frames=segment_its[..., 1],
        )

        return game_segment


def get_segment_its(lengths, n_samples, segment_length, stride, firsts_only):
    device = lengths.device
    idxs = (torch.rand(n_samples) * lengths.shape[0]).to(dtype=torch.long, device=device)
    t_start = (torch.rand(n_samples, device=device) * (lengths[idxs] - segment_length)).to(dtype=torch.long, device=device)
    if firsts_only:
        # t_start = torch.zeros_like(t_start)  # first frames
        t_start = torch.ones_like(t_start) * (lengths[idxs] - segment_length - 1)  # last frames
    segment_ts = torch.arange(0, segment_length, stride, device=device) + t_start[:, None]
    segment_its = torch.cat(
        [
            idxs[:, None, None].repeat_interleave(segment_ts.shape[-1], 1),
            segment_ts[:, :, None],
        ],
        dim=2,
    )
    return segment_its


def get_segment_its_favoring_obstacles(lengths, n_samples, segment_length, obstacle_times, timestamps):
    # Get away with just for loop for now
    batch_res = []
    for i in range(lengths.shape[0]):
        b, f = timestamps.shape
        device = lengths.device
        frames = torch.arange(f, device=device)

        good_timestamps_yes = (timestamps[[i]] >= obstacle_times[[i]].nan_to_num(torch.inf).min(1)[0][:, None] - 1) & (timestamps[[i]] <= obstacle_times[[i]].nan_to_num(-torch.inf).max(1)[0][:, None] + 1)
        good_timestamps = torch.ones_like(good_timestamps_yes, dtype=torch.float) * -1
        good_timestamps[good_timestamps_yes] = timestamps[[i]][good_timestamps_yes]
        good_frames = frames[None].repeat_interleave(good_timestamps.shape[0], 0)
        obstacle_frame_idxs = ((obstacle_times[[i]][:, :, None] - good_timestamps[:, None]) <= 1).to(dtype=torch.int).argmax(-1)
        obstacle_frames = torch.take_along_dim(good_frames, obstacle_frame_idxs, -1)

        idxs = (torch.rand(n_samples) * lengths[[i]].shape[0]).to(dtype=torch.long, device=device)
        # idxs = torch.ones(n_samples, device=device, dtype=torch.long) * i
        n_notnans = torch.sum(~torch.isnan(obstacle_times[[i]]), dim=-1)
        mixin_idxs = (torch.rand(n_samples, device=device) * n_notnans[idxs]).to(dtype=torch.long, device=device)
        mixin_times = obstacle_times[[i]][idxs, mixin_idxs]
        mixin_yes = ~torch.isnan(mixin_times)

        mixin_frames = obstacle_frames[idxs, mixin_idxs]
        mixin_frames += torch.randint_like(mixin_frames, -30, 31)
        mixin_frames = torch.clip(
            mixin_frames,
            torch.tensor(0, device=device),
            lengths[[i]][idxs] - segment_length - 1,
        )

        mixin_yes[: mixin_yes.shape[0] // 2] = False
        t_start = (torch.rand(n_samples, device=device) * (lengths[[i]][idxs] - segment_length)).to(dtype=torch.long, device=device)
        t_start[mixin_yes] = mixin_frames[mixin_yes]
        segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
        segment_its = torch.cat(
            [
                idxs[:, None, None].repeat_interleave(segment_length, 1),
                segment_ts[:, :, None],
            ],
            dim=2,
        )
        segment_its[..., 0] = i
        batch_res.append(segment_its)
    batch_res = torch.cat(batch_res, dim=0)
    random_idxs = torch.randperm(batch_res.shape[0])[:n_samples]
    return batch_res[random_idxs]


def nanpad(tensor: torch.Tensor, pad_until: int, dim: int = -1):
    length_to_go = np.maximum(pad_until - tensor.shape[dim], 0)
    good_shape = torch.Size(torch.maximum(torch.as_tensor(tensor.shape), torch.as_tensor(1)))
    nanpad = torch.ones(good_shape, device=tensor.device) * torch.nan
    nanpad = nanpad[[dim]].repeat_interleave(length_to_go, dim=dim)
    padded_tensor = torch.cat([tensor, nanpad], dim=dim)
    return padded_tensor
