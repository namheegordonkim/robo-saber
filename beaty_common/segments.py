from dataclasses import dataclass
from typing import List

import numpy as np
import torch


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
    note_ids: torch.Tensor  # note ids in the bag, (B, T, 20)
    bombs: torch.Tensor  # bomb bag sequence, (B, T, 20, 5)
    obstacles: torch.Tensor  # obstacle bag sequence, (B, T, 20, 9)
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


class VanillaAugmenter:
    def augment(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        timestamps: np.ndarray,
        size: int,
    ) -> (np.ndarray, np.ndarray):
        """
        Produce variable-dim input and output samples based on specified purview in seconds
        """
        # start_idxs = np.random.choice(len(timestamps) - 1, replace=True, size=size)
        start_idxs = np.arange(len(timestamps))
        # x_aug = torch.as_tensor(start_idxs, dtype=torch.long)
        x_aug = torch.as_tensor(timestamps[start_idxs][..., None], dtype=torch.float)
        y_aug = torch.as_tensor(source_y[start_idxs]).reshape(-1, 27)

        return x_aug, y_aug


class PurviewXYAugmenter:
    """
    Given Numpy arrays, prepares HuggingFace training / validation data
    """

    def augment(
        self,
        ds,
    ) -> (np.ndarray, np.ndarray):
        """
        Produce variable-dim input and output samples based on specified purview in seconds
        """
        source_xs = [d["song_np"] for d in ds]
        source_ys = [d["my_pos_expm"] for d in ds]
        timestampses = [d["timestamps"] for d in ds]
        rets = []
        for i in range(len(source_xs)):
            source_x = source_xs[i]
            source_y = source_ys[i]
            timestamps = timestampses[i]

            # interp_thetas = np.linspace(0, 1, 10, endpoint=False)[None].repeat(
            #     timestamps.shape[0] - 1, 0
            # )
            # lefts = timestamps[:-1]
            # rights = timestamps[1:]
            # interpolated_timestamps = (
            #     lefts[..., None] + (rights - lefts)[..., None] * interp_thetas
            # )
            # interpolated_timestamps = interpolated_timestamps.reshape(-1)
            #
            # lefts = source_y[:-1]
            # rights = source_y[1:]
            # interpolated_y = (
            #     lefts[:, None] + (rights - lefts)[:, None] * interp_thetas[..., None]
            # )
            # interpolated_y = interpolated_y.reshape(-1, interpolated_y.shape[-1])

            # max_sec = timestampses[i].max()
            # timestamps = np.arange(0, max_sec, 1 / 60)
            n_notes = source_x.shape[0]

            purview_sec = 2
            purview_notes = 20
            size = len(timestamps) - 1
            # size = len(interpolated_timestamps) - 1
            start_idxs = np.arange(size)
            start_times = timestamps[start_idxs]
            # start_times = interpolated_timestamps[start_idxs]
            end_times = start_times + purview_sec

            current_y_idxs = np.arange(size)
            current_y_times = timestamps[current_y_idxs]
            # current_y_times = interpolated_timestamps[current_y_idxs]
            left_times = current_y_times - purview_sec
            right_times = current_y_times + purview_sec

            # a = np.logical_and(
            #     source_x[..., 0][None] >= left_times[:, None],
            #     source_x[..., 0][None] <= right_times[:, None],
            # )
            # idxs = np.arange(n_notes)[None].repeat(size, 0)
            # idxs -= (n_notes // 2)
            # idxs -= a.sum(-1)[..., None] // 2
            # # idxs %= n_notes
            # # idxs -= a.sum(-1)[..., None] // 2
            # idxs %= n_notes
            # # recv_mask = np.take_along_axis(a, idxs, 0)
            # # roll_by_me = (n_notes // 2) - recv_mask.sum(-1)
            # # q = np.roll(recv_mask, roll_by_me, axis=1)
            #
            # xs = np.take_along_axis(source_x[None].repeat(size, 0), idxs[..., None], 1)
            # xs[..., 0] -= current_y_times[..., None]
            # xs = np.where(
            #     (np.abs(xs[..., 0]) > purview_sec)[..., None],
            #     np.nan,
            #     xs,
            # )
            #
            # print(np.where(~np.isnan(xs[1000, :, 0])))
            # idxs = idxs[..., -purview_notes:]
            # current_x = np.where(
            #     (np.sum(a, axis=-1) > 0)[:, None],
            #     source_x[np.argmax(a, axis=-1)],
            #     np.nan,
            # )
            #
            # Prepare left purview
            b = np.logical_and(
                source_x[..., 0][None] > left_times[:, None],
                source_x[..., 0][None] < current_y_times[:, None],
            )
            idxs = np.arange(n_notes)[None].repeat(size, 0)
            idxs += b.argmax(-1)[..., None] + b.sum(-1)[..., None]
            idxs %= n_notes
            idxs = idxs[..., -purview_notes:]
            left_recv_mask = np.take_along_axis(b, idxs, 1)
            left_x = np.where(
                left_recv_mask[..., None],
                np.take_along_axis(source_x[None].repeat(size, 0), idxs[..., None], 1),
                np.nan,
            )

            # Prepare right purview
            c = np.logical_and(
                source_x[..., 0][None] > current_y_times[:, None],  # double registration happening here!
                source_x[..., 0][None] < right_times[:, None],
            )
            idxs = np.arange(n_notes)[None].repeat(size, 0)
            idxs += c.argmax(-1)[..., None]
            idxs %= n_notes
            idxs = idxs[..., :purview_notes]
            right_recv_mask = np.take_along_axis(c, idxs, 1)
            right_x = np.where(
                right_recv_mask[..., None],
                np.take_along_axis(source_x[None].repeat(size, 0), idxs[..., None], 1),
                np.nan,
            )
            # x_aug = np.concatenate([left_x, right_x], axis=1)
            x_aug = np.concatenate([right_x], axis=1)
            x_aug[..., 0] -= current_y_times[..., None]
            x_aug = np.where(
                (np.abs(x_aug[..., 0]) > purview_sec)[..., None],
                np.nan,
                x_aug,
            )
            # x_aug[:, 0] = np.where(np.isnan(x_aug[:, 0]), 0, x_aug[:, 0])
            # x_aug[:, purview_notes] = np.where(
            #     np.isnan(x_aug[:, purview_notes]), 0, x_aug[:, purview_notes]
            # )
            # x_aug[np.isnan(x_aug)] = 0
            y_aug = source_y[current_y_idxs]
            # y_aug = interpolated_y[current_y_idxs]
            # torch.save(x_aug, f"{proj_dir}/data/beaterson/test/x_aug.pkl")
            # torch.save(y_aug, f"{proj_dir}/data/beaterson/test/y_aug.pkl")
            rets.append((x_aug, y_aug, current_y_times))

        return rets


class PurviewXAugmenter:

    def augment(self, song_np: np.ndarray, timestamps: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Produce variable-dim input and output samples based on specified purview in seconds
        """
        source_x = song_np
        n_notes = source_x.shape[0]

        purview_sec = 2
        purview_notes = 20
        size = len(timestamps) - 1

        current_y_idxs = np.arange(size)
        current_y_times = timestamps[current_y_idxs]
        right_times = current_y_times + purview_sec

        # Prepare right purview
        c = np.logical_and(
            source_x[..., 0][None] > current_y_times[:, None],  # double registration happening here!
            source_x[..., 0][None] < right_times[:, None],
        )
        idxs = np.arange(n_notes)[None].repeat(size, 0)
        idxs += c.argmax(-1)[..., None]
        idxs %= n_notes
        idxs = idxs[..., :purview_notes]
        right_recv_mask = np.take_along_axis(c, idxs, 1)
        right_x = np.where(
            right_recv_mask[..., None],
            np.take_along_axis(source_x[None].repeat(size, 0), idxs[..., None], 1),
            np.nan,
        )

        x_aug = np.concatenate([right_x], axis=1)
        x_aug[..., 0] -= current_y_times[..., None]
        x_aug = np.where(
            (np.abs(x_aug[..., 0]) > purview_sec)[..., None],
            np.nan,
            x_aug,
        )

        return x_aug


# class SegmentSampler:
#
#     def sample_for_training(
#         self,
#         notes_np: torch.Tensor,
#         bombs_np: torch.Tensor,
#         obstacles_np: torch.Tensor,
#         timestamps: torch.Tensor,
#         my_pos_expm: torch.Tensor,
#         lengths: torch.Tensor,
#         segment_length: int,
#         n_samples: int,
#         minibatch_size: int,
#         stride: int,
#     ) -> (GameSegment, MovementSegment):
#         """
#         Produce variable-dim input and output samples based on specified purview in seconds
#         """
#         purview_sec = 2
#         purview_notes = 20
#
#         device = notes_np.device
#         segment_its = get_segment_its(lengths, n_samples, segment_length, stride).to(device=device)
#         # segment_its = get_segment_its_favoring_obstacles(lengths, n_samples, segment_length, obstacles_np[..., 0], timestamps).to(device=device)
#
#         idxs = torch.arange(segment_its.shape[0], device="cuda")
#         batch_idxs = torch.split(idxs, minibatch_size)
#         batch_reses = []
#         for batch_i in batch_idxs:
#             batch_segment_its = segment_its[batch_i]
#
#             segment_3ps = my_pos_expm[batch_segment_its[..., 0], batch_segment_its[..., 1]]
#             segment_timestamps = timestamps[batch_segment_its[..., 0], batch_segment_its[..., 1]]
#
#             n_notes = notes_np.shape[1]
#             # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
#             all_note_bags = notes_np[batch_segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
#             # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
#             time_deltas = all_note_bags[..., 0] - segment_timestamps[..., None]
#             in_purview_yes = torch.logical_and(0 < time_deltas, time_deltas < purview_sec)  # get mask, up to `purview_notes` Trues and rest False
#             # Populate the "purview" information in note bag
#             # i.e. closest note to the right of the current timestamp should be the first one in the array
#             idxs = torch.arange(n_notes, dtype=torch.long, device=notes_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
#             idxs += in_purview_yes.max(-1)[-1][..., None]
#             idxs %= n_notes
#             idxs = idxs[..., :purview_notes]
#             right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
#             segment_notes = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
#             segment_notes[~right_recv_mask] = torch.nan
#             segment_notes[..., 0] -= segment_timestamps[..., None]
#
#             segment_note_ids = idxs * 1
#             segment_note_ids[~right_recv_mask] = -1  # integers don't like nans
#
#             n_notes = bombs_np.shape[1]
#             # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
#             all_note_bags = bombs_np[batch_segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
#             # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
#             time_deltas = all_note_bags[..., 0] - segment_timestamps[..., None]
#             in_purview_yes = torch.logical_and(0 < time_deltas, time_deltas < purview_sec)  # get mask, up to `purview_notes` Trues and rest False
#             # Populate the "purview" information in note bag
#             # i.e. closest note to the right of the current timestamp should be the first one in the array
#             idxs = torch.arange(n_notes, dtype=torch.long, device=bombs_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
#             idxs += in_purview_yes.max(-1)[-1][..., None]
#             idxs %= n_notes
#             idxs = idxs[..., :purview_notes]
#             right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
#             segment_bombs = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
#             segment_bombs[~right_recv_mask] = torch.nan
#             segment_bombs[..., 0] -= segment_timestamps[..., None]
#
#             n_notes = obstacles_np.shape[1]
#             # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
#             all_note_bags = obstacles_np[batch_segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
#             # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
#             time_delta_start = all_note_bags[..., 0] - segment_timestamps[..., None]
#             time_delta_end = (all_note_bags[..., 0] + all_note_bags[..., 4]) - segment_timestamps[..., None]
#             # time_deltas = (all_note_bags[..., 0] + all_note_bags[..., 4]) - segment_timestamps[..., None]
#             # in_purview_yes = (0 < time_deltas) & (
#             #     time_deltas < purview_sec
#             # )  # get mask, up to `purview_notes` Trues and rest False
#             in_purview_yes = torch.logical_and(time_delta_start < purview_sec, time_delta_end > 0)
#
#             # Populate the "purview" information in note bag
#             # i.e. closest note to the right of the current timestamp should be the first one in the array
#             idxs = torch.arange(n_notes, dtype=torch.long, device=obstacles_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
#             idxs += in_purview_yes.max(-1)[-1][..., None]
#             idxs %= n_notes
#             idxs = idxs[..., :purview_notes]
#             right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
#             segment_obstacles = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
#             segment_obstacles[~right_recv_mask] = torch.nan
#             segment_obstacles[..., 0] -= segment_timestamps[..., None]
#
#             batch_reses.append((segment_3ps, segment_notes, segment_note_ids, segment_bombs, segment_obstacles))
#
#         (segment_3ps, segment_notes, segment_note_ids, segment_bombs, segment_obstacles) = list(reduce(lambda acc, res: [torch.cat([a, r], dim=0) for a, r in zip(acc, res)], batch_reses))
#
#         game_segment = GameSegment(
#             notes=segment_notes,
#             note_ids=segment_note_ids,
#             bombs=segment_bombs,
#             obstacles=segment_obstacles,
#             idxs=segment_its[..., 0],
#             frames=segment_its[..., 1],
#         )
#         movement_segment = MovementSegment(three_p=segment_3ps, idxs=segment_its[..., 0], frames=segment_its[..., 1])
#
#         return game_segment, movement_segment
#
#     def sample_for_inference(
#         self,
#         notes_np: torch.Tensor,
#         bombs_np: torch.Tensor,
#         obstacles_np: torch.Tensor,
#         timestamps: torch.Tensor,
#         lengths: torch.Tensor,
#         segment_length: int,
#         n_samples: int,
#         minibatch_size: int,
#         stride: int,
#     ) -> GameSegment:
#         """
#         Produce variable-dim input and output samples based on specified purview in seconds
#         """
#         purview_sec = 2.0
#         # purview_sec = 0.25
#         purview_notes = 20
#
#         device = notes_np.device
#         segment_its = get_segment_its(lengths, n_samples, segment_length, stride).to(device=device)
#
#         idxs = torch.arange(segment_its.shape[0], device="cuda")
#         batch_idxs = torch.split(idxs, minibatch_size)
#         batch_reses = []
#         for batch_i in batch_idxs:
#             batch_segment_its = segment_its[batch_i]
#
#             segment_timestamps = timestamps[batch_segment_its[..., 0], batch_segment_its[..., 1]]
#
#             n_notes = notes_np.shape[1]
#             # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
#             all_note_bags = notes_np[batch_segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
#             # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
#             time_deltas = all_note_bags[..., 0] - segment_timestamps[..., None]
#             in_purview_yes = torch.logical_and(0 < time_deltas, time_deltas < purview_sec)  # get mask, up to `purview_notes` Trues and rest False
#             # Populate the "purview" information in note bag
#             # i.e. closest note to the right of the current timestamp should be the first one in the array
#             idxs = torch.arange(n_notes, dtype=torch.long, device=notes_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
#             idxs += in_purview_yes.max(-1)[-1][..., None]
#             idxs %= n_notes
#             idxs = idxs[..., :purview_notes]
#             right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
#             segment_notes = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
#             segment_notes[~right_recv_mask] = torch.nan
#             segment_notes[..., 0] -= segment_timestamps[..., None]
#
#             segment_note_ids = idxs * 1
#             segment_note_ids[~right_recv_mask] = -1  # integers don't like nans
#
#             n_notes = bombs_np.shape[1]
#             # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
#             all_note_bags = bombs_np[batch_segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
#             # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
#             time_deltas = all_note_bags[..., 0] - segment_timestamps[..., None]
#             in_purview_yes = torch.logical_and(0 < time_deltas, time_deltas < purview_sec)  # get mask, up to `purview_notes` Trues and rest False
#             # Populate the "purview" information in note bag
#             # i.e. closest note to the right of the current timestamp should be the first one in the array
#             idxs = torch.arange(n_notes, dtype=torch.long, device=bombs_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
#             idxs += in_purview_yes.max(-1)[-1][..., None]
#             idxs %= n_notes
#             idxs = idxs[..., :purview_notes]
#             right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
#             segment_bombs = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
#             segment_bombs[~right_recv_mask] = torch.nan
#             segment_bombs[..., 0] -= segment_timestamps[..., None]
#
#             n_notes = obstacles_np.shape[1]
#             # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
#             all_note_bags = obstacles_np[batch_segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
#             # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
#             time_delta_start = all_note_bags[..., 0] - segment_timestamps[..., None]
#             time_delta_end = (all_note_bags[..., 0] + all_note_bags[..., 4]) - segment_timestamps[..., None]
#             # time_deltas = (all_note_bags[..., 0] + all_note_bags[..., 4]) - segment_timestamps[..., None]
#             # in_purview_yes = (0 < time_deltas) & (
#             #     time_deltas < purview_sec
#             # )  # get mask, up to `purview_notes` Trues and rest False
#             in_purview_yes = torch.logical_and(time_delta_start < purview_sec, time_delta_end > 0)
#
#             # Populate the "purview" information in note bag
#             # i.e. closest note to the right of the current timestamp should be the first one in the array
#             idxs = torch.arange(n_notes, dtype=torch.long, device=obstacles_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
#             idxs += in_purview_yes.max(-1)[-1][..., None]
#             idxs %= n_notes
#             idxs = idxs[..., :purview_notes]
#             right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
#             segment_obstacles = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
#             segment_obstacles[~right_recv_mask] = torch.nan
#             segment_obstacles[..., 0] -= segment_timestamps[..., None]
#
#             batch_reses.append((segment_notes, segment_note_ids, segment_bombs, segment_obstacles))
#
#         (segment_notes, segment_note_ids, segment_bombs, segment_obstacles) = list(reduce(lambda acc, res: [torch.cat([a, r], dim=0) for a, r in zip(acc, res)], batch_reses))
#
#         game_segment = GameSegment(
#             notes=segment_notes,
#             note_ids=segment_note_ids,
#             bombs=segment_bombs,
#             obstacles=segment_obstacles,
#             idxs=segment_its[..., 0],
#             frames=segment_its[..., 1],
#         )
#
#         return game_segment
#

def get_segment_its(lengths, n_samples, segment_length, stride):
    device = lengths.device
    idxs = (torch.rand(n_samples) * lengths.shape[0]).to(dtype=torch.long, device=device)
    t_start = (torch.rand(n_samples, device=device) * (lengths[idxs] - segment_length)).to(dtype=torch.long, device=device)
    segment_ts = torch.arange(0, segment_length, stride, device=device) + t_start[:, None]
    segment_its = torch.cat(
        [
            idxs[:, None, None].repeat_interleave(segment_ts.shape[-1], 1),
            segment_ts[:, :, None],
        ],
        dim=2,
    )
    return segment_its

