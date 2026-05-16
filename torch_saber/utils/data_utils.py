from dataclasses import dataclass

import torch


@dataclass
class GameSegment:
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
class MovementSegment:
    three_p: torch.Tensor  # 3p sequence, (B, T, 18) or (B, T, 3, 6)
    idxs: torch.Tensor  # song idxs (B, T)
    frames: torch.Tensor  # frame idxs (B, T)

    def __len__(self):
        return self.three_p.shape[1]


class SegmentSampler:

    def sample(
        self,
        notes_np: torch.Tensor,
        bombs_np: torch.Tensor,
        obstacles_np: torch.Tensor,
        timestamps: torch.Tensor,
        my_pos_expm: torch.Tensor,
        lengths: torch.Tensor,
        segment_length: int,
        n_samples: int,
    ) -> (GameSegment, MovementSegment):
        """
        Produce variable-dim input and output samples based on specified purview in seconds
        """
        purview_sec = 2
        purview_notes = 20

        device = notes_np.device
        segment_its = get_segment_its(lengths, n_samples, segment_length).to(device=device)
        segment_3ps = my_pos_expm[segment_its[..., 0], segment_its[..., 1]]
        segment_timestamps = timestamps[segment_its[..., 0], segment_its[..., 1]]

        n_notes = notes_np.shape[1]
        # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
        all_note_bags = notes_np[segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
        # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
        time_deltas = all_note_bags[..., 0] - segment_timestamps[..., None]
        in_purview_yes = torch.logical_and(0 < time_deltas, time_deltas < purview_sec)  # get mask, up to `purview_notes` Trues and rest False
        # Populate the "purview" information in note bag
        # i.e. closest note to the right of the current timestamp should be the first one in the array
        idxs = torch.arange(n_notes, dtype=torch.long, device=notes_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
        idxs += in_purview_yes.max(-1)[-1][..., None]
        idxs %= n_notes
        idxs = idxs[..., :purview_notes]
        right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
        segment_notes = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
        segment_notes[~right_recv_mask] = torch.nan
        segment_notes[..., 0] -= segment_timestamps[..., None]

        segment_note_ids = idxs * 1
        segment_note_ids[~right_recv_mask] = -1  # integers don't like nans

        n_notes = bombs_np.shape[1]
        # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
        all_note_bags = bombs_np[segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
        # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
        time_deltas = all_note_bags[..., 0] - segment_timestamps[..., None]
        in_purview_yes = torch.logical_and(0 < time_deltas, time_deltas < purview_sec)  # get mask, up to `purview_notes` Trues and rest False
        # Populate the "purview" information in note bag
        # i.e. closest note to the right of the current timestamp should be the first one in the array
        idxs = torch.arange(n_notes, dtype=torch.long, device=bombs_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
        idxs += in_purview_yes.max(-1)[-1][..., None]
        idxs %= n_notes
        idxs = idxs[..., :purview_notes]
        right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
        segment_bombs = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
        segment_bombs[~right_recv_mask] = torch.nan
        segment_bombs[..., 0] -= segment_timestamps[..., None]

        n_notes = obstacles_np.shape[1]
        # Get the bipartite mapping between regular-intervaled timestamps and notes within a map
        all_note_bags = obstacles_np[segment_its[..., 0]]  # segment_its[..., 0] is tile-padded idxs for songs within current input batch of songs
        # Should filter the bag of all notes to up to `purview_notes` notes within up to `purview_sec` seconds
        time_delta_start = all_note_bags[..., 0] - segment_timestamps[..., None]
        time_delta_end = (all_note_bags[..., 0] + all_note_bags[..., 4]) - segment_timestamps[..., None]
        # time_deltas = (all_note_bags[..., 0] + all_note_bags[..., 4]) - segment_timestamps[..., None]
        # in_purview_yes = (0 < time_deltas) & (
        #     time_deltas < purview_sec
        # )  # get mask, up to `purview_notes` Trues and rest False
        in_purview_yes = torch.logical_and(time_delta_start < purview_sec, time_delta_end > 0)

        # Populate the "purview" information in note bag
        # i.e. closest note to the right of the current timestamp should be the first one in the array
        idxs = torch.arange(n_notes, dtype=torch.long, device=obstacles_np.device)[None, None].repeat_interleave(in_purview_yes.shape[0], 0).repeat_interleave(in_purview_yes.shape[1], 1)
        idxs += in_purview_yes.max(-1)[-1][..., None]
        idxs %= n_notes
        idxs = idxs[..., :purview_notes]
        right_recv_mask = torch.take_along_dim(in_purview_yes, idxs, 2)
        segment_obstacles = torch.take_along_dim(all_note_bags, idxs[..., None], 2)
        segment_obstacles[~right_recv_mask] = torch.nan
        segment_obstacles[..., 0] -= segment_timestamps[..., None]

        input_segment = GameSegment(
            notes=segment_notes,
            note_ids=segment_note_ids,
            bombs=segment_bombs,
            obstacles=segment_obstacles,
            idxs=segment_its[..., 0],
            frames=segment_its[..., 1],
        )
        output_segment = MovementSegment(three_p=segment_3ps, idxs=segment_its[..., 0], frames=segment_its[..., 1])

        return input_segment, output_segment


def get_segment_its(lengths, n_samples, segment_length):
    device = lengths.device
    idxs = (torch.rand(n_samples) * lengths.shape[0]).to(dtype=torch.long, device=device)
    t_start = (torch.rand(n_samples, device=device) * (lengths[idxs] - segment_length)).to(dtype=torch.long, device=device)
    segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
    segment_its = torch.cat(
        [
            idxs[:, None, None].repeat_interleave(segment_length, 1),
            segment_ts[:, :, None],
        ],
        dim=2,
    )
    return segment_its
