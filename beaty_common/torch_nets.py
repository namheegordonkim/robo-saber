from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class GameTensors:
    notes: torch.Tensor
    bombs: torch.Tensor
    obstacles: torch.Tensor
    history: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("notes", self.notes),
            ("bombs", self.bombs),
            ("obstacles", self.obstacles),
            ("history", self.history),
        ):
            assert isinstance(value, torch.Tensor), f"Expected {name} to be a torch.Tensor"
        assert self.notes.shape[:-1] == self.bombs.shape[:-1] == self.obstacles.shape[:-1]
        assert self.history.shape[:1] == self.notes.shape[:1]


@dataclass(slots=True)
class ReplayTensors:
    notes: torch.Tensor
    bombs: torch.Tensor
    obstacles: torch.Tensor
    history: torch.Tensor | None = None
    trajectory: torch.Tensor | None = None
    note_ids: torch.Tensor | None = None
    bomb_ids: torch.Tensor | None = None
    obstacle_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("notes", self.notes),
            ("bombs", self.bombs),
            ("obstacles", self.obstacles),
            ("history", self.history),
            ("trajectory", self.trajectory),
            ("note_ids", self.note_ids),
            ("bomb_ids", self.bomb_ids),
            ("obstacle_ids", self.obstacle_ids),
        ):
            if value is not None:
                assert isinstance(value, torch.Tensor), f"Expected {name} to be a torch.Tensor"
        assert self.notes.shape[:-1] == self.bombs.shape[:-1] == self.obstacles.shape[:-1]
        if self.history is not None:
            assert self.history.shape[:2] == self.notes.shape[:2]
        if self.trajectory is not None:
            assert self.trajectory.shape[:2] == self.notes.shape[:2]
        if self.note_ids is not None:
            assert self.note_ids.shape == self.notes.shape[:-1]
        if self.bomb_ids is not None:
            assert self.bomb_ids.shape == self.bombs.shape[:-1]
        if self.obstacle_ids is not None:
            assert self.obstacle_ids.shape == self.obstacles.shape[:-1]


@dataclass(slots=True)
class MapTensors:
    notes: torch.Tensor
    bombs: torch.Tensor
    obstacles: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("notes", self.notes),
            ("bombs", self.bombs),
            ("obstacles", self.obstacles),
        ):
            assert isinstance(value, torch.Tensor), f"Expected {name} to be a torch.Tensor"
        assert self.notes.shape[:1] == self.bombs.shape[:1] == self.obstacles.shape[:1]


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-20,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    uniform_noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
    soft_logits = ((logits + gumbel_noise) / tau).softmax(dim)
    max_index = soft_logits.argmax(dim)
    hard_logits = torch.eye(logits.shape[-1], device=logits.device)[max_index]
    hard_logits = hard_logits - soft_logits.detach() + soft_logits
    return soft_logits, hard_logits


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape: int | tuple[int, ...] = (), *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float), requires_grad=False)
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, arr: torch.Tensor) -> None:
        self.update_from_moments(torch.mean(arr, dim=0), torch.var(arr, dim=0), arr.shape[0])

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        mean_square_a = self.var * self.count
        mean_square_b = batch_var * batch_count
        mean_square_total = mean_square_a + mean_square_b + torch.square(delta) * self.count * batch_count / total_count

        self.mean.data = new_mean
        self.var.data = mean_square_total / total_count
        self.count = batch_count + self.count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip((arr - self.mean) / torch.sqrt(self.var + self.epsilon), -5, 5)

    def unnormalize(self, arr: torch.Tensor) -> torch.Tensor:
        return arr * torch.sqrt(self.var + self.epsilon) + self.mean


class InvarMixin:
    def pack_invar(self, value: torch.Tensor) -> torch.Tensor:
        packed_value = value.unflatten(-1, (3, -1)) * 1
        packed_value[..., 1:, :] -= packed_value[..., :1, :]
        return packed_value.reshape(value.shape)

    def unpack_invar(self, value: torch.Tensor) -> torch.Tensor:
        unpacked_value = value.unflatten(-1, (3, -1)) * 1
        unpacked_value[..., 1:, :] += unpacked_value[..., :1, :]
        return unpacked_value.reshape(value.shape)


class CondTransformerGSVAE(nn.Module, InvarMixin):
    def __init__(
        self,
        note_size: int,
        bomb_size: int,
        obstacle_size: int,
        history_size: int,
        hidden_size: int,
        embed_size: int,
        sentence_length: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.note_size = note_size
        self.bomb_size = bomb_size
        self.obstacle_size = obstacle_size
        self.threep_size = history_size
        self.effective_3p_size = self.threep_size // 3

        self.hidden_size = hidden_size
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = 0.0
        self.embed_size = embed_size
        self.activation = nn.GELU()

        self.note_proj = nn.Sequential(
            nn.Linear(self.note_size - 1, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.bomb_proj = nn.Sequential(
            nn.Linear(self.bomb_size - 1, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.obstacle_proj = nn.Sequential(
            nn.Linear(self.obstacle_size - 1, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.threep_proj = nn.Sequential(
            nn.Linear(self.effective_3p_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.playstyle_encoder = nn.TransformerEncoder(encoder_layer, 1)
        self.logit_predictor = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.deproj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.sentence_length * self.vocab_size),
        )
        self.note_rms = RunningMeanStd(shape=(self.note_size,))
        self.bomb_rms = RunningMeanStd(shape=(self.bomb_size,))
        self.obstacle_rms = RunningMeanStd(shape=(self.obstacle_size,))
        self.threep_rms = RunningMeanStd(shape=(self.threep_size,))
        self.tau = 1

    def setup(
        self,
        notes: torch.Tensor,
        bombs: torch.Tensor,
        obstacles: torch.Tensor,
        history: torch.Tensor,
    ) -> None:
        for stream, rms in zip(
            [notes, bombs, obstacles],
            [self.note_rms, self.bomb_rms, self.obstacle_rms],
        ):
            nan_mask = torch.isnan(stream).any(-1)
            if (~nan_mask).sum() > 0:
                rms.update(stream[~nan_mask])
        self.threep_rms.update(self.pack_invar(history[:, None]).reshape(-1, self.threep_size))

    def encode_game(self, game: GameTensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        game.obstacles[..., 0] = torch.clip(game.obstacles[..., 0], min=0)
        object_embeddings, object_mask = self.embed_game_obj(game.notes, game.bombs, game.obstacles)
        history_embeddings, history_mask = self.embed_3p(self.pack_invar(game.history))
        return object_embeddings, object_mask, history_embeddings, history_mask

    def predict_logits(self, game: GameTensors, playstyle: ReplayTensors) -> torch.Tensor:
        assert playstyle.history is not None
        assert playstyle.trajectory is not None
        playstyle.obstacles[..., 0] = torch.clip(playstyle.obstacles[..., 0], min=0)
        object_embeddings, object_mask, history_embeddings, history_mask = self.encode_game(game)
        playstyle_tokens = None
        playstyle_mask = None
        if playstyle.notes.shape[1] > 0:
            playstyle_tokens, playstyle_mask = self.encode_style(playstyle)

        return self.predict_logits_from_embeds(
            object_embeddings,
            object_mask,
            history_embeddings,
            history_mask,
            playstyle_tokens,
            playstyle_mask,
        )

    def encode_style(self, playstyle: ReplayTensors) -> tuple[torch.Tensor, torch.Tensor]:
        assert playstyle.history is not None
        assert playstyle.trajectory is not None
        object_embeddings, object_mask = self.embed_game_obj(playstyle.notes, playstyle.bombs, playstyle.obstacles)
        history_embeddings, history_mask = self.embed_3p(self.pack_invar(playstyle.history))
        trajectory_embeddings, trajectory_mask = self.embed_3p(self.pack_invar(playstyle.trajectory))

        sequence_embeddings = torch.cat(
            [
                object_embeddings.flatten(2, 3),
                history_embeddings.flatten(2, 3),
                trajectory_embeddings.flatten(2, 3),
            ],
            dim=2,
        )
        sequence_mask = torch.cat(
            [
                object_mask.flatten(2, 3),
                history_mask.flatten(2, 3),
                trajectory_mask.flatten(2, 3),
            ],
            dim=2,
        )

        for domain_index, embeddings in enumerate([object_embeddings, history_embeddings, trajectory_embeddings]):
            domain_encoding = torch.zeros_like(embeddings)
            div_term = torch.arange(0, domain_encoding.shape[-1] // 2, 1, device=domain_encoding.device).float()
            div_term = div_term / (domain_encoding.shape[-1] // 2) * 2 * np.pi
            domain_encoding[..., 0::2] = torch.sin(domain_index * div_term)
            domain_encoding[..., 1::2] = torch.cos(domain_index * div_term)
            embeddings += domain_encoding

        sequence_embeddings = torch.cat([sequence_embeddings, torch.zeros_like(sequence_embeddings[:, :, [0]])], dim=2)
        sequence_mask = torch.cat([sequence_mask, torch.zeros_like(sequence_mask[:, :, [0]])], dim=2)

        playstyle_tokens = self.playstyle_encoder.forward(
            src=sequence_embeddings.flatten(0, 1),
            src_key_padding_mask=sequence_mask.flatten(0, 1),
        )[..., -1, :].unflatten(0, (sequence_embeddings.shape[0], sequence_embeddings.shape[1]))
        playstyle_mask = torch.zeros(
            (playstyle_tokens.shape[0], playstyle_tokens.shape[1]),
            dtype=torch.bool,
            device=playstyle_tokens.device,
        )
        return playstyle_tokens, playstyle_mask

    def predict_logits_from_embeds(
        self,
        game_obj_embeds: torch.Tensor,
        game_obj_mask: torch.Tensor,
        game_hist_embeds: torch.Tensor,
        game_hist_mask: torch.Tensor,
        playstyle_tokens: torch.Tensor | None,
        playstyle_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        embedding_domains = [game_obj_embeds, game_hist_embeds]
        sequence_embeddings = [game_obj_embeds.flatten(1, 2), game_hist_embeds.flatten(1, 2)]
        sequence_mask = [game_obj_mask.flatten(1, 2), game_hist_mask.flatten(1, 2)]
        if playstyle_tokens is not None:
            embedding_domains.append(playstyle_tokens)
            sequence_embeddings.append(playstyle_tokens)
            sequence_mask.append(playstyle_mask)

        for domain_index, embeddings in enumerate(embedding_domains):
            domain_encoding = torch.zeros_like(embeddings)
            div_term = torch.arange(0, domain_encoding.shape[-1] // 2, 1, device=domain_encoding.device).float()
            div_term = div_term / (domain_encoding.shape[-1] // 2) * 2 * np.pi
            domain_encoding[..., 0::2] = torch.sin(domain_index * div_term)
            domain_encoding[..., 1::2] = torch.cos(domain_index * div_term)
            embeddings += domain_encoding

        sequence_embeddings = torch.cat(sequence_embeddings, dim=1)
        sequence_embeddings = torch.cat([sequence_embeddings, torch.zeros_like(sequence_embeddings[:, [0]])], dim=1)

        sequence_mask = torch.cat(sequence_mask, dim=1)
        sequence_mask = torch.cat([sequence_mask, torch.zeros_like(sequence_mask[:, [0]])], dim=-1)

        encoded_sequence = self.logit_predictor.forward(src=sequence_embeddings, src_key_padding_mask=sequence_mask)
        return self.deproj(encoded_sequence[:, -1])

    def decode(self, z: torch.Tensor) -> None:
        return None

    def embed_game_obj(self, notes: torch.Tensor, bombs: torch.Tensor, obstacles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        masks = []
        for object_bag, rms, projector in zip(
            [notes, bombs, obstacles],
            [self.note_rms, self.bomb_rms, self.obstacle_rms],
            [self.note_proj, self.bomb_proj, self.obstacle_proj],
        ):
            original_timestamps = object_bag[..., 0] * 1
            timestamps = original_timestamps * 1
            timestamps[torch.isnan(timestamps)] = 0

            object_bag = object_bag * 1
            object_bag[torch.isnan(object_bag)] = 0
            projected = projector(rms.normalize(object_bag)[..., 1:])

            time_encoding = torch.zeros_like(projected)
            div_term = torch.exp(
                torch.arange(0, projected.shape[-1] // 2, 1).float() * (-np.log(1000.0) / projected.shape[-1])
            ).to(projected.device)
            time_encoding[..., 0::2] = torch.sin(timestamps[..., None] * div_term)
            time_encoding[..., 1::2] = torch.cos(timestamps[..., None] * div_term)
            projected += time_encoding

            embeddings.append(projected)
            masks.append(torch.isnan(original_timestamps))
        return torch.stack(embeddings, dim=-3), torch.stack(masks, dim=-2)

    def embed_3p(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_timestamps = trajectory[..., 0] * 1
        trajectory[torch.isnan(trajectory)] = 0
        projected = self.threep_proj(
            self.threep_rms.normalize(trajectory).unflatten(-1, (-1, self.effective_3p_size))
        )
        projected = projected.flatten(-3, -2)

        position_encoding = torch.zeros_like(projected)
        div_term = torch.exp(
            torch.arange(0, projected.shape[-1] // 2, 1).float() * (-np.log(1000.0) / projected.shape[-1])
        ).to(projected.device)
        position_indices = torch.arange(projected.shape[-2], device=projected.device)
        position_encoding[..., 0::2] = torch.sin(position_indices[..., None] * div_term)
        position_encoding[..., 1::2] = torch.cos(position_indices[..., None] * div_term)
        projected += position_encoding

        projected = projected.unflatten(-2, (trajectory.shape[-2], -1))
        mask = torch.isnan(original_timestamps[..., None]).repeat_interleave(3, -1)
        return projected, mask

    def forward(
        self,
        game: GameTensors,
        playstyle: ReplayTensors,
        n: int = 1,
        temperature: float = 1.0,
        topk: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        logits = self.predict_logits(game, playstyle)
        return self.sample_from_z(logits, n, temperature, topk)

    def sample_from_z(
        self,
        z: torch.Tensor,
        n: int = 1,
        temperature: float = 1.0,
        topk: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        z = z.reshape(z.shape[0], 1, self.sentence_length, self.vocab_size).repeat_interleave(n, 1)
        if topk > 0:
            topk_values, topk_indices = torch.topk(z, topk, dim=-1)
            z = torch.full_like(z, -1e10).scatter(-1, topk_indices, topk_values)
        soft, hard = gumbel_softmax(z / temperature, tau=self.tau)
        if topk == 1:
            assert torch.all(hard.argmax(-1) == topk_indices.squeeze(-1))
        decoded_tokens = torch.eye(z.shape[-1], device=z.device)[z.argmax(-1)].flatten(-2, -1)
        soft = soft.reshape(z.shape).flatten(-2, -1)
        hard = hard.reshape(z.shape).flatten(-2, -1)
        return z, decoded_tokens, soft, hard, None


class GameplayEncoder(nn.Module, InvarMixin):
    def __init__(
        self,
        note_size: int,
        bomb_size: int,
        obstacle_size: int,
        history_size: int,
        hidden_size: int,
        embed_size: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.note_size = note_size
        self.bomb_size = bomb_size
        self.obstacle_size = obstacle_size
        self.threep_size = history_size
        self.effective_3p_size = self.threep_size // 3

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = 0.0
        self.embed_size = embed_size
        self.activation = nn.GELU()

        self.note_proj = nn.Sequential(
            nn.Linear(self.note_size - 1, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.bomb_proj = nn.Sequential(
            nn.Linear(self.bomb_size - 1, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.obstacle_proj = nn.Sequential(
            nn.Linear(self.obstacle_size - 1, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.threep_proj = nn.Sequential(
            nn.Linear(self.effective_3p_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.tenc = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.note_rms = RunningMeanStd(shape=(self.note_size,))
        self.bomb_rms = RunningMeanStd(shape=(self.bomb_size,))
        self.obstacle_rms = RunningMeanStd(shape=(self.obstacle_size,))
        self.threep_rms = RunningMeanStd(shape=(self.threep_size,))
        self.tau = 1

    def setup(
        self,
        notes: torch.Tensor,
        bombs: torch.Tensor,
        obstacles: torch.Tensor,
        history: torch.Tensor,
    ) -> None:
        for stream, rms in zip(
            [notes, bombs, obstacles],
            [self.note_rms, self.bomb_rms, self.obstacle_rms],
        ):
            nan_mask = torch.isnan(stream).any(-1)
            if (~nan_mask).sum() > 0:
                rms.update(stream[~nan_mask])
        self.threep_rms.update(self.pack_invar(history[:, None]).reshape(-1, self.threep_size))

    def encode(self, replay: ReplayTensors) -> torch.Tensor:
        assert replay.history is not None
        assert replay.trajectory is not None
        replay.obstacles[..., 0] = torch.clip(replay.obstacles[..., 0], min=0)
        object_embeddings, object_mask = self.embed_game_obj(replay.notes, replay.bombs, replay.obstacles)
        history_embeddings, history_mask = self.embed_3p(self.pack_invar(replay.history))
        trajectory_embeddings, trajectory_mask = self.embed_3p(self.pack_invar(replay.trajectory))

        embedding_domains = [object_embeddings, history_embeddings, trajectory_embeddings]
        sequence_embeddings = [
            object_embeddings.flatten(-3, -2),
            history_embeddings.flatten(-3, -2),
            trajectory_embeddings.flatten(-3, -2),
        ]
        sequence_mask = [
            object_mask.flatten(-2, -1),
            history_mask.flatten(-2, -1),
            trajectory_mask.flatten(-2, -1),
        ]

        for domain_index, embeddings in enumerate(embedding_domains):
            domain_encoding = torch.zeros_like(embeddings)
            div_term = torch.arange(0, domain_encoding.shape[-1] // 2, 1, device=domain_encoding.device).float()
            div_term = div_term / (domain_encoding.shape[-1] // 2) * 2 * np.pi
            domain_encoding[..., 0::2] = torch.sin(domain_index * div_term)
            domain_encoding[..., 1::2] = torch.cos(domain_index * div_term)
            embeddings += domain_encoding

        sequence_embeddings = torch.cat(sequence_embeddings, dim=-2)
        sequence_embeddings = torch.cat([sequence_embeddings, torch.zeros_like(sequence_embeddings[..., [0], :])], dim=-2)

        sequence_mask = torch.cat(sequence_mask, dim=-1)
        sequence_mask = torch.cat([sequence_mask, torch.zeros_like(sequence_mask[..., [0]])], dim=-1)

        encoded_sequence = self.tenc.forward(
            src=sequence_embeddings.flatten(0, 1),
            src_key_padding_mask=sequence_mask.flatten(0, 1),
        ).unflatten(0, (sequence_embeddings.shape[0], sequence_embeddings.shape[1]))
        return encoded_sequence[..., -1, :]

    def embed_game_obj(self, notes: torch.Tensor, bombs: torch.Tensor, obstacles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        masks = []
        for object_bag, rms, projector in zip(
            [notes, bombs, obstacles],
            [self.note_rms, self.bomb_rms, self.obstacle_rms],
            [self.note_proj, self.bomb_proj, self.obstacle_proj],
        ):
            original_timestamps = object_bag[..., 0] * 1
            timestamps = original_timestamps * 1
            timestamps[torch.isnan(timestamps)] = 0

            object_bag = object_bag * 1
            object_bag[torch.isnan(object_bag)] = 0
            projected = projector(rms.normalize(object_bag)[..., 1:])

            time_encoding = torch.zeros_like(projected)
            div_term = torch.exp(
                torch.arange(0, projected.shape[-1] // 2, 1).float() * (-np.log(1000.0) / projected.shape[-1])
            ).to(projected.device)
            time_encoding[..., 0::2] = torch.sin(timestamps[..., None] * div_term)
            time_encoding[..., 1::2] = torch.cos(timestamps[..., None] * div_term)
            projected += time_encoding

            embeddings.append(projected)
            masks.append(torch.isnan(original_timestamps))
        return torch.stack(embeddings, dim=-3), torch.stack(masks, dim=-2)

    def embed_3p(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_timestamps = trajectory[..., 0] * 1
        trajectory[torch.isnan(trajectory)] = 0
        projected = self.threep_proj(
            self.threep_rms.normalize(trajectory).unflatten(-1, (-1, self.effective_3p_size))
        )
        projected = projected.flatten(-3, -2)

        position_encoding = torch.zeros_like(projected)
        div_term = torch.exp(
            torch.arange(0, projected.shape[-1] // 2, 1).float() * (-np.log(1000.0) / projected.shape[-1])
        ).to(projected.device)
        position_indices = torch.arange(projected.shape[-2], device=projected.device)
        position_encoding[..., 0::2] = torch.sin(position_indices[..., None] * div_term)
        position_encoding[..., 1::2] = torch.cos(position_indices[..., None] * div_term)
        projected += position_encoding

        projected = projected.unflatten(-2, (trajectory.shape[-2], -1))
        mask = torch.isnan(original_timestamps[..., None]).repeat_interleave(3, -1)
        return projected, mask

    def forward(self, replay: ReplayTensors) -> torch.Tensor:
        return self.encode(replay)


class SentinelPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = 0.0
        self.activation = nn.GELU()

        self.deproj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_size,
            nhead=self.num_heads,
            dim_feedforward=self.input_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.tenc = nn.TransformerEncoder(encoder_layer, self.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, torch.zeros_like(x[..., [0], :])], dim=-2)
        return self.deproj(self.tenc(x)[..., -1, :])


class TransformerGSVAE(nn.Module, InvarMixin):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        embed_size: int,
        vocab_size: int,
        sentence_length: int,
        chunk_length: int,
        stride: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.effective_input_size = self.input_size // 3
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.chunk_length = chunk_length
        self.stride = stride
        self.effective_chunk_length = chunk_length // self.stride
        self.dropout = 0.0
        self.activation = nn.GELU()

        self.proj = nn.Sequential(
            nn.Linear(self.effective_input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.embed_size * self.effective_chunk_length * 3, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.vocab_size * self.sentence_length),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.vocab_size * self.sentence_length, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size * self.effective_chunk_length * 3),
        )
        self.deproj = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.effective_input_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.decoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.input_rms = RunningMeanStd(shape=(self.effective_input_size,))
        self.tau = 1.0

    def setup(self, my_3p: torch.Tensor) -> None:
        packed_3p = self.pack_invar(my_3p[:, None]).unflatten(-1, (3, -1))
        self.input_rms.update(packed_3p.reshape(-1, self.effective_input_size))

    def encode(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pack_invar(x).unflatten(-1, (3, -1))
        x = torch.clamp(self.input_rms.normalize(x), -10, 10)
        z = self.proj(x)
        z = z.reshape(z.shape[0], -1, self.effective_chunk_length * 3, self.embed_size)

        position_encoding = torch.zeros_like(z)
        position_indices = torch.arange(z.shape[-2], device=z.device)
        div_term = torch.exp(
            torch.arange(0, z.shape[-1] // 2, 1).float() * (-np.log(1000.0) / z.shape[-1])
        ).to(z.device)
        position_encoding[..., 0::2] = torch.sin(position_indices[..., None] * div_term[None])
        position_encoding[..., 1::2] = torch.cos(position_indices[..., None] * div_term[None])
        z = z + position_encoding

        z = self.encoder(z.reshape(-1, *z.shape[2:])).reshape(z.shape)
        z = self.encoder_fc(z.reshape(z.shape[0], z.shape[1], -1))
        return z.reshape(z.shape[0], -1, self.sentence_length * self.vocab_size)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder_fc(z).unflatten(-1, (self.effective_chunk_length * 3, self.embed_size))

        position_encoding = torch.zeros_like(z)
        position_indices = torch.arange(z.shape[-2], device=z.device)
        div_term = torch.exp(
            torch.arange(0, z.shape[-1] // 2, 1).float() * (-np.log(1000.0) / z.shape[-1])
        ).to(z.device)
        position_encoding[..., 0::2] = torch.sin(position_indices[..., None] * div_term[None])
        position_encoding[..., 1::2] = torch.cos(position_indices[..., None] * div_term[None])
        z = z + position_encoding

        x = self.decoder(z.flatten(0, -3)).reshape(z.shape)
        x = self.deproj(x)
        x = x.unflatten(-2, (self.effective_chunk_length, 3)).flatten(-2, -1)
        return self.unpack_invar(x)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor | None = None,
        n: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x, w)
        z = z.repeat_interleave(n, 1).unflatten(-1, (self.sentence_length, self.vocab_size))
        soft, hard = gumbel_softmax(z, tau=self.tau)
        token_ids = hard.argmax(-1)
        soft = soft.flatten(-2, -1)
        hard = hard.flatten(-2, -1)
        return z, token_ids, soft, hard, self.decode(hard)
