import numpy as np
import torch
from torch import nn

from beaty_common.segments import MovementSegment

def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1,
    eps: float = 1e-20,
    dim: int = -1,
) -> (torch.Tensor, torch.Tensor):
    u = torch.rand_like(logits)
    gumbels = -torch.log(-torch.log(u + eps) + eps)
    # dist = torch.distributions.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    # gumbels = dist.sample()
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    # Straight through.
    index = y_soft.argmax(dim)
    y_hard = torch.eye(logits.shape[-1], device=logits.device)[index]
    y_hard = y_hard - y_soft.detach() + y_soft
    return y_soft, y_hard


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape=(), *args, **kwargs):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float), requires_grad=False)
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip((arr - self.mean) / torch.sqrt(self.var + self.epsilon), -5, 5)

    def unnormalize(self, arr: torch.Tensor) -> torch.Tensor:
        return arr * torch.sqrt(self.var + self.epsilon) + self.mean


class ConvRes1d(nn.Module):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        chunk_length: int,
        padding: int,
        n_convs: int,
    ):
        super().__init__()
        self.conv1s, self.convs2 = (
            nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=padding,
                    )
                ]
                + [
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=padding,
                    )
                    for _ in range(n_convs)
                ]
                + [
                    nn.Conv1d(
                        out_channels,
                        in_channels,
                        kernel_size,
                        stride=1,
                        padding=padding,
                    )
                ]
            )
            for _ in range(2)
        )
        self.activation = nn.LeakyReLU()
        self.lns = nn.ModuleList([nn.LayerNorm([in_channels, chunk_length])] + [nn.LayerNorm([out_channels, chunk_length]) for _ in range(n_convs)] + [nn.LayerNorm([out_channels, chunk_length])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x * 1
        for i, (conv1, conv2, ln) in enumerate(zip(self.conv1s, self.convs2, self.lns)):
            x = ln(x)
            x = conv1(x)
            x = self.activation(x)
            x = xx + conv2(x)
            xx = x * 1
            if i < len(self.conv1s) - 1:
                x = self.activation(x)
        return x


class InvarMixin1:
    def pack_invar(self, w: torch.Tensor) -> torch.Tensor:
        # ww = w.reshape(*w.shape[:-1], 3, -1) * 1
        ww = w.unflatten(-1, (3, -1)) * 1
        ww[..., 1:, :] -= ww[..., :1, :]
        ww = ww.reshape(w.shape)
        return ww

    def unpack_invar(self, w: torch.Tensor) -> torch.Tensor:
        # ww = w.reshape(*w.shape[:-1], 3, -1) * 1
        ww = w.unflatten(-1, (3, -1)) * 1
        ww[..., 1:, :] += ww[..., :1, :]
        ww = ww.reshape(w.shape)
        return ww


class InvarMixin:
    def pack_invar(self, w: torch.Tensor) -> torch.Tensor:
        ww = w.unflatten(-1, (3, -1)) * 1
        ww[..., 1:, :] -= ww[..., :1, :]
        return ww

    def unpack_invar(self, w: torch.Tensor) -> torch.Tensor:
        ww = w.unflatten(-1, (3, -1)) * 1
        ww[..., 1:, :] += ww[..., :1, :]
        ww = ww.flatten(-2, -1)
        return ww


class MlpGSVAE(nn.Module, InvarMixin1):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        input_frames: int,
        hidden_size: int,
        vocab_size: int,
        sentence_length: int,
        stride: int,
    ):
        super().__init__()
        self.latent_activation = nn.Identity()
        self.input_rms = RunningMeanStd(shape=(input_size,))
        self.input_size = input_size
        self.input_frames = input_frames
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.stride = stride
        self.latent_size = self.sentence_length * self.vocab_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size * self.input_frames // self.stride, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(
                self.input_size * self.input_frames // self.stride + self.hidden_size,
                self.hidden_size,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.input_size * self.input_frames // self.stride + self.hidden_size,
                self.hidden_size,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.input_size * self.input_frames // self.stride + self.hidden_size,
                self.hidden_size,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.input_size * self.input_frames // self.stride + self.hidden_size,
                self.latent_size,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(
                self.latent_size + self.hidden_size,
                self.input_size * self.input_frames // self.stride,
            ),
        )
        self.tau = 1.0

    def setup(self, output_segment: MovementSegment):
        self.input_rms.update(self.pack_invar(output_segment.three_p).reshape(-1, self.input_size))

    def encode(self, x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
        xx = self.input_rms.normalize(self.pack_invar(x))
        xx = xx.reshape(xx.shape[0], -1)
        z = xx * 1
        for j, op in enumerate(self.encoder):
            if isinstance(op, nn.Linear):
                if j > 0:
                    z = torch.cat([xx, z], dim=-1)
            z = op(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = z * 1
        x = zz * 1
        for j, op in enumerate(self.decoder):
            if isinstance(op, nn.Linear):
                if j > 0:
                    x = torch.cat([zz, x], dim=-1)
            x = op(x)
        x = x.reshape(x.shape[0], -1, self.input_frames // self.stride, self.input_size)
        x = self.unpack_invar(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor = None,
        n: int = 1,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        zz = z.reshape(z.shape[0], 1, self.sentence_length, self.vocab_size).repeat_interleave(n, 1)
        soft, hard = gumbel_softmax(zz, tau=self.tau)
        k = hard.argmax(-1)
        soft = soft.reshape(z.shape[0], -1, z.shape[-1])
        hard = hard.reshape(z.shape[0], -1, z.shape[-1])
        # x_hat = self.decode(hard)
        x_hat = self.decode(hard)
        return z, k, soft, hard, x_hat


class CondMlpGSVAE(nn.Module):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        input_frames: int,
        cond_size: int,
        cond_frames: int,
        hidden_size: int,
        vocab_size: int,
        sentence_length: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_frames = input_frames
        self.cond_size = cond_size
        self.cond_frames = cond_frames
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.latent_size = self.sentence_length * self.vocab_size

        self.cond_size = cond_size
        self.encoder = nn.Sequential(
            nn.Linear(
                self.input_size * self.input_frames + self.cond_size * self.cond_frames,
                self.hidden_size,
            ),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.latent_size),
        )
        self.input_rms = RunningMeanStd(shape=(self.input_size,))
        self.cond_rms = RunningMeanStd(shape=(self.cond_size,))
        self.tau = 1

    def setup(self, x: torch.Tensor, w: torch.Tensor):
        xx = x * 1
        xx = xx.reshape(-1, xx.shape[-1])
        nan_yes = torch.isnan(xx).any(-1)
        self.input_rms.update(xx[~nan_yes])
        self.cond_rms.update(w.reshape(-1, w.shape[-1]))

    def encode(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xx = self.input_rms.normalize(x)
        xx[xx.isnan()] = 0
        ww = self.cond_rms.normalize(w)
        ww[ww.isnan()] = 0
        encoder_in = torch.cat([xx.reshape(xx.shape[0], -1), ww.reshape(ww.shape[0], -1)], dim=-1)
        z = self.encoder(encoder_in)
        return z

    def forward(self, x: torch.Tensor, w: torch.Tensor = None, n: int = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        zz = z.reshape(z.shape[0], self.sentence_length, self.vocab_size)
        soft, hard = gumbel_softmax(zz, tau=self.tau)
        k = hard.argmax(-1)
        soft = soft.reshape(z.shape)
        hard = hard.reshape(z.shape)
        x_hat = None
        return z, k, hard, x_hat


class CondTransformerGSVAE(nn.Module, InvarMixin1):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

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
    ):
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
        # self.bias = nn.Parameter(torch.zeros(self.hidden_size))

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
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=self.embed_size,
        #     nhead=self.num_heads,
        #     dim_feedforward=self.hidden_size,
        #     dropout=self.dropout,
        #     activation=self.activation,
        #     batch_first=True,
        #     bias=True,
        # )
        # self.logit_predictor = nn.TransformerDecoder(decoder_layer, self.num_layers)
        # self.logit_predictor = nn.Transformer(
        #     d_model=self.embed_size,
        #     nhead=self.num_heads,
        #     num_encoder_layers=self.num_layers,
        #     num_decoder_layers=self.num_layers,
        #     dim_feedforward=self.hidden_size,
        #     dropout=self.dropout,
        #     activation=self.activation,
        #     custom_encoder=self.playstyle_encoder,
        #     custom_decoder=self.logit_predictor,
        # )

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
    ):
        for x, rms in zip(
            [notes, bombs, obstacles],
            [self.note_rms, self.bomb_rms, self.obstacle_rms],
        ):
            nan_yes = torch.isnan(x).any(-1)
            if (~nan_yes).sum() > 0:
                rms.update(x[~nan_yes])

        self.threep_rms.update(self.pack_invar(history[:, None]).reshape(-1, self.threep_size))

    def encode_game(
        self,
        game_notes: torch.Tensor,
        game_bombs: torch.Tensor,
        game_obstacles: torch.Tensor,
        game_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        game_obstacles[..., 0] = torch.clip(game_obstacles[..., 0], min=0)
        game_obj_embeds, game_obj_mask = self.embed_game_obj(game_notes, game_bombs, game_obstacles)
        game_hist_embeds, game_hist_mask = self.embed_3p(self.pack_invar(game_history))
        return game_obj_embeds, game_obj_mask, game_hist_embeds, game_hist_mask

    def predict_logits(
        self,
        game_notes: torch.Tensor,
        game_bombs: torch.Tensor,
        game_obstacles: torch.Tensor,
        game_history: torch.Tensor,
        playstyle_notes: torch.Tensor,
        playstyle_bombs: torch.Tensor,
        playstyle_obstacles: torch.Tensor,
        playstyle_history: torch.Tensor,
        playstyle_3p: torch.Tensor,
    ) -> torch.Tensor:
        playstyle_obstacles[..., 0] = torch.clip(playstyle_obstacles[..., 0], min=0)
        game_obj_embeds, game_obj_mask, game_hist_embeds, game_hist_mask = self.encode_game(game_notes, game_bombs, game_obstacles, game_history)
        n_refs = playstyle_notes.shape[1]
        playstyle_tokens = None
        playstyle_mask = None
        if n_refs > 0:
            playstyle_tokens, playstyle_mask = self.encode_style(
                playstyle_notes,
                playstyle_bombs,
                playstyle_obstacles,
                playstyle_history,
                playstyle_3p,
            )

        z = self.predict_logits_from_embeds(
            game_obj_embeds,
            game_obj_mask,
            game_hist_embeds,
            game_hist_mask,
            playstyle_tokens,
            playstyle_mask,
        )
        return z

    def encode_style(
        self,
        playstyle_notes: torch.Tensor,
        playstyle_bombs: torch.Tensor,
        playstyle_obstacles: torch.Tensor,
        playstyle_history: torch.Tensor,
        playstyle_3p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # print("encode_style")
        playstyle_obj_embeds, playstyle_obj_mask = self.embed_game_obj(playstyle_notes, playstyle_bombs, playstyle_obstacles)
        playstyle_hist_embeds, playstyle_hist_mask = self.embed_3p(self.pack_invar(playstyle_history))
        playstyle_3p_embeds, playstyle_3p_mask = self.embed_3p(self.pack_invar(playstyle_3p))

        # For permutation invariance, turn the example sequences into example tokens
        emb_catted = torch.cat(
            [
                playstyle_obj_embeds.flatten(2, 3),
                playstyle_hist_embeds.flatten(2, 3),
                playstyle_3p_embeds.flatten(2, 3),
            ],
            dim=2,
        )
        mask_catted = torch.cat(
            [
                playstyle_obj_mask.flatten(2, 3),
                playstyle_hist_mask.flatten(2, 3),
                playstyle_3p_mask.flatten(2, 3),
            ],
            dim=2,
        )
        domain_encode_me = [
            playstyle_obj_embeds,
            playstyle_hist_embeds,
            playstyle_3p_embeds,
        ]
        # Add domain encodings
        for i, embeds in enumerate(domain_encode_me):
            de = torch.zeros_like(embeds)
            div_term = torch.arange(0, de.shape[-1] // 2, 1, device=de.device).float() / (de.shape[-1] // 2) * 2 * np.pi
            de[..., 0::2] = torch.sin(i * div_term)
            de[..., 1::2] = torch.cos(i * div_term)
            embeds += de

        # Register a dummy token to represent the end of the sequence
        emb_catted = torch.cat([emb_catted, torch.zeros_like(emb_catted[:, :, [0]])], dim=2)
        mask_catted = torch.cat([mask_catted, torch.zeros_like(mask_catted[:, :, [0]])], dim=2)

        # Encode the playstyle references independently
        playstyle_tokens = self.playstyle_encoder.forward(src=emb_catted.flatten(0, 1), src_key_padding_mask=mask_catted.flatten(0, 1))[..., -1, :].unflatten(0, (emb_catted.shape[0], emb_catted.shape[1]))
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
        enumerate_me = [game_obj_embeds, game_hist_embeds]
        cat_me_embeds = [game_obj_embeds.flatten(1, 2), game_hist_embeds.flatten(1, 2)]
        cat_me_mask = [game_obj_mask.flatten(1, 2), game_hist_mask.flatten(1, 2)]
        if playstyle_tokens is not None:
            enumerate_me = [game_obj_embeds, game_hist_embeds, playstyle_tokens]
            cat_me_embeds = [
                game_obj_embeds.flatten(1, 2),
                game_hist_embeds.flatten(1, 2),
                playstyle_tokens,
            ]
            cat_me_mask = [
                game_obj_mask.flatten(1, 2),
                game_hist_mask.flatten(1, 2),
                playstyle_mask,
            ]

        # The above modules' `forward()` method already applies TTA and position encoding
        # Apply domain encoding
        # Importantly, the playstyle references should be permutation-invariant
        for i, embeds in enumerate(enumerate_me):
            de = torch.zeros_like(embeds)
            div_term = torch.arange(0, de.shape[-1] // 2, 1, device=de.device).float() / (de.shape[-1] // 2) * 2 * np.pi
            de[..., 0::2] = torch.sin(i * div_term)
            de[..., 1::2] = torch.cos(i * div_term)
            embeds += de

        xxww = torch.cat(cat_me_embeds, dim=1)

        # # Add position embeddings
        # pe = torch.zeros_like(xxww)
        # idxs = torch.arange(xxww.shape[1], device=xxww.device)
        # div_term = torch.exp(torch.arange(0, xxww.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (xxww.shape[-1]))).to(xxww.device)
        # pe[..., 0::2] = torch.sin(idxs[..., None] * div_term[None])
        # pe[..., 1::2] = torch.cos(idxs[..., None] * div_term[None])
        # xxww = xxww + pe

        xxww = torch.cat([xxww, torch.zeros_like(xxww[:, [0]])], dim=1)

        mask_simple = torch.cat(cat_me_mask, dim=1)
        mask_simple = torch.cat([mask_simple, torch.zeros_like(mask_simple[:, [0]])], dim=-1)
        mask = mask_simple[..., None].repeat_interleave(mask_simple.shape[1], -1).permute(0, 2, 1)
        mask = mask.repeat_interleave(self.num_heads, 0)
        # embedded_seq = self.encoder.forward(src=xxww, mask=mask)
        # if playstyle_tokens is not None:
        #     embedded_seq = self.logit_predictor.forward(tgt=xxww, tgt_key_padding_mask=mask_simple, memory=playstyle_tokens)
        # else:
        #     embedded_seq = self.logit_predictor.forward(tgt=xxww, tgt_key_padding_mask=mask_simple)
        # embedded_seq = self.logit_predictor.forward(src=xxww, mask=mask)
        embedded_seq = self.logit_predictor.forward(src=xxww, src_key_padding_mask=mask_simple)
        z = self.deproj(embedded_seq[:, -1])

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return None

    def embed_game_obj(self, notes: torch.Tensor, bombs: torch.Tensor, obstacles: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        embeddings = []
        masks = []
        for i, (x, rms, proj) in enumerate(
            zip(
                [notes, bombs, obstacles],
                [self.note_rms, self.bomb_rms, self.obstacle_rms],
                [self.note_proj, self.bomb_proj, self.obstacle_proj],
            )
        ):
            t_orig = x[..., 0] * 1
            t = t_orig * 1
            t[torch.isnan(t)] = 0

            # Form x stuff
            x = x * 1
            x[torch.isnan(x)] = 0

            xx = proj(rms.normalize(x)[..., 1:])

            # TTA encoding on nail embeddings
            pe = torch.zeros_like(xx)
            div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (xx.shape[-1]))).to(xx.device)
            pe[..., 0::2] = torch.sin(t[..., None] * div_term)
            pe[..., 1::2] = torch.cos(t[..., None] * div_term)
            xx += pe

            # # Domain encoding, just use indices
            # de = torch.zeros_like(xx)
            # div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(10000.0) / (xx.shape[-1]))).to(xx.device)
            # de[..., 0::2] = torch.sin(i * div_term)
            # de[..., 1::2] = torch.cos(i * div_term)
            # xx += de

            x_mask = torch.isnan(t_orig)
            # x_mask = x_mask.reshape(x_mask.shape[0], -1)

            embeddings.append(xx)
            masks.append(x_mask)
        embeddings = torch.stack(embeddings, dim=-3)
        masks = torch.stack(masks, dim=-2)
        return embeddings, masks

    def embed_3p(self, my_3p: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = my_3p[..., 0] * 1
        my_3p[torch.isnan(my_3p)] = 0
        # ww = self.history_proj(self.history_rms.normalize(history))
        ww = self.threep_proj(self.threep_rms.normalize(my_3p).unflatten(-1, (-1, self.effective_3p_size)))

        ww = ww.flatten(-3, -2)

        pe = torch.zeros_like(ww)
        div_term = torch.exp(torch.arange(0, ww.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (ww.shape[-1]))).to(ww.device)
        pe[..., 0::2] = torch.sin(torch.arange(ww.shape[-2], device=ww.device)[..., None] * div_term)
        pe[..., 1::2] = torch.cos(torch.arange(ww.shape[-2], device=ww.device)[..., None] * div_term)
        ww += pe

        ww = ww.unflatten(-2, (my_3p.shape[-2], -1))

        # # TODO: Domain encoding can be done with nn.embedding...? probably not important
        # # History domain encoding
        # de = torch.zeros_like(ww)
        # div_term = torch.exp(torch.arange(0, ww.shape[-1] // 2, 1).float() * (-np.log(10000.0) / (ww.shape[-1]))).to(ww.device)
        # de[..., 0::2] = torch.sin(3 * div_term)
        # de[..., 1::2] = torch.cos(3 * div_term)
        # ww += de

        w_mask = torch.isnan(t_orig[..., None]).repeat_interleave(3, -1)

        return ww, w_mask

    def forward(
        self,
        game_notes: torch.Tensor,
        game_bombs: torch.Tensor,
        game_obstacles: torch.Tensor,
        game_history: torch.Tensor,
        playstyle_notes: torch.Tensor,
        playstyle_bombs: torch.Tensor,
        playstyle_obstacles: torch.Tensor,
        playstyle_history: torch.Tensor,
        playstyle_3p: torch.Tensor,
        n: int = 1,
        temperature: float = 1.0,
        topk: int = 0,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.predict_logits(
            game_notes,
            game_bombs,
            game_obstacles,
            game_history,
            playstyle_notes,
            playstyle_bombs,
            playstyle_obstacles,
            playstyle_history,
            playstyle_3p,
        )
        return self.sample_from_z(z, n, temperature, topk)

    def sample_from_z(
        self,
        z: torch.Tensor,
        n: int = 1,
        temperature: float = 1.0,
        topk: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = z.reshape(z.shape[0], 1, self.sentence_length, self.vocab_size).repeat_interleave(n, 1)
        if topk > 0:
            # Top-k sampling: use indices to mask exactly k values (handles ties correctly)
            topk_values, topk_indices = torch.topk(z, topk, dim=-1)
            z = torch.full_like(z, -1e10).scatter(-1, topk_indices, topk_values)
        soft, hard = gumbel_softmax(z / temperature, tau=self.tau)
        if topk == 1:
            assert torch.all(hard.argmax(-1) == topk_indices.squeeze(-1)), "topk=1 should match argmax"
        k = torch.eye(z.shape[-1], device=z.device)[z.argmax(-1)].flatten(-2, -1)
        soft = soft.reshape(z.shape).flatten(-2, -1)
        hard = hard.reshape(z.shape).flatten(-2, -1)
        x_hat = None
        return z, k, soft, hard, x_hat


class GameplayEncoder(nn.Module, InvarMixin1):

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
    ):
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
    ):
        for x, rms in zip(
            [notes, bombs, obstacles],
            [self.note_rms, self.bomb_rms, self.obstacle_rms],
        ):
            nan_yes = torch.isnan(x).any(-1)
            if (~nan_yes).sum() > 0:
                rms.update(x[~nan_yes])

        self.threep_rms.update(self.pack_invar(history[:, None]).reshape(-1, self.threep_size))

    def encode(
        self,
        game_notes: torch.Tensor,
        game_bombs: torch.Tensor,
        game_obstacles: torch.Tensor,
        game_history: torch.Tensor,
        game_3p: torch.Tensor,
    ) -> torch.Tensor:
        game_obstacles[..., 0] = torch.clip(game_obstacles[..., 0], min=0)
        game_obj_embeds, game_obj_mask = self.embed_game_obj(game_notes, game_bombs, game_obstacles)
        game_hist_embeds, game_hist_mask = self.embed_3p(self.pack_invar(game_history))
        game_3p_embeds, game_3p_mask = self.embed_3p(self.pack_invar(game_3p))
        enumerate_me = [game_obj_embeds, game_hist_embeds, game_3p_embeds]
        cat_me_embeds = [
            game_obj_embeds.flatten(-3, -2),
            game_hist_embeds.flatten(-3, -2),
            game_3p_embeds.flatten(-3, -2),
        ]
        cat_me_mask = [
            game_obj_mask.flatten(-2, -1),
            game_hist_mask.flatten(-2, -1),
            game_3p_mask.flatten(-2, -1),
        ]

        # The above modules' `forward()` method already applies TTA and position encoding
        # Apply domain encoding
        # Importantly, the playstyle references should be permutation-invariant
        for i, embeds in enumerate(enumerate_me):
            de = torch.zeros_like(embeds)
            div_term = torch.arange(0, de.shape[-1] // 2, 1, device=de.device).float() / (de.shape[-1] // 2) * 2 * np.pi
            de[..., 0::2] = torch.sin(i * div_term)
            de[..., 1::2] = torch.cos(i * div_term)
            embeds += de

        xxww = torch.cat(cat_me_embeds, dim=-2)

        # Sentinel <CLS> token
        xxww = torch.cat([xxww, torch.zeros_like(xxww[..., [0], :])], dim=-2)

        mask_simple = torch.cat(cat_me_mask, dim=-1)
        mask_simple = torch.cat([mask_simple, torch.zeros_like(mask_simple[..., [0]])], dim=-1)
        embedded_seq = self.tenc.forward(src=xxww.flatten(0, 1), src_key_padding_mask=mask_simple.flatten(0, 1)).unflatten(0, (xxww.shape[0], xxww.shape[1]))
        z = embedded_seq[..., -1, :]
        return z

    def embed_game_obj(self, notes: torch.Tensor, bombs: torch.Tensor, obstacles: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        embeddings = []
        masks = []
        for i, (x, rms, proj) in enumerate(
            zip(
                [notes, bombs, obstacles],
                [self.note_rms, self.bomb_rms, self.obstacle_rms],
                [self.note_proj, self.bomb_proj, self.obstacle_proj],
            )
        ):
            t_orig = x[..., 0] * 1
            t = t_orig * 1
            t[torch.isnan(t)] = 0

            # Form x stuff
            x = x * 1
            x[torch.isnan(x)] = 0

            xx = proj(rms.normalize(x)[..., 1:])

            # TTA encoding on nail embeddings
            pe = torch.zeros_like(xx)
            div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (xx.shape[-1]))).to(xx.device)
            pe[..., 0::2] = torch.sin(t[..., None] * div_term)
            pe[..., 1::2] = torch.cos(t[..., None] * div_term)
            xx += pe

            # # Domain encoding, just use indices
            # de = torch.zeros_like(xx)
            # div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(10000.0) / (xx.shape[-1]))).to(xx.device)
            # de[..., 0::2] = torch.sin(i * div_term)
            # de[..., 1::2] = torch.cos(i * div_term)
            # xx += de

            x_mask = torch.isnan(t_orig)
            # x_mask = x_mask.reshape(x_mask.shape[0], -1)

            embeddings.append(xx)
            masks.append(x_mask)
        embeddings = torch.stack(embeddings, dim=-3)
        masks = torch.stack(masks, dim=-2)
        return embeddings, masks

    def embed_3p(self, my_3p: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = my_3p[..., 0] * 1
        my_3p[torch.isnan(my_3p)] = 0
        # ww = self.history_proj(self.history_rms.normalize(history))
        ww = self.threep_proj(self.threep_rms.normalize(my_3p).unflatten(-1, (-1, self.effective_3p_size)))

        ww = ww.flatten(-3, -2)

        pe = torch.zeros_like(ww)
        div_term = torch.exp(torch.arange(0, ww.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (ww.shape[-1]))).to(ww.device)
        pe[..., 0::2] = torch.sin(torch.arange(ww.shape[-2], device=ww.device)[..., None] * div_term)
        pe[..., 1::2] = torch.cos(torch.arange(ww.shape[-2], device=ww.device)[..., None] * div_term)
        ww += pe

        ww = ww.unflatten(-2, (my_3p.shape[-2], -1))

        w_mask = torch.isnan(t_orig[..., None]).repeat_interleave(3, -1)

        return ww, w_mask

    def forward(
        self,
        game_notes: torch.Tensor,
        game_bombs: torch.Tensor,
        game_obstacles: torch.Tensor,
        game_history: torch.Tensor,
        game_3p: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(game_notes, game_bombs, game_obstacles, game_history, game_3p)
        return z


class SentinelPredictor(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
    ):
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
        # Sentinel <CLS> token at the end
        x = torch.cat([x, torch.zeros_like(x[..., [0], :])], dim=-2)
        z = self.tenc(x)
        z = self.deproj(z[..., -1, :])
        return z


class ConvGSVAE(nn.Module, InvarMixin):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        vocab_size: int,
        sentence_length: int,
        chunk_length: int,
        padding: int,
    ):
        super().__init__()
        self.latent_activation = nn.Identity()
        self.input_size = input_size
        self.input_rms = RunningMeanStd(shape=(input_size,))
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.latent_size = self.sentence_length * self.vocab_size
        self.vocab_size = vocab_size
        self.chunk_length = chunk_length
        self.padding = padding
        n_convs = 1
        n_blocks = 1

        self.encoder_proj = nn.Linear(self.input_size * self.chunk_length, self.hidden_size * self.sentence_length)
        self.encoder = nn.Sequential(
            *(
                nn.Sequential(
                    nn.LeakyReLU(),
                    ConvRes1d(
                        self.hidden_size,
                        self.hidden_size,
                        self.kernel_size,
                        self.sentence_length,
                        padding=self.padding,
                        n_convs=n_convs,
                    ),
                )
                for _ in range(n_blocks)
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                self.kernel_size,
                padding=self.padding,
            ),
        )
        self.encoder_fc = nn.Linear(
            self.hidden_size * self.sentence_length,
            self.vocab_size * self.sentence_length,
        )
        self.decoder_fc = nn.Linear(
            self.vocab_size * self.sentence_length,
            self.hidden_size * self.sentence_length,
        )
        self.decoder = nn.Sequential(
            *(
                nn.Sequential(
                    nn.LeakyReLU(),
                    ConvRes1d(
                        self.hidden_size,
                        self.hidden_size,
                        self.kernel_size,
                        self.sentence_length,
                        padding=self.padding,
                        n_convs=n_convs,
                    ),
                )
                for _ in range(n_blocks)
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                self.kernel_size,
                padding=self.padding,
            ),
        )
        self.decoder_deproj = nn.Linear(self.hidden_size * self.sentence_length, self.input_size * self.chunk_length)

        self.tau = 1

    def setup(self, movement_segment: MovementSegment):
        self.input_rms.update(self.pack_invar(movement_segment.three_p))

    def encode(self, x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
        xx = self.input_rms.normalize(self.pack_invar(x))
        z = self.encoder_proj(xx.reshape(xx.shape[0], -1))
        z = z.reshape(z.shape[0], self.sentence_length, self.hidden_size)
        z = self.encoder(z.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.encoder_fc(z.reshape(z.shape[0], -1))
        z = z.reshape(z.shape[0], -1, self.sentence_length * self.vocab_size)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = self.decoder_fc(z)
        zz = zz.reshape(z.shape[0], -1, self.sentence_length, self.hidden_size) * 1
        # for conv1d we need to flatten and then reflatten
        zz = zz.reshape(-1, *zz.shape[2:])
        x = self.decoder(zz.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.decoder_deproj(x.reshape(x.shape[0], -1))
        x = x.reshape(x.shape[0], -1, self.chunk_length, self.input_size)
        x = self.unpack_invar(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor = None,
        n: int = 1,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        zz = z.reshape(z.shape[0], 1, self.sentence_length, self.vocab_size).repeat_interleave(n, 1)
        soft, hard = gumbel_softmax(zz, tau=self.tau)
        k = hard.argmax(-1)
        soft = soft.reshape(z.shape[0], -1, z.shape[-1])
        hard = hard.reshape(z.shape[0], -1, z.shape[-1])
        x_hat = self.decode(hard)
        return z, k, soft, hard, x_hat


class TransformerGSVAE(nn.Module, InvarMixin):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

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
    ):
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
        # self.effective_chunk_length = chunk_length
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
        # self.deproj = nn.Linear(self.hidden_size * self.sentence_length, self.input_size * self.chunk_length)
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

    def setup(self, my_3p: torch.Tensor):
        self.input_rms.update(self.pack_invar(my_3p[:, None]).reshape(-1, self.effective_input_size))

    def encode(self, x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
        xx = self.input_rms.normalize(self.pack_invar(x))
        xx = torch.clamp(xx, -10, 10)
        z = self.proj(xx)
        z = z.reshape(z.shape[0], -1, self.effective_chunk_length * 3, self.embed_size)

        pe = torch.zeros_like(z)
        idxs = torch.arange(z.shape[-2], device=z.device)
        div_term = torch.exp(torch.arange(0, z.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (z.shape[-1]))).to(z.device)
        pe[..., 0::2] = torch.sin(idxs[..., None] * div_term[None])
        pe[..., 1::2] = torch.cos(idxs[..., None] * div_term[None])
        z = z + pe

        z = self.encoder(z.reshape(-1, *z.shape[2:])).reshape(z.shape)
        z = self.encoder_fc(z.reshape(z.shape[0], z.shape[1], -1))
        z = z.reshape(z.shape[0], -1, self.sentence_length * self.vocab_size)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = self.decoder_fc(z)
        zz = zz.unflatten(-1, (self.effective_chunk_length * 3, self.embed_size))

        pe = torch.zeros_like(zz)
        idxs = torch.arange(zz.shape[-2], device=zz.device)
        div_term = torch.exp(torch.arange(0, zz.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (zz.shape[-1]))).to(zz.device)
        pe[..., 0::2] = torch.sin(idxs[..., None] * div_term[None])
        pe[..., 1::2] = torch.cos(idxs[..., None] * div_term[None])
        zz = zz + pe

        x = self.decoder(zz.flatten(0, -3)).reshape(zz.shape)
        x = self.deproj(x)
        x = x.unflatten(-2, (self.effective_chunk_length, 3)).flatten(-2, -1)
        x = self.unpack_invar(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor = None,
        n: int = 1,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        zz = z.repeat_interleave(n, 1).unflatten(-1, (self.sentence_length, self.vocab_size))
        soft, hard = gumbel_softmax(zz, tau=self.tau)
        k = hard.argmax(-1)
        soft = soft.flatten(-2, -1)
        hard = hard.flatten(-2, -1)
        x_hat = self.decode(hard)
        return zz, k, soft, hard, x_hat


class CondTransformerDenoiser(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        cond_size: int,
        hidden_size: int,
        embed_size: int,
        history_length: int,
        chunk_length: int,
        stride: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_size = input_size
        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.history_length = history_length
        self.chunk_length = chunk_length
        self.stride = stride
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = nn.SiLU()

        # Temporal embedding for denoising
        self.temb = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, embed_size))

        # MLP projector for signal to be denoised
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embed_size),
        )

        # Condition projectors: one for notes, one for bombs, one for obstacles, one for history
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embed_size),
        )

        # Deprojector mapping from transformer's next token logit to clean signal (ViT-like)
        self.deproj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, input_size),
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
        self.rms_input = RunningMeanStd(shape=(input_size,))
        self.rms_cond = RunningMeanStd(shape=(cond_size,))

    def cond_project(self, cond: torch.Tensor, typ: str):
        t_orig = cond[..., 0] * 1
        t = t_orig * 1
        t[torch.isnan(t)] = 0

        # Form x stuff
        cond = cond * 1
        cond[torch.isnan(cond)] = 0

        xx = proj(rms.normalize(cond)[..., 1:])

        # TTA encoding on nail embeddings
        pe = torch.zeros_like(xx)
        div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (xx.shape[-1]))).to(xx.device)
        pe[..., 0::2] = torch.sin(t[..., None] * div_term)
        pe[..., 1::2] = torch.cos(t[..., None] * div_term)
        xx += pe

        # Domain encoding, just use indices
        de = torch.zeros_like(xx)
        div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(10000.0) / (xx.shape[-1]))).to(xx.device)
        de[..., 0::2] = torch.sin(i * div_term)
        de[..., 1::2] = torch.cos(i * div_term)
        xx += de

        x_mask = torch.isnan(t_orig)
        x_mask = x_mask.reshape(x_mask.shape[0], -1)

    def forward(
        self,
        p: torch.Tensor,
        c: torch.Tensor,
        b: torch.Tensor,
        o: torch.Tensor,
        h: torch.Tensor,
        t: torch.Tensor,
    ):
        p = self.proj(p)
        # TODO: apply projection separately for each domain and apply TTA attention masking
        cc, c_mask = self.cond_project(c, "c")
        bb, b_mask = self.cond_project(b, "b")
        oo, o_mask = self.cond_project(o, "o")
        hh, h_mask = self.cond_project(h, "h")
        ww = torch.cat([cc, bb, oo], dim=1)

        tt = self.temb(t)
        xwt = torch.cat([p, ww, tt], dim=1)

        # Position encoding
        pe = torch.zeros_like(xwt)
        idxs = torch.arange(xwt.shape[1], device=xwt.device)
        div_term = torch.exp(torch.arange(0, xwt.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (xwt.shape[-1]))).to(xwt.device)
        pe[..., 0::2] = torch.sin(idxs[..., None] * div_term[None])
        pe[..., 1::2] = torch.cos(idxs[..., None] * div_term[None])
        xwt += pe

        # TODO: make attention mask, just yoink from GS-VAE code
        # TODO: use src and mask kwargs to receive attention mask
        xwt = self.encoder(xwt)
        p = self.deproj(xwt[:, : self.chunk_length])
        return p

    def embed_input_segment(self, notes: torch.Tensor, bombs: torch.Tensor, obstacles: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        embeddings = []
        masks = []
        for i, (x, rms, proj) in enumerate(
            zip(
                [notes, bombs, obstacles],
                [self.note_rms, self.bomb_rms, self.obstacle_rms],
                [self.note_proj, self.bomb_proj, self.obstacle_proj],
            )
        ):
            t_orig = x[..., 0] * 1
            t = t_orig * 1
            t[torch.isnan(t)] = 0

            # Form x stuff
            x = x * 1
            x[torch.isnan(x)] = 0

            xx = proj(rms.normalize(x)[..., 1:])

            # TTA encoding on nail embeddings
            pe = torch.zeros_like(xx)
            div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (xx.shape[-1]))).to(xx.device)
            pe[..., 0::2] = torch.sin(t[..., None] * div_term)
            pe[..., 1::2] = torch.cos(t[..., None] * div_term)
            xx += pe

            # Domain encoding, just use indices
            de = torch.zeros_like(xx)
            div_term = torch.exp(torch.arange(0, xx.shape[-1] // 2, 1).float() * (-np.log(10000.0) / (xx.shape[-1]))).to(xx.device)
            de[..., 0::2] = torch.sin(i * div_term)
            de[..., 1::2] = torch.cos(i * div_term)
            xx += de

            x_mask = torch.isnan(t_orig)
            x_mask = x_mask.reshape(x_mask.shape[0], -1)

            embeddings.append(xx)
            masks.append(x_mask)
        embeddings = torch.cat(embeddings, dim=2)
        masks = torch.cat(masks, dim=1)
        return embeddings, masks

    def embed_history(self, history: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = history[..., 0] * 1
        # history[torch.isnan(history)] = 0
        # ww = self.history_proj(self.history_rms.normalize(history))
        ww = self.history_proj(self.history_rms.normalize(history).reshape(history.shape[0], -1, 3, self.effective_history_size))

        ww = ww.reshape(ww.shape[0], -1, ww.shape[-1])

        pe = torch.zeros_like(ww)
        div_term = torch.exp(torch.arange(0, ww.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (ww.shape[-1]))).to(ww.device)
        pe[..., 0::2] = torch.sin(torch.arange(ww.shape[1], device=ww.device)[..., None] * div_term)
        pe[..., 1::2] = torch.cos(torch.arange(ww.shape[1], device=ww.device)[..., None] * div_term)
        ww += pe

        # TODO: Domain encoding can be done with nn.embedding...? probably not important
        # History domain encoding
        de = torch.zeros_like(ww)
        div_term = torch.exp(torch.arange(0, ww.shape[-1] // 2, 1).float() * (-np.log(10000.0) / (ww.shape[-1]))).to(ww.device)
        de[..., 0::2] = torch.sin(3 * div_term)
        de[..., 1::2] = torch.cos(3 * div_term)
        ww += de

        w_mask = torch.isnan(t_orig.reshape(t_orig.shape[0], -1)).repeat_interleave(3, -1)

        return ww, w_mask


class ConditionalEDM(nn.Module):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        p_size: int,  # dim of each individual component of 3p, usually p_size = 9
        sequence_length: int,  # 3p sequence, flattened. 3 * number of frames
        note_size: int,
        bomb_size: int,
        obstacle_size: int,
        history_size: int,
        hidden_size: int,
        embed_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.p_size = p_size
        self.sequence_length = sequence_length
        self.note_size = note_size
        self.bomb_size = bomb_size
        self.obstacle_size = obstacle_size
        self.history_size = history_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.time_size = 1
        self.rms_input = RunningMeanStd(shape=(input_size,))
        self.denoiser = CondTransformerDenoiser(
            self.input_size,
            self.cond_size,
            self.hidden_size,
            self.embed_size,
            self.history_length,
            self.chunk_length,
            self.stride,
            self.num_heads,
            self.num_layers,
            self.dropout,
        )
        self.mse_loss = nn.MSELoss()
        self.sigma_min = 0.002
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        # self.rolling_yes = True
        self.rolling_yes = False

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        Use forward diffusion to corrupt the clean signal `x`,
        and train the model to predict the corrupted signal.
        Eps model is aware of the amount of corruption involved,
        i.e. number of forward diffusion steps used and thus how much noise level there must be.
        """

        eps = torch.randn_like(x)

        # EDM: sigma approach
        log_sigma = (
            -1.2
            + torch.randn(
                (x.shape[0], x.shape[1], *(1 for _ in range(len(x.shape) - 3)), 1),
                device=x.device,
            )
            * 1.2
        )

        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)

        if self.rolling_yes:
            yes = torch.rand(x.shape[0]) < 0.4  # Make sure to change this for rolling diffusion...

            idxs = torch.arange(x.shape[1] + 1, device=x.device)
            sigmas = sigma[:, [0]] * idxs / x.shape[1]
            sigmas = sigmas[..., 1:]
            sigmas = torch.clamp(sigmas, self.sigma_min, self.sigma_max)
            sigma[yes] = sigmas.swapaxes(-2, -1)[yes]

        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = 1 / 4 * torch.log(sigma)
        lammy = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2

        x = self.rms_input.normalize(x)
        noised_x = x + eps * sigma
        eps_tar = x
        eps_out = c_skip * noised_x + c_out * self.denoise(c_in * noised_x, w, c_noise).view(x.shape)
        return eps_out, eps_tar, lammy

    def denoise(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor):
        return self.denoiser(x, w, t)

    def step(self, x_t: torch.Tensor, w: torch.Tensor, tt: int):
        device = x_t.device
        t = (
            torch.ones(
                (
                    x_t.shape[0],
                    x_t.shape[1],
                    *(1 for _ in range(len(x_t.shape) - 3)),
                    1,
                ),
                device=device,
            )
            * tt
        )
        sigma = torch.clamp(t, self.sigma_min, self.sigma_max)
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = 1 / 4 * torch.log(sigma)

        eps_out = c_skip * x_t + c_out * self.denoise(c_in * x_t, w, c_noise).view(x_t.shape)

        return eps_out

    def setup(self, x: torch.Tensor):
        self.rms_input.update(x)
        self.denoiser.rms_input.update(x)
        self.denoiser.rms_cond.update(x)
