from datasets import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train.my_tokenizers import RVQTokenizer, my_kmeans
from beaty_common.train_utils import ThroughDataset
import torch.nn.functional as F


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape=(), *args, **kwargs):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(
            torch.zeros(shape, dtype=torch.float), requires_grad=False
        )
        self.var = nn.Parameter(
            torch.ones(shape, dtype=torch.float), requires_grad=False
        )
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip(
            (arr - self.mean) / torch.sqrt(self.var + self.epsilon), -5, 5
        )

    def unnormalize(self, arr: torch.Tensor) -> torch.Tensor:
        return arr * torch.sqrt(self.var + self.epsilon) + self.mean


class MLP(nn.Module):

    def __init__(
        self, nail_size: int, fourier_size: int, hidden_size: int, hammer_size: int
    ):
        super().__init__()
        self.rms = RunningMeanStd(shape=(nail_size,))
        self.emb = GaussianFourierFeatures(nail_size, fourier_size, 1e0)
        # self.emb = nn.Identity()
        self.layers = nn.Sequential(
            nn.Linear(fourier_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fourier_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fourier_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fourier_size + hidden_size, hammer_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if train_yes:
        #     self.rms.update(x)
        # x = self.rms.normalize(x)
        x = self.emb.forward(x)
        z = x * 1
        for i, op in enumerate(self.layers):
            if i == 0 or isinstance(op, nn.LeakyReLU):
                z = op.forward(z)
            else:
                z = op.forward(torch.cat([x, z], dim=-1))
        y = z
        return y


class MLPSkipper(nn.Module):

    def __init__(self, nail_size: int, hidden_size: int, hammer_size: int):
        super().__init__()
        self.lns = nn.ModuleList(
            [
                nn.LayerNorm(nail_size),
                nn.LayerNorm(hidden_size),
                nn.LayerNorm(hidden_size),
            ]
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(nail_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                ),
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                ),
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(hidden_size, hammer_size),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lns[0].forward(x)
        x = self.layers[0].forward(x)

        x = self.lns[1].forward(x)
        y = self.layers[1].forward(x)
        x = x + y

        x = self.lns[2].forward(x)
        x = self.layers[2].forward(x)
        return x


# class ConvSkipper(nn.Module):
#
#     def __new__(cls, *args, **kwargs):
#         instance = super().__new__(cls)
#         instance.args = args
#         instance.kwargs = kwargs
#         return instance
#
#     def __init__(
#         self,
#         input_size: int,
#         channel_size: int,
#         kernel_size: int,
#         latent_size: int,
#         vocab_size: int,
#         sentence_length: int,
#     ):
#         super().__init__()
#         self.lns = nn.ModuleList(
#             [
#                 nn.LayerNorm(input_size),
#                 nn.LayerNorm(hidden_size),
#                 nn.LayerNorm(hidden_size),
#             ]
#         )
#         self.layers = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(nail_size, hidden_size),
#                     nn.ReLU(),
#                     nn.Linear(hidden_size, hidden_size),
#                 ),
#                 nn.Sequential(
#                     nn.ReLU(),
#                     nn.Linear(hidden_size, hidden_size),
#                 ),
#                 nn.Sequential(
#                     nn.ReLU(),
#                     nn.Linear(hidden_size, hammer_size),
#                 ),
#             ]
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.lns[0].forward(x)
#         x = self.layers[0].forward(x)
#
#         x = self.lns[1].forward(x)
#         y = self.layers[1].forward(x)
#         x = x + y
#
#         x = self.lns[2].forward(x)
#         x = self.layers[2].forward(x)
#         return x


class MLPAutoregressive(nn.Module):

    def __init__(
        self,
        nail_size: int,
        cond_size: int,
        # fourier_size: int,
        hidden_size: int,
        hammer_size: int,
        # fourier_sigma: float,
        history_len: int,
    ):
        super().__init__()
        self.rms_nail = RunningMeanStd(shape=(nail_size,))
        self.rms_cond = RunningMeanStd(shape=(cond_size,))
        self.cond_size = cond_size
        self.emb_size = 2
        self.nail_embs = nn.ModuleList(
            [nn.Embedding(16, self.emb_size) for _ in range(4)]
        )
        # self.emb_nail = GaussianFourierFeatures(
        #     nail_size, fourier_size // 2, fourier_sigma
        # )
        # self.emb_cond = GaussianFourierFeatures(
        #     cond_size, fourier_size // 2, fourier_sigma
        # )
        # self.emb = nn.Identity()
        # in_size = (
        #     self.emb_size * (nail_size - 1) * 20 + cond_size * 3 * history_len
        # )  # just to disregard Fourier stuff
        in_size = (
            nail_size * 20 + cond_size * 3 * history_len
        )  # just to disregard Fourier stuff

        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(fourier_size + hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(fourier_size + hidden_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Linear(in_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_size + hidden_size, hammer_size),
        )

    def setup(self, nail, cond, hamm, device):
        self.to(device)
        self.rms_nail = self.rms_nail.to(device)
        self.rms_cond = self.rms_cond.to(device)
        self.rms_nail.update(nail)
        self.rms_cond.update(cond)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor = None,
        topk: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        w_last_orig = w[:, -1] * 1
        # if train_yes:
        #     self.rms.update(x)
        # t = x[..., 0] * 1
        # t[torch.isnan(t)] = 0
        # x_long = x[..., 1:].to(dtype=torch.long) + 1
        # x_long[torch.isnan(x_long)] = 0  # 0 marks the unknown
        # t_repeated = t[..., None].repeat_interleave(x_long.shape[-1], -1)
        # t_repeated.reshape(t_repeated.shape[0], -1)
        #
        # xx = []
        # for i, emb in enumerate(self.nail_embs):
        #     xx.append(emb(x_long[..., i]))
        # xx = torch.stack(xx, dim=2)
        #
        # # TTA encoding on nail embeddings
        # pe = torch.zeros_like(xx)
        # div_term = torch.exp(
        #     torch.arange(0, self.emb_size // 2, 1).float()
        #     * (-np.log(10000.0) / (self.emb_size))
        # ).to(xx.device)
        # # pe[..., 0::2] = torch.sin(t[..., None] * div_term)
        # # pe[..., 1::2] = torch.cos(t[..., None] * div_term)
        # pe[..., 0::2] = torch.sin(t_repeated[..., None] * div_term)
        # pe[..., 1::2] = torch.cos(t_repeated[..., None] * div_term)
        # xx += pe
        xx = self.rms_nail.normalize(x)
        xx[torch.isnan(xx)] = 0

        w = self.rms_cond.normalize(w).reshape(w.shape[0], -1)
        w[torch.isnan(w)] = 0
        z = torch.cat([xx.reshape(xx.shape[0], -1), w], dim=-1)
        for i, op in enumerate(self.layers):
            if i == 0 or isinstance(op, nn.LeakyReLU):
                z = op.forward(z)
            else:
                z = op.forward(torch.cat([xx.reshape(xx.shape[0], -1), w, z], dim=-1))
        y = z
        y = y.reshape(y.shape[0], -1, 6)
        # with torch.no_grad():
        #     y += w_last_orig

        return y, None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(1000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TransformerContinuous(nn.Module):

    def __init__(
        self,
        nail_size: int,
        cond_size: int,
        hidden_size: int,
        hammer_size: int,
        num_heads: int,
        num_layers: int,
        ff_size: int,
        dropout: float,
        vocab_size: int,
        n_parts: int,
        history_len: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hammer_size = hammer_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_size = ff_size
        self.dropout = dropout
        self.nail_proj = nn.Linear(nail_size - 1, self.hidden_size, bias=True)
        self.n_parts = n_parts
        self.history_len = history_len

        self.nail_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(nail_size - 1, self.hidden_size, bias=True),
        )
        self.cond_proj = nn.Linear(cond_size, self.hidden_size, bias=True)

        self.activation = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.hamm_deproj = nn.Linear(self.hidden_size, self.hammer_size, bias=True)

        self.rms_nail = RunningMeanStd(shape=(nail_size,))
        self.rms_cond = RunningMeanStd(shape=(cond_size,))

    def setup(self, nail, cond, hamm, device):
        self.to(device)
        self.rms_nail = self.rms_nail.to(device)
        self.rms_cond = self.rms_cond.to(device)
        self.rms_nail.update(nail)
        self.rms_cond.update(cond)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor or None = None,
        y: torch.Tensor or None = None,
        topk: int = 1,
        temperature: float = 1,
    ) -> torch.Tensor:
        w_last_orig = w[:, -1] * 1
        n_w_toks = w.shape[1] * w.shape[2]
        # total_paragraph_length = (x.shape[1] * 4) + n_w_toks + self.paragraph_length
        total_paragraph_length = (x.shape[1] * 1) + n_w_toks + 1
        t_orig = x[..., 0] * 1
        t = t_orig * 1
        t[torch.isnan(t)] = 0

        xx, x_mask = self.embed_x(x)
        ww, w_toks, w_mask = self.embed_w(w)

        tgt = torch.cat([xx, ww], dim=1)

        tgt_mask = ~torch.tril(
            torch.ones(
                total_paragraph_length - 1,
                total_paragraph_length - 1,
                dtype=torch.bool,
                device=x.device,
            ),
            0,
        )
        # tgt_mask = torch.where(tgt_mask, -torch.inf, 0)
        tgt_mask = tgt_mask[None].repeat_interleave(x.shape[0], 0)
        lo = 0
        hi = xx.shape[1]
        tgt_mask[:, :, lo:hi] = x_mask[:, None]
        # disable overwriting by whether w was nan or not; sentences marking "unknowns" are good things to learn along
        # lo = hi
        # hi = lo + ww.shape[1]
        # tgt_mask[:, :, lo:hi] = torch.logical_or(tgt_mask[:, :, lo:hi], w_mask[:, None])  # apply causality if w is given
        tgt_mask = tgt_mask.repeat_interleave(self.num_heads, 0)

        pe = torch.zeros(
            x.shape[0],
            total_paragraph_length,
            self.hidden_size,
            device=x.device,
        )
        div_term = torch.exp(
            torch.arange(0, self.hidden_size // 2, 1).float()
            * (-np.log(1000.0) / self.hidden_size)
        ).to(xx.device)
        pe[..., 0::2] = torch.sin(
            torch.arange(pe.shape[1], device=xx.device)[:, None] * div_term
        )
        pe[..., 1::2] = torch.cos(
            torch.arange(pe.shape[1], device=xx.device)[:, None] * div_term
        )
        tgt[:, xx.shape[1] :] += pe[:, xx.shape[1] : -1]
        if y is not None:
            with torch.no_grad():
                y_resid = y - w_last_orig
            # apply position embedding to w and y
            decoder_out = self.transformer_decoder(
                src=tgt,
                mask=tgt_mask,
                # is_causal=True,
            )
            y_resid_hat = self.hamm_deproj(decoder_out[:, -1])
            y_resid_hat = y_resid_hat.reshape(y_resid_hat.shape[0], -1, 6)
            loss = nn.functional.mse_loss(y_resid_hat, y_resid)
            y = None
        else:
            decoder_out = self.transformer_decoder(
                src=tgt,
                mask=tgt_mask,
            )
            y_resid_hat = self.hamm_deproj(decoder_out[:, -1])
            y_resid_hat = y_resid_hat.reshape(y_resid_hat.shape[0], -1, 6)
            with torch.no_grad():
                y = y_resid_hat + w_last_orig
            loss = None

        return y, loss

    def compute_loss(
        self, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, w, y)[-1]

    def embed_x(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = x[..., 0] * 1
        t = t_orig * 1
        t[torch.isnan(t)] = 0

        # Form x stuff
        x = x * 1
        x[torch.isnan(x)] = 0

        xx = self.nail_proj(self.rms_nail.normalize(x)[..., 1:])
        xx = xx.reshape(xx.shape[0], -1, xx.shape[-1])

        # TTA encoding on nail embeddings
        pe = torch.zeros_like(xx)
        div_term = torch.exp(
            torch.arange(0, xx.shape[-1] // 2, 1).float()
            * (-np.log(1000.0) / (xx.shape[-1]))
        ).to(xx.device)
        pe[..., 0::2] = torch.sin(t[..., None] * div_term)
        pe[..., 1::2] = torch.cos(t[..., None] * div_term)
        xx += pe
        x_mask = torch.isnan(t_orig)
        x_mask = x_mask.reshape(x_mask.shape[0], -1)
        return xx, x_mask

    def embed_w(self, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = w[..., 0] * 1
        w[torch.isnan(w)] = 0
        w = self.rms_cond.normalize(w).reshape(w.shape[0], -1, w.shape[-1])
        ww = self.cond_proj(w)

        ww = ww.reshape(ww.shape[0], -1, ww.shape[-1])
        w_mask = torch.where(
            torch.isnan(t_orig.reshape(t_orig.shape[0], -1)),
            -torch.inf,
            0,
        )

        return ww, None, w_mask

    def get_tgt_mask(self, xx, ww):
        pass


class TransformerDiscrete(nn.Module):

    def __init__(
        self,
        nail_size: int,
        cond_size: int,
        hidden_size: int,
        hammer_size: int,
        num_heads: int,
        num_layers: int,
        ff_size: int,
        dropout: float,
        vocab_size: int,
        sentence_length: int,
        n_parts: int,
        history_len: int,
        pred_len: int,
        args,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hammer_size = hammer_size
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_size = ff_size
        self.dropout = dropout
        self.pred_len = pred_len
        self.n_parts = n_parts
        self.history_len = history_len
        self.args = args

        emb_size = 1024
        self.nail_emb = nn.Embedding(vocab_size, emb_size)

        if self.args.tokenize_notes_yes:
            self.nail_proj = nn.Linear(emb_size, self.hidden_size, bias=True)
        else:
            self.nail_proj = nn.Linear(nail_size - 1, self.hidden_size, bias=True)

        self.cond_emb = nn.Embedding(vocab_size, emb_size)
        if self.args.tokenize_history_yes:
            self.cond_proj = nn.Linear(emb_size, self.hidden_size, bias=True)
        else:
            self.cond_proj = nn.Linear(cond_size, self.hidden_size, bias=True)
        # self.cond_proj = nn.Linear(cond_size, self.hidden_size, bias=True)

        self.hamm_emb = nn.Embedding(vocab_size, self.hidden_size)
        self.activation = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.hamm_deproj = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        self.part_size = self.hammer_size // self.n_parts
        self.paragraph_length = self.sentence_length * self.n_parts * self.pred_len
        self.hamm_tokenizer = RVQTokenizer(
            self.part_size, self.sentence_length, self.vocab_size, 50, 1, False
        )
        self.cond_tokenizer = RVQTokenizer(
            self.part_size, self.sentence_length, self.vocab_size, 50, 1, False
        )
        self.ln = nn.LayerNorm(self.hidden_size)

        self.rms_nail = RunningMeanStd(shape=(nail_size,))
        self.rms_cond = RunningMeanStd(shape=(cond_size,))
        self.rms_hamm = RunningMeanStd(shape=(self.part_size,))
        self.rms_xyz = RunningMeanStd(shape=(3,))
        self.rms_expm = RunningMeanStd(shape=(3,))
        self.xyz_tokenizer = RVQTokenizer(3, 1, self.vocab_size, 50, 1, False)
        self.expm_tokenizer = RVQTokenizer(3, 1, self.vocab_size, 50, 1, False)

        self.xyzexpm_rmses = []
        self.xyzexpm_tokenizers = []
        for i in range(self.n_parts):
            xyzexpm_rms = RunningMeanStd(
                shape=(self.args.tok_horizon * self.part_size,)
            )
            xyzexpm_tokenizer = RVQTokenizer(
                self.args.tok_horizon * self.part_size,
                self.sentence_length,
                self.vocab_size,
                50,
                1,
                False,
            )
            self.xyzexpm_rmses.append(xyzexpm_rms)
            self.xyzexpm_tokenizers.append(xyzexpm_tokenizer)

    def setup(self, nail, cond, hamm, device):
        self.to(device)

        self.rms_nail = self.rms_nail.to(device)
        self.rms_cond = self.rms_cond.to(device)
        for i in range(self.n_parts):
            self.xyzexpm_rmses[i].to(device)

        self.rms_nail.update(nail)
        self.rms_cond.update(cond)
        hamm = hamm.reshape(hamm.shape[0], hamm.shape[1], self.n_parts, self.part_size)
        for i in range(self.n_parts):
            self.xyzexpm_rmses[i].update(hamm[:, :, i].reshape(hamm.shape[0], -1))

        # self.vqvae = ConvVQVAE(16, 3, self.vocab_size)

        # random_idxs = torch.randint(0, hamm.shape[0], (int(1e5),))
        # random_hamm = hamm[random_idxs]
        # for i in range(self.n_parts):
        #     self.xyzexpm_tokenizers[i].build_codebook(
        #         self.xyzexpm_rmses[i].normalize(
        #             random_hamm[:, :, i].reshape(random_hamm.shape[0], -1)
        #         ),
        #         batch_size=int(1e5),
        #         device=device,
        #     )
        #
        # # Compute quantization error
        # random_idxs = torch.randint(0, hamm.shape[0], (int(1e5),))
        # random_hamm = hamm[random_idxs]
        # q_errs = []
        # for i in range(self.n_parts):
        #     random_hamm_reshaped = random_hamm[:, :, i].reshape(
        #         random_hamm.shape[0], -1
        #     )
        #     encoded, quantized = self.xyzexpm_tokenizers[i].encode(
        #         self.xyzexpm_rmses[i].normalize(random_hamm_reshaped), device=device
        #     )
        #     q_err = torch.nn.functional.mse_loss(
        #         self.xyzexpm_rmses[i].unnormalize(quantized), random_hamm_reshaped
        #     ).item()
        #     q_errs.append(q_err)
        # print(f"Quantization errors: {q_errs}")
        # self.quantization_error = np.mean(q_errs).item()

    def setup2(self, hamm, device):
        self.vqvae = ConvVQVAE(hamm.shape[-1], 768, 3, 32, self.vocab_size).to(device)
        hamm_normalized = self.rms_hamm.normalize(hamm)
        self.vqvae.setup(hamm_normalized)

        # Do some pretraining here...
        dataset = ThroughDataset(hamm_normalized, hamm)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        optimizer = torch.optim.RAdam(self.vqvae.parameters(), lr=3e-4)
        n_epochs = 1000
        peak_lr = 3e-4
        pbar = tqdm(total=n_epochs)
        for epoch in range(n_epochs):
            lr = peak_lr * (1 - epoch / n_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            for x_in, x_gt in dataloader:
                z_t, k, z_q, x_hat = self.vqvae.forward(x_in)
                recon_loss = F.mse_loss(x_hat, x_gt)
                code_loss = F.mse_loss(z_t, z_q.detach())
                commitment_loss = F.mse_loss(z_q, z_t.detach())

                optimizer.zero_grad()
                loss = recon_loss + 1e0 * 1.0 * code_loss + 1e0 * 0.25 * commitment_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(), 1.0)
                optimizer.step()

            pbar.update(1)
            pbar.set_postfix({"loss": loss.item(), "epoch": epoch})
        pbar.close()

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor or None = None,
        y: torch.Tensor or None = None,
        topk: int = 1,
        temperature: float = 1,
    ) -> torch.Tensor:
        t_orig = x[..., 0] * 1
        t = t_orig * 1
        t[torch.isnan(t)] = 0

        xx, x_mask = self.embed_x(x)
        ww, w_mask = self.embed_w(w)
        self.paragraph_length = self.sentence_length * self.n_parts * self.pred_len
        total_paragraph_length = xx.shape[1] + ww.shape[1] + self.paragraph_length

        bos = torch.zeros_like(xx[:, [0]])

        tgt = torch.cat([bos, xx, ww], dim=1)

        tgt_mask = ~torch.tril(
            torch.ones(
                total_paragraph_length,
                total_paragraph_length,
                dtype=torch.bool,
                device=x.device,
            ),
            0,
        )
        tgt_mask = tgt_mask[None].repeat_interleave(x.shape[0], 0)
        lo = 1
        hi = xx.shape[1] + 1
        tgt_mask[:, :, lo:hi] = x_mask[:, None]
        tgt_mask = tgt_mask.repeat_interleave(self.num_heads, 0)

        ype = torch.zeros(
            (x.shape[0], self.paragraph_length, self.hidden_size), device=x.device
        )
        div_term = torch.exp(
            torch.arange(0, self.hidden_size // 2, 1).float()
            * (-np.log(1000.0) / (self.hidden_size))
        ).to(ype.device)
        ype[..., 0::2] = torch.sin(
            torch.arange(ype.shape[1], device=ype.device)[..., None] * div_term
        )
        ype[..., 1::2] = torch.cos(
            torch.arange(ype.shape[1], device=ype.device)[..., None] * div_term
        )
        if y is not None:
            with torch.no_grad():
                tokenizer_in = y.reshape(
                    y.shape[0], y.shape[1], self.n_parts, self.part_size
                )

                y_toks = []
                for i in range(self.n_parts):
                    normalized = self.xyzexpm_rmses[i].normalize(
                        tokenizer_in[:, :, i].reshape(tokenizer_in.shape[0], -1)
                    )
                    encoded, _ = self.xyzexpm_tokenizers[i].encode(
                        normalized, device=y.device
                    )
                    y_toks.append(encoded)
                y_toks = torch.stack(y_toks, dim=-1).reshape(y.shape[0], -1)
            tar_idxs = y_toks * 1

            yy = self.hamm_emb(y_toks)
            # yy = self.hamm_proj(yy)

            yy = yy.reshape(y.shape[0], -1, yy.shape[-1])
            yy += ype[:, : yy.shape[1]]
            tgt = torch.cat([tgt, yy[:, :-1]], dim=1)
            decoder_out = self.transformer_decoder(
                src=tgt,
                mask=tgt_mask,
                # is_causal=True,
            )
            decoder_out = self.ln(decoder_out)
            logits = self.hamm_deproj(decoder_out[:, -tar_idxs.shape[-1] :])
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), tar_idxs.view(-1)
            )
            y = None
        else:
            paragraphs = []
            t = 0
            tt = tgt.shape[1]
            for i in range(self.n_parts):
                for j in range(self.sentence_length):
                    my_mask = tgt_mask[
                        :,
                        : tt + t,
                        : tt + t,
                    ]
                    decoder_out = self.transformer_decoder(
                        src=tgt,
                        mask=my_mask,
                        # is_causal=True,
                    )
                    decoder_out = self.ln(decoder_out)
                    next_logits = self.hamm_deproj(decoder_out[:, -1])
                    next_logits /= temperature
                    v, _ = torch.topk(next_logits, min(topk, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -torch.inf
                    # Deterministic
                    # next_tok = next_logits.argmax(-1)
                    # Stochastic
                    next_tok = torch.multinomial(next_logits.softmax(-1), 1)
                    paragraphs.append(next_tok)
                    # next_emb = self.hamm_embs[i](next_tok)
                    # next_emb = self.hamm_proj(self.hamm_emb(next_tok))
                    next_emb = self.hamm_emb(next_tok)
                    tgt = torch.cat([tgt, next_emb + ype[:, [t]]], dim=1)
                    t += 1
            paragraphs = torch.cat(paragraphs, dim=1)
            paragraphs = paragraphs.reshape(
                paragraphs.shape[0], 1, self.sentence_length, self.n_parts
            )
            # TRICK: permute so we do causal sampling wrt low-frequency info first and then high-frequency info next
            paragraphs = paragraphs.permute(0, 1, 3, 2)
            decoded_trajs = []
            for i in range(self.n_parts):
                decoded = self.xyzexpm_tokenizers[i].decode(
                    paragraphs[:, -1, i], device=paragraphs.device
                )
                unnormalized = self.xyzexpm_rmses[i].unnormalize(decoded)
                reshaped = unnormalized.reshape(
                    unnormalized.shape[0], -1, self.part_size
                )
                decoded_trajs.append(reshaped)
            y = torch.stack(decoded_trajs, dim=2)
            y = y.reshape(y.shape[0], -1, 3, 6)

            loss = None

        return y, loss

    def compute_loss(
        self, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, w, y)[-1]

    def embed_x(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = x[..., 0] * 1
        t = t_orig * 1
        t[torch.isnan(t)] = 0

        # Form x stuff
        x = x * 1
        x[torch.isnan(x)] = 0

        if self.args.tokenize_notes_yes:
            x_long = x[..., 1:].to(dtype=torch.long)
            x_long += torch.arange(x_long.shape[-1], device=x.device) * 10
            t_orig_repeated = t_orig[..., None].repeat_interleave(x_long.shape[-1], -1)
            t_orig_repeated.reshape(t_orig_repeated.shape[0], -1)
            t_repeated = t[..., None].repeat_interleave(x_long.shape[-1], -1)
            t_repeated = t_repeated.reshape(t_repeated.shape[0], -1)
            xx = self.nail_proj(self.nail_emb(x_long))
        else:
            t_orig_repeated = t_orig
            t_repeated = t
            xx = self.nail_proj(self.rms_nail.normalize(x)[..., 1:])
        xx = xx.reshape(xx.shape[0], -1, xx.shape[-1])

        # TTA encoding on nail embeddings
        pe = torch.zeros_like(xx)
        div_term = torch.exp(
            torch.arange(0, xx.shape[-1] // 2, 1).float()
            * (-np.log(1000.0) / (xx.shape[-1]))
        ).to(xx.device)
        # pe[..., 0::2] = torch.sin(t[..., None] * div_term)
        # pe[..., 1::2] = torch.cos(t[..., None] * div_term)
        pe[..., 0::2] = torch.sin(t_repeated[..., None] * div_term)
        pe[..., 1::2] = torch.cos(t_repeated[..., None] * div_term)
        xx += pe
        # xx = self.nail_proj(xx)

        # x_mask = torch.isnan(t_orig.reshape(t_orig.shape[0], -1))
        # x_mask = (torch.cos(t_orig * np.pi / 2) + 1) * 0.5
        # x_mask = torch.arctanh(-t_orig / 2)
        # x_mask[torch.isnan(x_mask)] = -torch.inf
        # x_mask = torch.where(
        #     torch.isnan(t_orig), -torch.inf, torch.arctanh(-t_orig / 2)
        # )
        # x_mask = torch.isnan(t_orig)
        x_mask = torch.isnan(t_orig_repeated)
        x_mask = x_mask.reshape(x_mask.shape[0], -1)
        return xx, x_mask

    def embed_w(self, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = w[..., 0] * 1
        w[torch.isnan(w)] = 0
        # w = self.rms_cond.normalize(w).reshape(w.shape[0], -1, w.shape[-1])
        xyz = self.rms_xyz.normalize(w[..., :3])
        expm = self.rms_expm.normalize(w[..., 3:])

        # Tokenizing history
        if self.args.tokenize_history_yes:
            xyz_toks, _ = self.xyz_tokenizer.encode(xyz.reshape(-1, xyz.shape[-1]))
            xyz_toks = xyz_toks.reshape(xyz.shape[0], -1, xyz_toks.shape[-1])
            expm_toks, _ = self.expm_tokenizer.encode(expm.reshape(-1, expm.shape[-1]))
            expm_toks = expm_toks.reshape(expm.shape[0], -1, expm_toks.shape[-1])
            w_toks = torch.cat([xyz_toks, expm_toks], dim=-1)
            ww = self.cond_emb(w_toks)
            ww = self.cond_proj(ww)
        else:
            # Alternative: linear-projecting history
            xyzexpm = torch.cat([xyz, expm], dim=-1)
            ww = self.cond_proj(xyzexpm)

        ww = ww.reshape(ww.shape[0], -1, ww.shape[-1])

        pe = torch.zeros_like(ww)
        div_term = torch.exp(
            torch.arange(0, ww.shape[-1] // 2, 1).float()
            * (-np.log(1000.0) / (ww.shape[-1]))
        ).to(ww.device)
        pe[..., 0::2] = torch.sin(
            torch.arange(ww.shape[1], device=ww.device)[..., None] * div_term
        )
        pe[..., 1::2] = torch.cos(
            torch.arange(ww.shape[1], device=ww.device)[..., None] * div_term
        )
        ww += pe
        # w_mask = ~torch.isnan(t_orig.reshape(t_orig.shape[0], -1))
        w_mask = torch.where(
            torch.isnan(t_orig.reshape(t_orig.shape[0], -1)),
            -torch.inf,
            0,
        )
        w_mask = w_mask.repeat_interleave(self.sentence_length, 1)

        # converting to float
        # w_mask = w_mask.to(dtype=torch.float)

        return ww, w_mask

    def get_tgt_mask(self, xx, ww):
        pass


class GaussianFourierFeatures(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, sigma: float):
        """
        Assume output_dim is divisible by 2
        """
        super().__init__()
        self.b = nn.Parameter(
            torch.randn(input_dim, output_dim // 2) * sigma, requires_grad=False
        )
        self.output_dim = output_dim

    def forward(self, v: torch.tensor) -> torch.tensor:
        """
        input dims is (batch_size x input_dim)
        output dims is (batch_size x output_dim)
        """
        n_examples = v.shape[0]
        x = v @ self.b  # (batch_size x 2 x output_dim)
        x = torch.stack(
            [torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * x)], dim=1
        )


class MlpRes(nn.Module):

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
        output_size: int,
        n_mlps: int,
    ):
        super().__init__()
        self.mlp1s, self.mlp2s = [
            nn.ModuleList(
                [nn.Linear(input_size + cond_size, hidden_size)]
                + [
                    nn.Linear(hidden_size + cond_size, hidden_size)
                    for _ in range(n_mlps)
                ]
                + [nn.Linear(hidden_size + cond_size, output_size)]
            )
            for i in range(2)
        ]
        self.activation = nn.LeakyReLU()
        self.lns = nn.ModuleList(
            [nn.LayerNorm(input_size + cond_size)]
            + [nn.LayerNorm(hidden_size + cond_size) for _ in range(n_mlps)]
            + [nn.LayerNorm(hidden_size + cond_size)]
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xx = x * 1
        for i, (mlp1, mlp2, ln) in enumerate(zip(self.mlp1s, self.mlp2s, self.lns)):
            x = torch.cat([x, w], dim=-1)
            x = ln(x)
            x = mlp1(x)
            x = self.activation(x)
            x = torch.cat([x, w], dim=-1)
            x = xx + mlp2(x)
            xx = x * 1
            if i < len(self.mlp1s) - 1:
                x = self.activation(x)
        return x


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
        segment_length: int,
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
        self.lns = nn.ModuleList(
            [nn.LayerNorm([in_channels, segment_length])]
            + [nn.LayerNorm([out_channels, segment_length]) for _ in range(n_convs)]
            + [nn.LayerNorm([out_channels, segment_length])]
        )

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


class MlpMixin(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        vocab_size: int,
        sentence_length: int,
        n_mlps: int,
        n_blocks: int,
    ):
        super().__init__()
        self.latent_activation = nn.Identity()
        self.input_rms = RunningMeanStd(shape=(input_size,))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.latent_size = latent_size
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_size + hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_size + hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_size + hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_size + hidden_size, latent_size),
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_size + hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_size + hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_size + hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_size + hidden_size, input_size),
        # )
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def setup(self, x: torch.Tensor):
        self.input_rms.update(x)

    def setup2(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.encode(x)
            # RVQ initialization
            quantized = torch.zeros_like(z)
            for i in range(self.sentence_length):
                means, assignments = my_kmeans(
                    z - quantized, self.vocab_size, 50, None, False
                )
                self.codebooks[i].data = means
                quantized += self.codebooks[i][assignments]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.input_rms.normalize(x)
        # z = xx * 1
        # for j, op in enumerate(self.encoder):
        #     if isinstance(op, MlpRes):
        #         z = op(z, xx)
        #     else:
        #         if isinstance(op, nn.Linear):
        #             if j > 0:
        #                 z = torch.cat([xx, z], dim=-1)
        #         z = op(z)
        z = self.encoder(xx)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = z * 1
        # x = zz * 1
        # for j, op in enumerate(self.decoder):
        #     if isinstance(op, MlpRes):
        #         x = op(x, zz)
        #     else:
        #         if isinstance(op, nn.Linear):
        #             if j > 1:
        #                 x = torch.cat([zz, x], dim=-1)
        #         x = op(x)
        x = self.decoder(zz)
        return x


class MlpVQVAE(MlpMixin):

    def forward(
        self,
        x: torch.Tensor,
        dropout_mask: torch.Tensor = None,
        quant_yes: bool = False,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x.reshape(x.shape[0], -1))
        quantized = torch.zeros_like(z)
        k = []

        if not quant_yes:
            # Sanity check
            k = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=torch.long)
            z_q = z
        else:
            for i in range(self.sentence_length):
                with torch.no_grad():
                    deltas = (z - quantized)[:, None] - self.codebooks[i][None]
                    dists = torch.mean(deltas**2, dim=-1)
                    words = torch.argmin(dists, dim=1)
                    k.append(words)
                if dropout_mask is None:
                    quantized += self.codebooks[i][words]
                else:
                    quantized += self.codebooks[i][words] * dropout_mask[i]
            # Straight-through estimator
            z_q = z + (quantized - z.detach())
            k = torch.stack(k, dim=-1)
        x_hat = self.decode(z_q).reshape(x.shape)
        return z, k, z_q, x_hat


class MlpFSQVAE(MlpMixin):

    def forward(
        self,
        x: torch.Tensor,
        dropout_mask: torch.Tensor = None,
        quant_yes: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x)
        tanh_z = torch.tanh(z)
        with torch.no_grad():
            lim = 1 - 1e-3
            tanh_z = torch.clip(tanh_z, -lim, lim)
        k = torch.round(tanh_z * (self.vocab_size // 2)) + (self.vocab_size // 2)
        z_q = torch.arctanh((k - (self.vocab_size // 2)) / (self.vocab_size // 2))
        x_hat = self.decode(z_q)
        return z, k, z_q, x_hat


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


class GSVAEMixin(nn.Module):
    def __init__(self, sentence_length: int, vocab_size: int):
        super().__init__()
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.tau = 1

    def encode(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        zz = z.reshape(z.shape[0], 1, self.sentence_length, self.vocab_size)
        soft, hard = gumbel_softmax(zz, tau=self.tau)
        k = hard.argmax(-1)
        soft = soft.reshape(z.shape)
        hard = hard.reshape(z.shape)
        x_hat = self.decode(hard)
        return z, k, hard, x_hat


class MlpGSVAE(GSVAEMixin):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        vocab_size: int,
        sentence_length: int,
    ):
        super().__init__(sentence_length, vocab_size)
        self.latent_activation = nn.Identity()
        self.input_rms = RunningMeanStd(shape=(input_size,))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.latent_size = self.sentence_length * self.vocab_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.input_size + self.hidden_size, self.latent_size),
        )

        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.latent_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size + self.hidden_size, self.input_size),
        )

    def setup(self, x: torch.Tensor):
        self.input_rms.update(x)

    def encode(self, x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
        xx = self.input_rms.normalize(x)
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
                if j > 1:
                    x = torch.cat([zz, x], dim=-1)
            x = op(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        zz = z.reshape(z.shape[0], 1, self.sentence_length, self.vocab_size)
        soft, hard = gumbel_softmax(zz, tau=self.tau)
        k = hard.argmax(-1)
        soft = soft.reshape(z.shape)
        hard = hard.reshape(z.shape)
        x_hat = self.decode(soft)
        return z, k, hard, x_hat


class CondMlpGSVAE(MlpMixin):

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
        latent_size: int,
        vocab_size: int,
        sentence_length: int,
        n_mlps: int,
        n_blocks: int,
    ):
        super().__init__(
            input_size,
            hidden_size,
            latent_size,
            vocab_size,
            sentence_length,
            n_mlps,
            n_blocks,
        )
        self.tau = 1
        # del self.codebooks

        # self.encoder[0] = nn.Linear(input_size * input_frames, hidden_size)

        self.cond_size = cond_size
        self.cond_rms = RunningMeanStd(shape=(self.cond_size,))
        self.encoder = nn.Sequential(
            nn.Linear(input_size * input_frames + cond_size * cond_frames, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        del self.decoder

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
        encoder_in = torch.cat(
            [xx.reshape(xx.shape[0], -1), ww.reshape(ww.shape[0], -1)], dim=-1
        )
        z = self.encoder(encoder_in)
        return z

    def forward(
            self,
            x: torch.Tensor,
            w: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        zz = z.reshape(z.shape[0], 1, self.sentence_length, self.vocab_size)
        soft, hard = gumbel_softmax(zz, tau=self.tau)
        k = hard.argmax(-1)
        soft = soft.reshape(z.shape)
        hard = hard.reshape(z.shape)
        x_hat = self.decode(soft)
        return z, k, hard, x_hat


class TransformerGSVAE(GSVAEMixin):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sentence_length: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__(sentence_length, vocab_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = 0.0
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(
                self.input_size + self.hidden_size,
                self.sentence_length * self.vocab_size,
            ),
        )
        self.activation = nn.GELU()
        self.d_model = self.hidden_size // 64
        self.codebook = nn.Parameter(torch.zeros(self.vocab_size, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.deproj = nn.Linear(self.d_model, self.input_size)
        self.input_rms = RunningMeanStd(shape=(self.input_size,))
        self.tau = 1

        self.pe = nn.Parameter(
            torch.zeros(
                self.sentence_length,
                self.d_model,
            ),
            requires_grad=False,
        )
        div_term = torch.exp(
            torch.arange(0, self.pe.shape[1] // 2, 1).float()
            * (-np.log(1000.0) / self.pe.shape[1])
        )
        self.pe.data[..., 0::2] = torch.sin(
            torch.arange(self.pe.shape[0])[:, None] * div_term
        )
        self.pe.data[..., 1::2] = torch.cos(
            torch.arange(self.pe.shape[0])[:, None] * div_term
        )

    def setup(self, nail: torch.Tensor):
        self.input_rms.update(nail)
        self.tau = 1
        self.codebook.data = torch.rand_like(self.codebook.data)

    def encode(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xx = self.input_rms.normalize(x)
        z = xx * 1
        for j, op in enumerate(self.encoder):
            if isinstance(op, nn.Linear):
                if j > 0:
                    z = torch.cat([xx, z], dim=-1)
            z = op(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Assume the input is one-hot, with backprop enabled with Gumbel-Softmax trick
        """
        zz = z.reshape(z.shape[0], self.sentence_length, self.vocab_size)
        zzz = torch.matmul(zz, self.codebook)  # soft indexing
        zzzz = zzz + self.pe.detach()
        zzzzz = torch.cat([zzzz, torch.zeros_like(zzzz[:, [0]])], dim=1)
        xx = self.decoder(zzzzz)[:, [-1]]
        x = self.deproj(xx)
        return x


class CondTransformerGSVAE(GSVAEMixin):

    def __init__(
        self,
        input_size: int,
        cond_size: int,
        hidden_size: int,
        sentence_length: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__(sentence_length, vocab_size)
        self.input_size = input_size
        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = 0.0
        self.activation = nn.GELU()
        self.input_proj = nn.Linear(self.input_size - 1, self.hidden_size)
        self.cond_proj = nn.Linear(self.cond_size, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.deproj = nn.Linear(
            self.hidden_size, self.sentence_length * self.vocab_size
        )
        self.input_rms = RunningMeanStd(shape=(self.input_size,))
        self.cond_rms = RunningMeanStd(shape=(self.cond_size,))

        self.pe = nn.Parameter(
            torch.zeros(
                self.sentence_length,
                self.hidden_size,
            ),
            requires_grad=False,
        )
        div_term = torch.exp(
            torch.arange(0, self.pe.shape[1] // 2, 1).float()
            * (-np.log(1000.0) / self.pe.shape[1])
        )
        self.pe.data[..., 0::2] = torch.sin(
            torch.arange(self.pe.shape[0])[:, None] * div_term
        )
        self.pe.data[..., 1::2] = torch.cos(
            torch.arange(self.pe.shape[0])[:, None] * div_term
        )

    def setup(self, x: torch.Tensor, w: torch.Tensor):
        xx = x.reshape(-1, x.shape[-1])
        nan_yes = torch.isnan(xx).any(-1)
        self.input_rms.update(xx[~nan_yes])
        self.cond_rms.update(w)
        self.tau = 1

    def encode(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xx, x_mask = self.embed_x(x)
        ww, w_mask = self.embed_w(w)

        # position embeddings are implcitly done!
        xxww = torch.cat([xx[:, 0], ww], dim=1)
        xxww = torch.cat([xxww, torch.zeros_like(xxww[:, [0]])], dim=1)

        mask_simple = torch.cat([x_mask, w_mask], dim=1)
        mask_simple = torch.cat(
            [mask_simple, torch.zeros_like(mask_simple[:, [0]])], dim=-1
        )
        mask = (
            mask_simple[..., None]
            .repeat_interleave(mask_simple.shape[1], -1)
            .permute(0, 2, 1)
        )
        mask = mask.repeat_interleave(self.num_heads, 0)
        embedded_seq = self.encoder.forward(
            src=xxww,
            mask=mask,
        )

        z = self.deproj(embedded_seq[:, [-1]])
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return None

    def embed_x(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = x[..., 0] * 1
        t = t_orig * 1
        t[torch.isnan(t)] = 0

        # Form x stuff
        x = x * 1
        x[torch.isnan(x)] = 0

        xx = self.input_proj(self.input_rms.normalize(x)[..., 1:])

        # TTA encoding on nail embeddings
        pe = torch.zeros_like(xx)
        div_term = torch.exp(
            torch.arange(0, xx.shape[-1] // 2, 1).float()
            * (-np.log(1000.0) / (xx.shape[-1]))
        ).to(xx.device)
        pe[..., 0::2] = torch.sin(t[..., None] * div_term)
        pe[..., 1::2] = torch.cos(t[..., None] * div_term)
        xx += pe

        x_mask = torch.isnan(t_orig)
        x_mask = x_mask.reshape(x_mask.shape[0], -1)
        return xx, x_mask

    def embed_w(self, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t_orig = w[..., 0] * 1
        w[torch.isnan(w)] = 0
        ww = self.cond_proj(self.cond_rms.normalize(w))

        ww = ww.reshape(ww.shape[0], -1, ww.shape[-1])

        pe = torch.zeros_like(ww)
        div_term = torch.exp(
            torch.arange(0, ww.shape[-1] // 2, 1).float()
            * (-np.log(1000.0) / (ww.shape[-1]))
        ).to(ww.device)
        pe[..., 0::2] = torch.sin(
            torch.arange(ww.shape[1], device=ww.device)[..., None] * div_term
        )
        pe[..., 1::2] = torch.cos(
            torch.arange(ww.shape[1], device=ww.device)[..., None] * div_term
        )
        ww += pe
        w_mask = torch.isnan(t_orig.reshape(t_orig.shape[0], -1))

        return ww, w_mask


class ConvVQVAE(nn.Module):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        channel_size: int,
        kernel_size: int,
        latent_size: int,
        vocab_size: int,
        sentence_length: int,
        segment_length: int,
        padding: int,
        n_convs: int,
    ):
        super().__init__()
        self.latent_activation = nn.Identity()
        # self.latent_size = latent_size
        self.input_rms = RunningMeanStd(shape=(input_size,))
        self.channel_size = channel_size
        self.vocab_size = vocab_size
        self.codebook = nn.Parameter(
            torch.zeros(vocab_size, latent_size, dtype=torch.float).detach().clone(),
            requires_grad=True,
        )
        self.latent_size = latent_size
        self.sentence_length = sentence_length
        self.n_convs = n_convs
        # self.codebooks = nn.ParameterList(
        #     [
        #         nn.Parameter(
        #             torch.zeros(
        #                 # self.sentence_length,
        #                 self.vocab_size,
        #                 self.latent_size,
        #                 dtype=torch.float,
        #             ),
        #             # requires_grad=True if i == 0 else False,
        #             requires_grad=True,
        #         )
        #         for i in range(self.sentence_length)
        #     ]
        # )
        self.codebooks = nn.Parameter(
            torch.zeros(
                self.sentence_length,
                self.vocab_size,
                self.latent_size,
                dtype=torch.float,
            ),
            requires_grad=True,
        )
        self.segment_length = segment_length
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.LayerNorm([input_size, segment_length]),
            nn.Conv1d(input_size, channel_size, kernel_size, padding=padding),
            nn.LeakyReLU(),
            ConvRes1d(
                channel_size,
                channel_size,
                kernel_size,
                segment_length,
                padding=padding,
                n_convs=n_convs,
            ),
            nn.LeakyReLU(),
            nn.LayerNorm([channel_size, segment_length]),
            nn.Conv1d(channel_size, latent_size, kernel_size, padding=padding),
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.LayerNorm([latent_size, segment_length]),
            nn.Conv1d(latent_size, channel_size, kernel_size, padding=padding),
            nn.LeakyReLU(),
            ConvRes1d(
                channel_size,
                channel_size,
                kernel_size,
                segment_length,
                padding=padding,
                n_convs=n_convs,
            ),
            nn.LeakyReLU(),
            nn.LayerNorm([channel_size, segment_length]),
            nn.Conv1d(channel_size, input_size, kernel_size, padding=padding),
        )

        # self.decoder = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(latent_size, channel_size, kernel_size, padding=padding),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         channel_size, channel_size, kernel_size, padding=padding
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         channel_size, channel_size, kernel_size, padding=padding
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         channel_size, channel_size, kernel_size, padding=padding
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(channel_size, input_size, kernel_size, padding=padding),
        # )

    def setup(self, x: torch.Tensor):
        self.input_rms.update(x)
        for i in range(self.sentence_length):
            self.codebooks[i].data = torch.randn(
                self.vocab_size, self.latent_size, dtype=torch.float
            )

    def setup2(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.encode(x)
            # RVQ initialization
            quantized = torch.zeros_like(z)
            for i in range(self.sentence_length):
                means, assignments = my_kmeans(
                    z - quantized, self.vocab_size, 50, None, False
                )
                self.codebooks[i].data = means
                quantized += self.codebooks[i][assignments]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.input_rms.normalize(x).permute(0, 2, 1)
        # z = xx * 1
        z = self.encoder(xx)
        # for j, op in enumerate(self.encoder):
        #     if isinstance(op, nn.Conv1d):
        #         if j == 0:
        #             z = op(z)
        #         else:
        #             # z = torch.cat([xx, z], dim=1)  # strong conditioning skip connection
        #             z = xx + op(z)  # residual connection
        #         xx = z * 1
        #     else:
        #         z = op(z)
        return z.permute(0, 2, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = z.permute(0, 2, 1) * 1
        # x = zz * 1
        # # for j, op in enumerate(self.decoder):
        # #     # if isinstance(op, nn.ConvTranspose1d):
        # #     #     if j > 1:
        # #     #         x = torch.cat([zz, x], dim=1)
        # #     x = op(x)
        #
        # for j, op in enumerate(self.decoder):
        #     if isinstance(op, nn.Conv1d):
        #         if j == 0:
        #             x = op(x)
        #         else:
        #             # z = torch.cat([xx, z], dim=1)  # strong conditioning skip connection
        #             x = zz + op(x)  # residual connection
        #         zz = x * 1
        #     else:
        #         x = op(x)
        x = self.decoder(zz)
        return x.permute(0, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        dropout_mask: torch.Tensor = None,
        quant_yes: bool = False,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x)
        quantized = torch.zeros_like(z)
        k = []

        if not quant_yes:
            # Sanity check
            k = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=torch.long)
            z_q = z
        else:
            for i in range(self.sentence_length):
                with torch.no_grad():
                    deltas = (z - quantized)[:, None] - self.codebooks[i][None, :, None]
                    dists = torch.mean(deltas**2, dim=-1)
                    words = torch.argmin(dists, dim=1)
                    k.append(words)
                if dropout_mask is None:
                    quantized += self.codebooks[i][words]
                else:
                    quantized += self.codebooks[i][words] * dropout_mask[i]
            z_q = z + (quantized - z.detach())
            # z_q = z + (quantized - z).detach()
            k = torch.stack(k, dim=-1)

        x_hat = self.decode(z_q)
        return z, k, z_q, x_hat


class ConvFSQVAE(nn.Module):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        channel_size: int,
        kernel_size: int,
        latent_size: int,
        vocab_size: int,
        sentence_length: int,
        segment_length: int,
        padding: int,
        n_convs: int,
    ):
        super().__init__()
        self.latent_activation = nn.Identity()
        # self.latent_size = latent_size
        self.input_rms = RunningMeanStd(shape=(input_size,))
        self.channel_size = channel_size
        self.vocab_size = vocab_size
        self.codebook = nn.Parameter(
            torch.zeros(vocab_size, latent_size, dtype=torch.float).detach().clone(),
            requires_grad=True,
        )
        self.latent_size = latent_size
        self.sentence_length = sentence_length
        self.n_convs = n_convs
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        # self.sentence_length,
                        self.vocab_size,
                        self.latent_size,
                        dtype=torch.float,
                    ),
                    # requires_grad=True if i == 0 else False,
                    requires_grad=True,
                )
                for i in range(self.sentence_length)
            ]
        )
        self.segment_length = segment_length
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.LayerNorm([input_size, segment_length]),
            nn.Conv1d(input_size, channel_size, kernel_size, padding=padding),
            nn.LeakyReLU(),
            ConvRes1d(
                channel_size,
                channel_size,
                kernel_size,
                segment_length,
                padding=padding,
                n_convs=n_convs,
            ),
            nn.LeakyReLU(),
            nn.LayerNorm([channel_size, segment_length]),
            nn.Conv1d(channel_size, latent_size, kernel_size, padding=padding),
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.LayerNorm([latent_size, segment_length]),
            nn.Conv1d(latent_size, channel_size, kernel_size, padding=padding),
            nn.LeakyReLU(),
            ConvRes1d(
                channel_size,
                channel_size,
                kernel_size,
                segment_length,
                padding=padding,
                n_convs=n_convs,
            ),
            nn.LeakyReLU(),
            nn.LayerNorm([channel_size, segment_length]),
            nn.Conv1d(channel_size, input_size, kernel_size, padding=padding),
        )

        # self.decoder = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(latent_size, channel_size, kernel_size, padding=padding),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         channel_size, channel_size, kernel_size, padding=padding
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         channel_size, channel_size, kernel_size, padding=padding
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         channel_size, channel_size, kernel_size, padding=padding
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(channel_size, input_size, kernel_size, padding=padding),
        # )

    def setup(self, x: torch.Tensor):
        self.input_rms.update(x)

    def setup2(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.encode(x)
            # RVQ initialization
            quantized = torch.zeros_like(z)
            for i in range(self.sentence_length):
                means, assignments = my_kmeans(
                    z - quantized, self.vocab_size, 50, None, False
                )
                self.codebooks[i].data = means
                quantized += self.codebooks[i][assignments]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.input_rms.normalize(x).permute(0, 2, 1)
        z = self.encoder(xx)
        return z.permute(0, 2, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = z.permute(0, 2, 1) * 1
        x = self.decoder(zz)
        return x.permute(0, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        dropout_mask: torch.Tensor = None,
        quant_yes: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        z = self.encode(x)
        k = torch.round(torch.tanh(z) * (self.vocab_size // 2)) + (self.vocab_size // 2)
        z_q = torch.arctanh((k - (self.vocab_size // 2)) / (self.vocab_size // 2))
        x_hat = self.decode(z_q)
        return z, k, z_q, x_hat
