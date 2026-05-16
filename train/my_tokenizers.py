from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn as nn
from torch.quasirandom import SobolEngine
from tqdm import tqdm


class BucketizeTokenizer(nn.Module):
    """
    Baseline, but probably good enough
    """

    def __init__(
        self,
        n_quantizers: int,
        codebook_size: int,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        # self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.codebooks = torch.as_tensor(
            np.mgrid[min_value : max_value : complex(0, codebook_size)],
            dtype=torch.float,
        )
        self.min_value = min_value
        self.max_value = max_value
        assert self.codebook_size > 0

    def initialize(self, x: torch.Tensor, device="cuda:0"):
        pass

    def encode(self, x: torch.Tensor, device="cuda:0"):
        encoded = torch.clip(x, self.min_value, self.max_value)  # [min, max]
        encoded -= self.min_value  # [0, max - min]
        encoded /= self.max_value - self.min_value  # [0, 1]
        encoded *= float(self.codebook_size - 1)
        encoded = torch.round(encoded).long().to(device)

        quantized = self.decode(encoded, device=device)

        return encoded, quantized

    def decode(self, x: torch.Tensor, device="cuda:0"):
        decoded = x.float() / (self.codebook_size - 1)
        decoded *= self.max_value - self.min_value
        decoded += self.min_value
        return decoded.to(device)


def my_kmeans(
    x: torch.Tensor,
    k: int,
    iters: int,
    means: torch.Tensor,
    cosine_sim_yes: bool,
    batch_size: int = None,
):
    with torch.no_grad():
        xx = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        # initial guess from low-disc sequence
        if means is None:
            # means = SobolEngine(d, seed=0).draw(k).double()
            # means = SobolEngine(d).draw(k).to(x.device, dtype=torch.float) * 2 - 1
            # means = means * 6 - 3
            means = xx[torch.randperm(n)[:k], :]
        for i in range(iters):
            if not cosine_sim_yes:
                if batch_size is None:  # full-batch distance compute
                    x_means_dist = torch.cdist(xx, means)
                else:
                    x_means_dist = (
                        torch.ones((n, k), dtype=x.dtype, device=x.device) * np.inf
                    )
                    for j in range(0, n, batch_size):
                        dists = torch.cdist(x[j : j + batch_size, :], means)
                        x_means_dist[j : j + batch_size, : dists.shape[-1]] = dists
            else:
                x_means_dist = 1 - torch.mm(x, means.t())
                x_means_dist /= (
                    x.norm(dim=-1, keepdim=True) * means.norm(dim=-1, keepdim=True).t()
                )
            assignments = torch.argmin(x_means_dist, dim=-1)

            # update centroids
            nn = xx.shape[0]
            mask = torch.zeros((nn, k), dtype=x.dtype, device=x.device)
            mask[torch.arange(nn), assignments] = 1 / nn
            n_assigns = mask.sum(dim=0, keepdim=True).t()
            zero_assign_yes = (n_assigns == 0).squeeze()
            zero_assign_idx = torch.where(zero_assign_yes)[0]
            n_zero_assigns = torch.sum(zero_assign_yes).cpu().numpy()
            means = torch.mm(torch.where(n_assigns == 0, 0, mask.t()), xx) / torch.where(
                n_assigns == 0, 1, n_assigns
            )
            to_assign = np.minimum(nn, n_zero_assigns)
            means[zero_assign_idx[:to_assign]] = xx[
                torch.randperm(n)[:n_zero_assigns], :
            ]

        x_means_dist = torch.cdist(xx, means)
        assignments = torch.argmin(x_means_dist, dim=-1)
        assignments = assignments.reshape(x.shape[:-1])
    return means, assignments


class RVQTokenizer(nn.Module):

    def __init__(
        self,
        feature_size: int,
        n_quantizers: int,
        codebook_size: int,
        kmeans_iters: int,
        kmeans_trials: int,
        cosine_sim_yes: bool,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.kmeans_trials = kmeans_trials
        self.cosine_sim_yes = cosine_sim_yes

        # self.feature_size: int or None = None
        # self.codebooks: torch.Tensor or None = None

        self.codebooks = (
            torch.rand(
                self.n_quantizers,
                self.codebook_size,
                self.feature_size,
                requires_grad=False,
                dtype=torch.float,
            )
            - 0.5
        ) * 2

    def build_codebook(
        self,
        x: torch.Tensor,
        scratch_yes: bool = True,
        batch_size: int = None,
        device="cuda:0",
    ):
        """
        Use the given dataset to build the initial RVQ codebooks
        """
        assert self.feature_size == x.shape[1]
        if scratch_yes:
            self.codebooks = torch.zeros(
                self.n_quantizers,
                self.codebook_size,
                self.feature_size,
                requires_grad=False,
                dtype=torch.float,
            )
        elif self.codebooks is None:
            raise RuntimeError(
                "scratch_yes must be False unless codebooks have been populated"
            )
        x = x.to(device)
        self.codebooks = self.codebooks.to(device)
        quantized = torch.zeros_like(x)
        for i in tqdm(range(self.n_quantizers)):
            # Hacky: support inheriting top level codebook
            if i <= 0 and not scratch_yes:
                means_, assignments = my_kmeans(
                    x - quantized,
                    self.codebook_size,
                    self.kmeans_iters,
                    self.codebooks[i],
                    self.cosine_sim_yes,
                    batch_size,
                )
                self.codebooks[i] = means_
                # x_means_dist = torch.cdist(x, self.codebooks[i])
                # assignments = torch.argmin(x_means_dist, dim=-1)
            else:
                means_, assignments = my_kmeans(
                    x - quantized,
                    self.codebook_size,
                    self.kmeans_iters,
                    None,
                    self.cosine_sim_yes,
                    batch_size,
                )
                self.codebooks[i] = means_
            quantized += self.codebooks[i][assignments]
        x = x.cpu()
        self.codebooks = self.codebooks.cpu()

    def encode(self, x: torch.Tensor, device="cuda:0"):
        if self.codebooks is None or self.feature_size is None:
            raise RuntimeError("RVQTokenizer not initialized")
        assert x.shape[1] == self.feature_size
        x = x.to(device)
        self.codebooks = self.codebooks.to(device)
        encoded = torch.zeros(
            x.shape[0],
            self.n_quantizers,
            dtype=torch.long,
            device=x.device,
            requires_grad=False,
        )
        quantized = torch.zeros_like(x, requires_grad=False)
        for j in range(self.n_quantizers):
            raw_to_means_dist = torch.cdist(x - quantized, self.codebooks[j])
            encoded[:, j] = torch.argmin(raw_to_means_dist, dim=1)
            quantized += self.codebooks[j, encoded[:, j]]
        x = x.cpu()
        self.codebooks = self.codebooks.cpu()
        return encoded, quantized

    def decode(self, x: torch.Tensor, device):
        if self.codebooks is None or self.feature_size is None:
            raise RuntimeError("RVQTokenizer not initialized")
        assert x.shape[1] == self.n_quantizers
        gathered = torch.gather(
            self.codebooks.swapaxes(0, 1).to(device),
            0,
            index=x.unsqueeze(-1).expand(-1, -1, self.feature_size),
        )
        decoded = torch.sum(gathered, dim=1)
        self.codebooks = self.codebooks.cpu()
        return decoded
