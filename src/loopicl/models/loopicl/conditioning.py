"""Conditioning modules for LoopICL.

Includes:
  - Histogram utilities (_adaptive_bin_geometry, _soft_log_histogram)
  - AugmentedLogHistConditioner
  - FourierQuantileEncoderWithOOD
  - DiscriminativeHistConditioner
  - MaskedICLTransformerBlock
  - ClassNormalizedManyClassDecoder
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCacheEntry
from .layers import (
    ICLAttention,
    MLP,
    ManyClassDecoder,
    MLPClassDecoder,
    SoftmaxScalingMLP,
    _DtypeMatchingRMSNorm,
    _chunked_class_attention,
    _safe_log_seqlen,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_HIST_BINS: int = 32
_N_FOURIER: int = 8
_N_MOMENTS: int = 4  # mean, std, skewness, excess_kurtosis


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------


def _adaptive_bin_geometry(
    x_sorted_BCN: torch.Tensor,
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical-quantile bin edges → centers and widths."""
    _, _, N = x_sorted_BCN.shape
    q = torch.linspace(0.0, 1.0, n_bins + 1, device=x_sorted_BCN.device)
    idx_float = q * (N - 1)
    idx_lo = idx_float.long().clamp(0, N - 1)
    idx_hi = (idx_lo + 1).clamp(0, N - 1)
    frac = (idx_float - idx_lo.float()).to(x_sorted_BCN.dtype)
    edges = (
        x_sorted_BCN[:, :, idx_lo] * (1.0 - frac)
        + x_sorted_BCN[:, :, idx_hi] * frac
    )
    centers = (edges[:, :, :-1] + edges[:, :, 1:]) / 2
    widths  = (edges[:, :, 1:]  - edges[:, :, :-1]).clamp(min=1e-6)
    return centers, widths


def _soft_log_histogram(
    x_train_BCN: torch.Tensor,
    centers: torch.Tensor,
    widths: torch.Tensor,
    num_train: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Triangular soft-binning → Laplace-smoothed log-probs → zero-mean.

    Computed in chunks over N to avoid materialising the full (B,C,N,K) tensor.
    """
    n_bins = centers.shape[-1]
    N = x_train_BCN.shape[2]
    soft_counts = torch.zeros(
        x_train_BCN.shape[0], x_train_BCN.shape[1], n_bins,
        dtype=x_train_BCN.dtype, device=x_train_BCN.device,
    )
    for start in range(0, N, chunk_size):
        chunk = x_train_BCN[:, :, start : start + chunk_size]
        dist  = (chunk.unsqueeze(-1) - centers.unsqueeze(2)) / widths.unsqueeze(2)
        soft_counts += (1.0 - dist.abs()).clamp(min=0.0).sum(2)
    probs    = (soft_counts + 0.5) / (num_train + 0.5 * n_bins)
    log_hist = torch.log(probs)
    return log_hist - log_hist.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Distribution moments
# ---------------------------------------------------------------------------


def _compute_moments(x_train_BCN: torch.Tensor) -> torch.Tensor:
    """Per-column [mean, std, skewness, excess_kurtosis].

    Mean and std are N-invariant after standardisation.
    Skewness and kurtosis are clamped to [-10, 10]: at small N they're noisy,
    at large N they're accurate — but either way the MLP was only trained on
    values typical of N≤1024, so extreme values are OOD. Clamping keeps them
    in-distribution regardless of N.
    """
    mean = x_train_BCN.mean(dim=-1)
    diff = x_train_BCN - mean.unsqueeze(-1)
    std  = diff.pow(2).mean(dim=-1).sqrt().clamp(min=1e-6)
    z    = diff / std.unsqueeze(-1)
    skew = z.pow(3).mean(dim=-1).clamp(-10.0, 10.0)
    kurt = (z.pow(4).mean(dim=-1) - 3.0).clamp(-10.0, 10.0)
    moments = torch.stack([mean, std, skew, kurt], dim=-1)
    return torch.nan_to_num(moments, nan=0.0, posinf=10.0, neginf=-10.0)


# ---------------------------------------------------------------------------
# AugmentedLogHistConditioner
# ---------------------------------------------------------------------------


class AugmentedLogHistConditioner(nn.Module):
    """Projects (log-histogram ∥ moments) to embed_dim."""

    def __init__(self, embed_dim: int, n_hidden: int = 64, n_bins: int = _N_HIST_BINS) -> None:
        super().__init__()
        in_dim = n_bins + _N_MOMENTS
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, embed_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, log_hist_BCK: torch.Tensor, moments_BC4: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([log_hist_BCK, moments_BC4], dim=-1))


# ---------------------------------------------------------------------------
# FourierQuantileEncoderWithOOD
# ---------------------------------------------------------------------------


class FourierQuantileEncoderWithOOD(nn.Module):
    """Per-cell empirical rank encoder with Fourier basis + OOD flag."""

    def __init__(self, embed_dim: int, n_fourier: int = _N_FOURIER, n_hidden: int = 64) -> None:
        super().__init__()
        self.n_fourier = n_fourier
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_fourier + 1, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, embed_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_BRC: torch.Tensor, x_sorted_BCN: torch.Tensor, num_train: int,
                chunk_size: int = 2048) -> torch.Tensor:
        B, R, C = x_BRC.shape
        N = num_train
        sorted_flat = x_sorted_BCN.reshape(B * C, N)
        query_flat  = x_BRC.permute(0, 2, 1).reshape(B * C, R)
        rank_idx = torch.searchsorted(sorted_flat.float().contiguous(), query_flat.float().contiguous())
        is_extrap = ((rank_idx == 0) | (rank_idx == N)).to(x_BRC.dtype)
        ranks      = (rank_idx.to(x_BRC.dtype) / N).reshape(B, C, R).permute(0, 2, 1)  # (B,R,C)
        extrap_BRC = is_extrap.reshape(B, C, R).permute(0, 2, 1)
        k     = torch.arange(self.n_fourier, device=x_BRC.device, dtype=x_BRC.dtype)
        freqs = (2.0 ** k) * math.pi

        # Chunk over R to avoid materialising (B, R, C, n_hidden) intermediates.
        # At R=100k+M, C=100, n_hidden=64: full tensor ≈ 1.28 GB bfloat16.
        out = torch.empty(B, R, C, self.mlp[-1].out_features,
                          dtype=x_BRC.dtype, device=x_BRC.device)
        for start in range(0, R, chunk_size):
            end = min(start + chunk_size, R)
            angles_chunk = ranks[:, start:end].unsqueeze(-1) * freqs       # (B, chunk, C, n_fourier)
            fourier_chunk = torch.cat(
                [angles_chunk.sin(), angles_chunk.cos(),
                 extrap_BRC[:, start:end].unsqueeze(-1)], dim=-1
            )
            out[:, start:end] = self.mlp(fourier_chunk)
        return out


# ---------------------------------------------------------------------------
# DiscriminativeHistConditioner
# ---------------------------------------------------------------------------


class DiscriminativeHistConditioner(nn.Module):
    """Per-column discriminative signal via class-vs-marginal histogram residuals."""

    def __init__(self, embed_dim: int, max_classes: int, n_hidden: int = 64,
                 n_bins: int = _N_HIST_BINS) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.max_classes = max_classes
        self.mlp = nn.Sequential(
            nn.Linear(n_bins, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, embed_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_train_BCN: torch.Tensor, centers: torch.Tensor,
                widths: torch.Tensor, log_hist_BCK: torch.Tensor,
                y_BN: torch.Tensor, num_train: int,
                chunk_size: int = 2048) -> torch.Tensor:
        B, C, N = x_train_BCN.shape
        T = self.max_classes
        K = self.n_bins
        y_onehot = F.one_hot(y_BN.long().clamp(0, T - 1), num_classes=T).to(x_train_BCN.dtype)
        class_sizes = y_onehot.sum(1)  # (B, T)

        # Accumulate class soft-counts in chunks over N to avoid (B,C,N,K) OOM.
        # At N=100k with C=100, K=32: full tensor = 100*100k*32 floats ≈ 1.2 GB.
        class_counts = torch.zeros(B, C, T, K, dtype=x_train_BCN.dtype, device=x_train_BCN.device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_train_BCN[:, :, start:end]                          # (B, C, chunk)
            y_chunk = y_onehot[:, start:end]                                 # (B, chunk, T)
            dist_c  = (x_chunk.unsqueeze(-1) - centers.unsqueeze(2)) / widths.unsqueeze(2)
            soft_c  = (1.0 - dist_c.abs()).clamp(min=0.0)                   # (B, C, chunk, K)
            class_counts += torch.einsum("bcnk,bnt->bctk", soft_c, y_chunk)

        denom    = class_sizes.view(B, 1, T, 1).clamp(min=1.0) + 0.5 * K
        probs    = (class_counts + 0.5) / denom
        log_hist_cls = torch.log(probs)
        log_hist_cls = log_hist_cls - log_hist_cls.mean(dim=-1, keepdim=True)
        residuals = log_hist_cls - log_hist_BCK.unsqueeze(2)
        present   = (class_sizes > 0).float().view(B, 1, T, 1)
        residuals = residuals * present
        BCT = B * C * T
        emb = self.mlp(residuals.reshape(BCT, K)).view(B, C, T, -1)
        n_present = (class_sizes > 0).float().sum(-1).view(B, 1, 1).clamp(min=1.0)
        return emb.sum(2) / n_present


# ---------------------------------------------------------------------------
# ICL block
# ---------------------------------------------------------------------------


class MaskedICLTransformerBlock(nn.Module):
    """FA4-compatible ICL transformer block with sandwich normalisation support."""

    def __init__(self, *, emsize: int, nhead: int, dim_feedforward: int,
                 norm_factory, softmax_scaling_layer: nn.Module | None = None,
                 num_kv_heads: int | None = None,
                 num_kv_heads_test: int | None = None,
                 post_norm_factory=None,
                 device=None, dtype=None) -> None:
        nn.Module.__init__(self)
        assert emsize % nhead == 0
        kw = {"device": device, "dtype": dtype}
        self.icl_attention = ICLAttention(
            embedding_size=emsize, num_heads=nhead, head_dim=emsize // nhead,
            softmax_scaling_layer=softmax_scaling_layer,
            num_kv_heads=num_kv_heads, num_kv_heads_test=num_kv_heads_test, **kw,
        )
        self.layernorm     = norm_factory(emsize)
        self.layernorm_mlp = norm_factory(emsize)
        self.mlp = MLP(emsize, dim_feedforward, **kw)
        self.post_norm_attn = post_norm_factory(emsize) if post_norm_factory is not None else None
        self.post_norm_mlp = post_norm_factory(emsize) if post_norm_factory is not None else None

    def forward(self, x_BRE: torch.Tensor, single_eval_pos: int,
                save_peak_memory_factor: int | None = None, *,
                cached_kv: KVCacheEntry | None = None,
                return_kv: bool = False,
                residual_alpha: torch.Tensor | None = None,
                residual_beta: torch.Tensor | None = None) -> tuple[torch.Tensor, KVCacheEntry | None]:
        def _residual(x, delta):
            if residual_alpha is None:
                return x + delta
            b = residual_beta if residual_beta is not None else (1 - residual_alpha)
            return residual_alpha * x + b * delta

        attn_out, kv_entry = self.icl_attention(self.layernorm(x_BRE), single_eval_pos=single_eval_pos,
                                                cached_kv=cached_kv, return_kv=return_kv)
        x_BRE = _residual(x_BRE, attn_out)
        if self.post_norm_attn is not None:
            x_BRE = self.post_norm_attn(x_BRE)
        x_BRE = _residual(x_BRE, self.mlp(self.layernorm_mlp(x_BRE)))
        if self.post_norm_mlp is not None:
            x_BRE = self.post_norm_mlp(x_BRE)
        return x_BRE, kv_entry


# ---------------------------------------------------------------------------
# ClassNormalizedManyClassDecoder
# ---------------------------------------------------------------------------


class ClassNormalizedManyClassDecoder(ManyClassDecoder):
    """ManyClassDecoder that normalizes attention mass by per-class count."""

    def forward(self, train_embeddings: torch.Tensor, test_embeddings: torch.Tensor,
                targets: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, M, _ = test_embeddings.shape
        T = self.max_num_classes
        N = train_embeddings.shape[1]

        if train_embeddings.dtype != test_embeddings.dtype:
            train_embeddings = train_embeddings.to(test_embeddings.dtype)

        q_BME = self.q_projection(test_embeddings)
        k_BNE = self.k_projection(train_embeddings)

        if M == 0:
            empty = test_embeddings.new_empty((0, B, T))
            return empty + (q_BME.sum() + k_BNE.sum()) * 0.0

        one_hot_BNT = F.one_hot(targets.long(), num_classes=T).to(q_BME.dtype)

        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            one_hot_BNT = one_hot_BNT * (~key_padding_mask).unsqueeze(-1).to(q_BME.dtype)
            attn_mask = torch.zeros(B, 1, 1, N, device=q_BME.device, dtype=q_BME.dtype)
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        q_BMHD = q_BME.view(B, M, self.num_heads, self.head_dim).contiguous()
        k_BNHD = k_BNE.view(B, N, self.num_heads, self.head_dim).contiguous()
        one_hot_BNHT = one_hot_BNT.unsqueeze(2).expand(-1, -1, self.num_heads, -1).contiguous()

        raw_BMHT = _chunked_class_attention(q_BMHD, k_BNHD, one_hot_BNHT,
                                            softmax_scaling_layer=self.softmax_scaling_layer,
                                            attn_mask=attn_mask)

        counts_BT = one_hot_BNT.sum(1)
        avg_BMHT  = raw_BMHT / counts_BT.view(B, 1, 1, T).clamp(min=1)
        probs_BMHT = avg_BMHT / avg_BMHT.sum(-1, keepdim=True).clamp(min=1e-8)
        probs_BMT  = probs_BMHT.mean(2)
        probs_MBT  = probs_BMT.transpose(0, 1)
        return torch.log(probs_MBT + 3e-5)
