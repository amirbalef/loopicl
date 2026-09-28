"""Softmax scaling for ICL attention.

Provides:
  SoftmaxScalingMLP  — query-aware FiLM scaling with guaranteed length extrapolation.
  soft_cap_sdpa      — manual SDPA that applies tanh soft-capping to logits before
                       softmax, preventing attention collapse or over-sharpening at
                       unseen sequence lengths.

Design principles
-----------------
1. Normalized input: log(n / n_ref) instead of raw log(n).
   At training time the input is clustered around 0; at test-time extrapolation
   it is a smooth positive continuation rather than a jump to an unseen large value.

2. Residual base (gamma): 1 + gamma_mlp(log_ratio).
   The output layer of gamma_mlp is zero-initialised so gamma=1 at init.
   If the MLP saturates OOD the scale falls back to 1 — never collapses.

3. Logit soft-capping: tanh(logits / cap) * cap before softmax.
   Bounds logits to (-cap, cap), preventing extreme peaking or dilution at
   unseen lengths regardless of the scaling path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _safe_log_seqlen(n: int | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(n, torch.Tensor):
        return n.to(torch.float32).clamp(min=1).log().to(dtype)
    one = torch.ones((), dtype=torch.float32, device=device)
    return (one * n).clamp(min=1).log().to(dtype)


# Max bytes for the logit tensor [B, H, S_q, S_k] before chunking over B.
# At 2 GB: chunk_size = 2e9 / (H * S_q * S_k * 2). Cross-col with H=8, S=667:
# chunk ≈ 2e9 / (8 * 667 * 667 * 2) ≈ 281 rows → use 256 for safety.
_SOFT_CAP_MAX_LOGIT_BYTES: int = 2 * 1024 ** 3  # 2 GB


def soft_cap_sdpa(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor,
    v_BSJD: torch.Tensor,
    cap: float,
    attn_mask: torch.Tensor | None = None,
    allow_fa4: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention with tanh logit soft-capping.

    Identical interface to layers.scaled_dot_product_attention.
    Input layout: (B, S, H, D). GQA supported via head repeat.

    Prefers FA4 (flash_attn.cute) which handles softcap natively via
    scaled_dot_product_attention(softcap=cap). Only falls back to the
    explicit O(S²) materialization when FA4 is unavailable (CPU, float32,
    or large D).
    """
    # FA4 handles softcap natively with O(S) backward memory.
    # Only eligible when: FA4 available, CUDA tensor, fp16/bf16, no attn_mask
    # (FA4 doesn't support additive bias masks in BSHD mode).
    # D is padded to next valid FA4 size inside scaled_dot_product_attention.
    if (allow_fa4
            and attn_mask is None
            and q_BSHD.is_cuda
            and q_BSHD.dtype in (torch.float16, torch.bfloat16)):
        try:
            from .layers import _HAS_FLASH_ATTN_FAST, _fa_func
            if _HAS_FLASH_ATTN_FAST and _fa_func is not None:
                from .layers import scaled_dot_product_attention as _sdpa
                return _sdpa(q_BSHD, k_BSJD, v_BSJD, attn_mask=None, softcap=cap, allow_fa4=True)
        except (ImportError, AttributeError):
            pass

    # Manual O(S²) fallback (CPU, float32, attn_mask present, or FA4 absent).
    # NOTE: ALL chunk logit tensors stay live on the autograd tape until their
    # backward is processed; chunking helps forward peak but not backward peak.
    import torch as _t
    B, S_q, H, D = q_BSHD.shape
    S_k = k_BSJD.shape[1]
    logit_bytes_per_row = H * S_q * S_k * q_BSHD.element_size()
    chunk = max(1, _SOFT_CAP_MAX_LOGIT_BYTES // logit_bytes_per_row)

    if chunk >= B:
        return _soft_cap_sdpa_chunk(q_BSHD, k_BSJD, v_BSJD, cap, attn_mask)

    chunks = []
    for i in range(0, B, chunk):
        chunks.append(_soft_cap_sdpa_chunk(
            q_BSHD[i:i+chunk], k_BSJD[i:i+chunk], v_BSJD[i:i+chunk], cap, attn_mask))
    return torch.cat(chunks, dim=0)


def _soft_cap_sdpa_chunk(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor,
    v_BSJD: torch.Tensor,
    cap: float,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    q = q_BSHD.permute(0, 2, 1, 3)   # (B, H, S, D)
    k = k_BSJD.permute(0, 2, 1, 3)
    v = v_BSJD.permute(0, 2, 1, 3)
    if q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    scale = q.shape[-1] ** -0.5
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    logits = torch.tanh(logits / cap) * cap
    if attn_mask is not None:
        logits = logits + attn_mask
    weights = F.softmax(logits, dim=-1)
    out = torch.matmul(weights, v)
    return out.permute(0, 2, 1, 3)


class SoftmaxScalingMLP(nn.Module):
    """Query-aware attention scaling with guaranteed length extrapolation.

    Parameters
    ----------
    num_heads:          number of attention heads.
    head_dim:           dimension per head.
    n_hidden:           hidden size for both MLP branches.
    n_ref:              reference sequence length used to normalize the log input.
                        Choose the typical / max training length (e.g. 512 or 1024).
    attn_logit_softcap: if not None, _batched_scaled_dot_product_attention reads
                        this value from the module and applies tanh soft-capping
                        to logits before softmax.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
        n_ref: int = 512,
        attn_logit_softcap: float | None = None,
        use_tanh_log_ratio: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.log_n_ref = math.log(max(n_ref, 1))
        self.attn_logit_softcap = attn_logit_softcap
        self.use_tanh_log_ratio = use_tanh_log_ratio

        base_out_dim = num_heads * head_dim

        # Residual scale: gamma = 1 + gamma_mlp(log_ratio)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out_dim)
        )
        # Per-query content modulation (unchanged interface from v1)
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, head_dim)
        )
        # Zero-init outputs: identity at init AND safe OOD fallback
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.zeros_(self.gamma_mlp[-1].bias)
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q_BSHD: torch.Tensor, n: int) -> torch.Tensor:
        log_ratio = _safe_log_seqlen(n, q_BSHD.device, q_BSHD.dtype) - self.log_n_ref
        if self.use_tanh_log_ratio:
            # tanh(log(n/n_ref)) ∈ [-1, 1]: bounded, ~0 at n_ref, smooth OOD.
            log_ratio = torch.tanh(log_ratio)
        gamma = 1.0 + self.gamma_mlp(log_ratio.reshape(1, 1)).view(1, 1, self.num_heads, self.head_dim)
        modulation = 1.0 + torch.tanh(self.query_mlp(q_BSHD))
        return q_BSHD * gamma * modulation
