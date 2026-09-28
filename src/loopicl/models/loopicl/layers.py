"""Building-block layers for LoopICL.

Shape suffix convention:
    B  : batch size
    R  : total rows (train + test)
    N  : train rows
    M  : test rows
    C  : total columns
    E  : embedding dimension
    T  : target dim (num classes)
    Cl : number of CLS tokens
    D  : ICL dim  (= Cl * E)
    H  : num heads
    S  : sequence length
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import ParamSpec

from .kv_cache import KVCache, KVCacheEntry
from .softmax_scaling import SoftmaxScalingMLP as SoftmaxScalingMLPv2, soft_cap_sdpa

if TYPE_CHECKING:
    pass

P = ParamSpec("P")

# ---------------------------------------------------------------------------
# Flash Attention 3 / 4 (Hopper + Blackwell) — optional fast path
# ---------------------------------------------------------------------------

_HAS_FLASH_ATTN_FAST = False
_FLASH_ATTN_FAST_SOURCE = "none"
# FA4 uses BSHD layout directly; FA3/FA2 use varlen format.
_fa_func = None          # flash_attn.cute.flash_attn_func  (FA4, BSHD)
_fa_varlen_func = None   # flash_attn_varlen_func            (FA3/FA2, varlen)

# Ensure CUDA_TOOLKIT_PATH is set so FA4's JIT compiler can recompile new kernel
# variants during training (e.g. different seqlen or head_dim triggers recompile).
import os as _os
if 'CUDA_TOOLKIT_PATH' not in _os.environ:
    try:
        import torch.utils.cpp_extension as _cppext
        _cuda_home = getattr(_cppext, 'CUDA_HOME', None)
        if _cuda_home and _os.path.isdir(_cuda_home):
            _os.environ['CUDA_TOOLKIT_PATH'] = _cuda_home
    except Exception:
        pass
    if 'CUDA_TOOLKIT_PATH' not in _os.environ:
        for _p in ['/usr/local/cuda', '/usr/cuda']:
            if _os.path.isdir(_p):
                _os.environ['CUDA_TOOLKIT_PATH'] = _p
                break
del _os


# Probe at head_dim=32 AND 64 (FA4-supported dims). The model uses head_dim=16,
# which FA4 doesn't support natively; the fast path zero-pads to the next valid
# size in (32, 64, 128, 256). Probing both 32 and 64 pre-compiles those kernel
# variants so JIT recompilation isn't needed mid-training.
_PROBE_SEQLEN   = 4
_PROBE_NHEADS   = 2


def _cuda_probe_fwd_bwd(fn, q, k, v) -> bool:
    """Probe both forward and backward pass. Required to catch FA4 bwd JIT bugs."""
    try:
        import torch as _t
        result = fn(q, k, v)
        out = result[0] if isinstance(result, tuple) else result
        out.sum().backward()
        _t.cuda.synchronize()
        return True
    except Exception as _e:
        _probe_error = str(_e)  # noqa: F841 — visible in debugger / tracebacks
        return False


def _cuda_probe_fa4_all_dims(fn) -> bool:
    """Probe FA4 with all head_dims the model may use (32, 64) to pre-compile kernels."""
    import torch as _t
    for _dim in (32, 64):
        _q = _t.randn(1, _PROBE_SEQLEN, _PROBE_NHEADS, _dim,
                      dtype=_t.bfloat16, device="cuda", requires_grad=True)
        _k = _t.randn_like(_q)
        _v = _t.randn_like(_q)
        if not _cuda_probe_fwd_bwd(fn, _q, _k, _v):
            return False
        del _q, _k, _v
    return True


def _cuda_probe_varlen_fwd_bwd(fn, q, k, v, cu_q, cu_k, sq, sk) -> bool:
    """Probe varlen interface fwd+bwd."""
    try:
        import torch as _t
        result = fn(q, k, v, cu_q, cu_k, sq, sk)
        out = result[0] if isinstance(result, tuple) else result
        out.sum().backward()
        _t.cuda.synchronize()
        return True
    except Exception as _e:
        _probe_error = str(_e)  # noqa: F841
        return False


# ---- FA4 (flash_attn.cute) — Blackwell-native BSHD [B,S,H,D] interface ----
try:
    from flash_attn.cute import flash_attn_func as _fa4_func
    # Probe with all head_dims used during training (32, 64) so their CUDA
    # kernels are pre-compiled now rather than JIT-compiled mid-training.
    # requires_grad=True so backward compiles during probe, catching JIT bugs early.
    if _cuda_probe_fa4_all_dims(_fa4_func):
        _fa_func = _fa4_func
        _HAS_FLASH_ATTN_FAST = True
        _FLASH_ATTN_FAST_SOURCE = "flash_attn.cute (FA4)"
except Exception:
    pass

# ---- FA3 (flash_attn_interface) — Hopper-native varlen [total,H,D] interface ----
if not _HAS_FLASH_ATTN_FAST:
    try:
        from flash_attn_interface import flash_attn_varlen_func as _fai_varlen
        import torch as _pt
        _q1 = _pt.randn(_PROBE_SEQLEN, _PROBE_NHEADS, 64,
                        dtype=_pt.bfloat16, device="cuda", requires_grad=True)
        _k1 = _pt.randn_like(_q1)
        _v1 = _pt.randn_like(_q1)
        _cu = _pt.tensor([0, _PROBE_SEQLEN], dtype=_pt.int32, device="cuda")
        if _cuda_probe_varlen_fwd_bwd(_fai_varlen, _q1, _k1, _v1, _cu, _cu,
                                      _PROBE_SEQLEN, _PROBE_SEQLEN):
            _fa_varlen_func = _fai_varlen
            _HAS_FLASH_ATTN_FAST = True
            _FLASH_ATTN_FAST_SOURCE = "flash_attn_interface (FA3)"
        del _q1, _k1, _v1, _cu, _pt
    except Exception:
        pass

# ---- FA2 (flash_attn) — varlen [total,H,D] interface ----
if not _HAS_FLASH_ATTN_FAST:
    try:
        from flash_attn import flash_attn_varlen_func as _fa2_varlen
        import torch as _pt
        _q1 = _pt.randn(_PROBE_SEQLEN, _PROBE_NHEADS, 64,
                        dtype=_pt.bfloat16, device="cuda", requires_grad=True)
        _k1 = _pt.randn_like(_q1)
        _v1 = _pt.randn_like(_q1)
        _cu = _pt.tensor([0, _PROBE_SEQLEN], dtype=_pt.int32, device="cuda")
        if _cuda_probe_varlen_fwd_bwd(_fa2_varlen, _q1, _k1, _v1, _cu, _cu,
                                      _PROBE_SEQLEN, _PROBE_SEQLEN):
            _fa_varlen_func = _fa2_varlen
            _HAS_FLASH_ATTN_FAST = True
            _FLASH_ATTN_FAST_SOURCE = "flash_attn (FA2)"
        del _q1, _k1, _v1, _cu, _pt
    except Exception:
        pass

# ---------------------------------------------------------------------------
# SDPA — backend logging helpers
# ---------------------------------------------------------------------------

def _sdpa_log(q_shape, k_shape, backend: str) -> None:
    pass


def _sdpa_oom(backend: str, q_shape, k_shape) -> None:
    pass


# ---------------------------------------------------------------------------
# SDPA
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor,
    v_BSJD: torch.Tensor,
    _backends_override=None,
    attn_mask: torch.Tensor | None = None,
    softcap: float | None = None,
    allow_fa4: bool = False,
) -> torch.Tensor:
    """SDPA in (B, S, H, D) layout with optional GQA via head-repeat.

    Priority:
      1. FA4 via flash_attn.cute (Blackwell SM_100+) — BSHD layout, fastest
         Supports softcap natively → avoids O(S²) logit materialisation.
      2. FA3 via flash_attn_interface / FA2 via flash_attn — varlen layout
      3. PyTorch SDPA (mem-efficient preferred over math to avoid O(S²) OOM)
    Fast paths require: no attn_mask, CUDA tensor, fp16/bf16 dtype.
    """
    _fast = (
        allow_fa4
        and _HAS_FLASH_ATTN_FAST
        and attn_mask is None
        and q_BSHD.is_cuda
        and q_BSHD.dtype in (torch.float16, torch.bfloat16)
    )

    # ---- FA4: flash_attn.cute — takes BSHD directly, no reshape needed ----
    # Returns (output, softmax_lse) tuple; take [0] for the output tensor.
    # FA4 requires head_dim ∈ {32, 64, 128, 256}. For smaller head_dims (e.g.
    # ISAB/cross-col with D=16) we zero-pad Q/K/V to the next valid size and
    # slice the output back. Zero-padding doesn't change attention scores because
    # the padded dimensions contribute 0 to every dot product; the output for
    # padded V dimensions is 0 and is dropped. This gives O(S) backward memory
    # (FA4 flash-style) instead of O(S²) from the MATH fallback.
    _FA4_VALID = (32, 64, 128, 256)
    if _fast and _fa_func is not None:
        D = q_BSHD.shape[-1]
        q_h, k_h = q_BSHD.shape[2], k_BSJD.shape[2]
        D_fa4 = next((s for s in _FA4_VALID if s >= D), None)
        if D_fa4 is not None and (q_h == k_h or q_h % k_h == 0):
            pad = D_fa4 - D
            q_f = F.pad(q_BSHD, (0, pad)) if pad else q_BSHD
            k_f = F.pad(k_BSJD, (0, pad)) if pad else k_BSJD
            v_f = F.pad(v_BSJD, (0, pad)) if pad else v_BSJD
            kw = {} if softcap is None else {"softcap": softcap}
            tag = ("FA4" + (f"→{D_fa4}" if pad else "") +
                   ("+softcap" if softcap else ""))
            _sdpa_log(q_BSHD.shape, k_BSJD.shape, tag)
            try:
                out = _fa_func(
                    q_f.contiguous(),
                    k_f.contiguous(),
                    v_f.contiguous(),
                    **kw,
                )[0]
                return out[..., :D] if pad else out
            except torch.cuda.OutOfMemoryError:
                _sdpa_oom(tag, q_BSHD.shape, k_BSJD.shape)
                raise

    # Shared permute + GQA expand for FA3/FA2 and PyTorch SDPA paths
    q = q_BSHD.permute(0, 2, 1, 3)  # [B, H, S, D]
    k = k_BSJD.permute(0, 2, 1, 3)
    v = v_BSJD.permute(0, 2, 1, 3)
    if q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # ---- FA3/FA2: varlen interface ----
    if _fast and _fa_varlen_func is not None:
        B, H, S_q, D = q.shape
        S_k = k.shape[2]
        q_fa = q.permute(0, 2, 1, 3).reshape(B * S_q, H, D)
        k_fa = k.permute(0, 2, 1, 3).reshape(B * S_k, H, D)
        v_fa = v.permute(0, 2, 1, 3).reshape(B * S_k, H, D)
        cu_q = torch.arange(0, (B + 1) * S_q, S_q, dtype=torch.int32, device=q.device)
        cu_k = torch.arange(0, (B + 1) * S_k, S_k, dtype=torch.int32, device=q.device)
        _sdpa_log(q_BSHD.shape, k_BSJD.shape, "FA3/FA2-varlen")
        out = _fa_varlen_func(q_fa, k_fa, v_fa, cu_q, cu_k, S_q, S_k)
        return out.view(B, S_q, H, D)

    # Fallback: PyTorch SDPA dispatcher.
    # Always force EFFICIENT_ATTENTION (O(S) memory) before MATH (O(S²)).
    # On SM_100, EFFICIENT_ATTENTION may not be available for some head_dims,
    # causing silent fallback to MATH which OOMs on large (B, S) combinations
    # (e.g. cross-col: B=100k rows, S=667 groups → 711 GB with MATH).
    # Auto-chunk over B when the logit tensor would exceed 2 GB.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _MAX_LOGIT_BYTES = 2 * 1024 ** 3
    B, H, S_q, _ = q.shape
    S_k = k.shape[2]
    logit_bytes = B * H * S_q * S_k * q.element_size()
    if logit_bytes > _MAX_LOGIT_BYTES:
        chunk = max(1, _MAX_LOGIT_BYTES // (H * S_q * S_k * q.element_size()))
        _sdpa_log(q_BSHD.shape, k_BSJD.shape, f"EFFICIENT/MATH-chunked(chunk={chunk})")
        out_chunks = []
        for i in range(0, B, chunk):
            try:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out_chunks.append(F.scaled_dot_product_attention(
                        q[i:i+chunk], k[i:i+chunk], v[i:i+chunk], attn_mask=attn_mask))
            except torch.cuda.OutOfMemoryError:
                _sdpa_oom(f"EFFICIENT/MATH-chunked[{i}:{i+chunk}]", q_BSHD.shape, k_BSJD.shape)
                raise
        return torch.cat(out_chunks, dim=0).permute(0, 2, 1, 3)
    _sdpa_log(q_BSHD.shape, k_BSJD.shape, "EFFICIENT/MATH")
    try:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    except torch.cuda.OutOfMemoryError:
        _sdpa_oom("EFFICIENT/MATH", q_BSHD.shape, k_BSJD.shape)
        raise
    return out.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Chunked evaluate
# ---------------------------------------------------------------------------


def chunked_evaluate_maybe_inplace(
    f: Callable,
    x: torch.Tensor,
    save_peak_memory_factor: int | None,
    residual: bool,
    batch_dims: int,
    *args,
    **kwargs,
) -> torch.Tensor:
    if save_peak_memory_factor is None:
        result = f(x.flatten(0, batch_dims - 1), *args, **kwargs).view(x.shape)
        return x + result if residual else result

    assert not x.requires_grad
    x_flat_batch = x.flatten(0, batch_dims - 1)
    split_size = (x_flat_batch.shape[0] + save_peak_memory_factor - 1) // save_peak_memory_factor
    for x_chunk in torch.split(x_flat_batch, split_size):
        if residual:
            x_chunk.add_(f(x_chunk, *args, **kwargs))
        else:
            x_chunk[:] = f(x_chunk, *args, **kwargs)
    return x


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def apply_rope(
    t: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    dtype = t.dtype
    seq_len = t.shape[-2]
    positions = torch.arange(seq_len, device=t.device, dtype=inv_freq.dtype)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = freqs.cos()
    sin = freqs.sin()
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        t_even = t[..., 0::2]
        t_odd = t[..., 1::2]
        t_rotated = torch.stack((-t_odd, t_even), dim=-1).flatten(-2)
    else:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        half = t.shape[-1] // 2
        t_rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return (t * cos + t_rotated * sin).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, theta: float = 10_000.0, interleaved: bool = False) -> None:
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.freqs = nn.Parameter(inv_freq, requires_grad=False)
        self.interleaved = interleaved

    def rotate_queries_or_keys(self, t_BHSD: torch.Tensor) -> torch.Tensor:
        return apply_rope(t_BHSD, self.freqs, interleaved=self.interleaved)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class _DtypeMatchingRMSNorm(nn.RMSNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.dtype != input.dtype:
            return F.rms_norm(input, self.normalized_shape, self.weight.to(input.dtype), self.eps)
        return super().forward(input)


# ---------------------------------------------------------------------------
# log(seqlen) helper
# ---------------------------------------------------------------------------


def _safe_log_seqlen(n: int | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(n, torch.Tensor):
        return n.to(torch.float32).clamp(min=1).log().to(dtype)
    one = torch.ones((), dtype=torch.float32, device=device)
    return (one * n).clamp(min=1).log().to(dtype)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TrainableOrthogonalEmbedding(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self._init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())

    def _init(self) -> None:
        weight = self.embedding.weight
        num_classes, embed_dim = weight.shape
        k = min(num_classes, embed_dim)
        q, _ = torch.linalg.qr(torch.randn(embed_dim, k))
        ortho_rows = q.T
        with torch.no_grad():
            weight[:k].copy_(ortho_rows)
            if num_classes > embed_dim:
                extra = torch.randn(num_classes - k, embed_dim)
                extra = extra / extra.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                weight[k:].copy_(extra)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Sequential):
    """Two-layer GELU feed-forward with zero-initialised output."""

    def __init__(self, emsize: int, dim_feedforward: int,
                 device=None, dtype=None) -> None:
        kw: dict = {"device": device, "dtype": dtype}
        linear2 = nn.Linear(dim_feedforward, emsize, bias=False, **kw)
        nn.init.zeros_(linear2.weight)
        super().__init__(
            nn.Linear(emsize, dim_feedforward, bias=False, **kw),
            nn.GELU(),
            linear2,
        )


class GatedMLP(nn.Module):
    """SwiGLU feed-forward: (Linear → SiLU) ⊙ Linear → Linear, zero-init output.

    Drop-in replacement for MLP with the same interface.  More expressive than the
    two-layer GELU variant at identical hidden dimension — the element-wise gate
    allows each neuron to be selectively suppressed based on input content.
    """

    def __init__(self, emsize: int, dim_feedforward: int,
                 device=None, dtype=None) -> None:
        super().__init__()
        kw: dict = {"device": device, "dtype": dtype}
        self.gate = nn.Linear(emsize, dim_feedforward, bias=False, **kw)
        self.up   = nn.Linear(emsize, dim_feedforward, bias=False, **kw)
        self.down = nn.Linear(dim_feedforward, emsize, bias=False, **kw)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# SoftmaxScalingMLP
# ---------------------------------------------------------------------------


class SoftmaxScalingMLP(nn.Module):
    """Query-aware attention scaling."""

    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        base_out_dim = num_heads * head_dim
        self.base_mlp = nn.Sequential(
            nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out_dim)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, head_dim)
        )
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q_BSHD: torch.Tensor, n: int) -> torch.Tensor:
        logn_11 = _safe_log_seqlen(n, q_BSHD.device, q_BSHD.dtype).reshape(1, 1)
        base_scales = self.base_mlp(logn_11).view(1, 1, self.num_heads, self.head_dim)
        modulation = 1 + torch.tanh(self.query_mlp(q_BSHD))
        return q_BSHD * base_scales * modulation


# ---------------------------------------------------------------------------
# SDPA wrapper
# ---------------------------------------------------------------------------


def _batched_scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor,
    v_BSJD: torch.Tensor,
    softmax_scaling_layer: nn.Module | None = None,
    _backends_override=None,
    attn_mask: torch.Tensor | None = None,
    allow_fa4: bool = False,
) -> torch.Tensor:
    if softmax_scaling_layer is not None:
        src_len = k_BSJD.shape[1]
        q_BSHD = softmax_scaling_layer(q_BSHD, src_len)
    cap = getattr(softmax_scaling_layer, "attn_logit_softcap", None)
    if cap is not None:
        # FA4 handles softcap natively (O(S) memory). Any D ≤ 256 is eligible:
        # scaled_dot_product_attention will zero-pad to the next valid FA4
        # head_dim (32/64/128/256), giving O(S) backward vs O(S²) for MATH.
        D = q_BSHD.shape[-1]
        q_h, k_h = q_BSHD.shape[2], k_BSJD.shape[2]
        _fa4_ok = (
            allow_fa4
            and _HAS_FLASH_ATTN_FAST and _fa_func is not None
            and attn_mask is None
            and q_BSHD.is_cuda
            and q_BSHD.dtype in (torch.float16, torch.bfloat16)
            and D <= 256
            and (q_h == k_h or q_h % k_h == 0)
        )
        if _fa4_ok:
            return scaled_dot_product_attention(
                q_BSHD, k_BSJD, v_BSJD, attn_mask=attn_mask, softcap=cap, allow_fa4=True)
        return soft_cap_sdpa(q_BSHD, k_BSJD, v_BSJD, cap, attn_mask=attn_mask, allow_fa4=allow_fa4)
    return scaled_dot_product_attention(q_BSHD, k_BSJD, v_BSJD, attn_mask=attn_mask, allow_fa4=allow_fa4)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, head_dim: int,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        kw: dict = {"device": device, "dtype": dtype, "bias": False}
        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.out_projection.weight)

    def forward(
        self,
        x_BSE: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = x_BSE.shape
        q = self.q_projection(x_BSE).view(B, S, -1, self.head_dim)
        k = self.k_projection(x_BSE).view(B, S, -1, self.head_dim)
        v = self.v_projection(x_BSE).view(B, S, -1, self.head_dim)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q.transpose(1, 2)).transpose(1, 2)
            k = rope.rotate_queries_or_keys(k.transpose(1, 2)).transpose(1, 2)
        out = _batched_scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).reshape(B, S, self.head_dim * self.num_heads)
        return self.out_projection(out)


class CrossAttention(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, head_dim: int,
                 softmax_scaling_layer: nn.Module | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer
        kw: dict = {"device": device, "dtype": dtype, "bias": False}
        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.out_projection.weight)

    def forward(self, x_for_query_BQE: torch.Tensor, x_for_key_and_value_BVE: torch.Tensor) -> torch.Tensor:
        B, Q, _ = x_for_query_BQE.shape
        _, V, _ = x_for_key_and_value_BVE.shape
        q = self.q_projection(x_for_query_BQE).view(B, Q, -1, self.head_dim)
        k = self.k_projection(x_for_key_and_value_BVE).view(B, V, -1, self.head_dim)
        v = self.v_projection(x_for_key_and_value_BVE).view(B, V, -1, self.head_dim)
        out = _batched_scaled_dot_product_attention(q, k, v, softmax_scaling_layer=self.softmax_scaling_layer)
        return self.out_projection(out.reshape(B, Q, self.head_dim * self.num_heads))


class ICLAttention(nn.Module):
    """ICL attention: all rows attend to train-only keys/values."""

    def __init__(self, embedding_size: int, num_heads: int, head_dim: int,
                 softmax_scaling_layer: nn.Module | None = None,
                 num_kv_heads: int | None = None,
                 num_kv_heads_test: int | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads_test = num_kv_heads_test
        kw: dict = {"device": device, "dtype": dtype, "bias": False}
        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.zeros_(self.out_projection.weight)
        kv_dim = self.num_kv_heads * head_dim
        self.k_projection = nn.Linear(embedding_size, kv_dim, **kw)
        self.v_projection = nn.Linear(embedding_size, kv_dim, **kw)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)

    def forward(self, x_BRE: torch.Tensor, single_eval_pos: int, *,
                cached_kv: KVCacheEntry | None = None,
                return_kv: bool = False) -> tuple[torch.Tensor, KVCacheEntry | None]:
        B, R, _ = x_BRE.shape
        q = self.q_projection(x_BRE).view(B, R, self.num_heads, self.head_dim)

        if cached_kv is not None:
            k = cached_kv.key.to(q.dtype)
            v = cached_kv.value.to(q.dtype)
            out = _batched_scaled_dot_product_attention(q, k, v, softmax_scaling_layer=self.softmax_scaling_layer, allow_fa4=True)
        else:
            N = R if single_eval_pos is None else single_eval_pos
            k = self.k_projection(x_BRE[:, :N]).view(B, N, self.num_kv_heads, self.head_dim)
            v = self.v_projection(x_BRE[:, :N]).view(B, N, self.num_kv_heads, self.head_dim)
            if self.num_kv_heads_test is not None and single_eval_pos is not None and N < R:
                out_train = _batched_scaled_dot_product_attention(q[:, :N], k, v, softmax_scaling_layer=self.softmax_scaling_layer, allow_fa4=True)
                nh = self.num_kv_heads_test
                out_test = _batched_scaled_dot_product_attention(q[:, N:], k[:, :, :nh], v[:, :, :nh], softmax_scaling_layer=self.softmax_scaling_layer, allow_fa4=True)
                out = torch.cat([out_train, out_test], dim=1)
            else:
                out = _batched_scaled_dot_product_attention(q, k, v, softmax_scaling_layer=self.softmax_scaling_layer, allow_fa4=True)

        result = self.out_projection(out.reshape(B, R, self.head_dim * self.num_heads))
        kv_entry: KVCacheEntry | None = None
        if return_kv:
            kv_entry = KVCacheEntry(key=k.detach(), value=v.detach())
        return result, kv_entry


# ---------------------------------------------------------------------------
# Many-class decoder
# ---------------------------------------------------------------------------


def _chunked_class_attention(
    q_BSHD: torch.Tensor,
    k_BJHD: torch.Tensor,
    v_BJHT: torch.Tensor,
    softmax_scaling_layer: nn.Module | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    B, S, H, D = q_BSHD.shape
    T = v_BJHT.shape[-1]
    num_chunks = math.ceil(T / D)
    pad = num_chunks * D - T
    if pad > 0:
        v_BJHT = F.pad(v_BJHT, (0, pad))
    J = v_BJHT.shape[1]
    v_folded = v_BJHT.reshape(B, J, H, num_chunks, D).permute(0, 3, 1, 2, 4).reshape(B * num_chunks, J, H, D).contiguous()
    q_folded = q_BSHD.unsqueeze(1).expand(-1, num_chunks, -1, -1, -1).reshape(B * num_chunks, S, H, D).contiguous()
    k_folded = k_BJHD.unsqueeze(1).expand(-1, num_chunks, -1, -1, -1).reshape(B * num_chunks, J, H, D).contiguous()
    mask_folded: torch.Tensor | None = None
    if attn_mask is not None:
        mask_folded = attn_mask.unsqueeze(1).expand(-1, num_chunks, -1, -1, -1).reshape(
            B * num_chunks, *attn_mask.shape[1:]
        )
    out_folded = _batched_scaled_dot_product_attention(q_folded, k_folded, v_folded,
                                                       softmax_scaling_layer=softmax_scaling_layer,
                                                       attn_mask=mask_folded)
    return out_folded.reshape(B, num_chunks, S, H, D).permute(0, 2, 3, 1, 4).reshape(B, S, H, num_chunks * D)[..., :T]


class ManyClassDecoder(nn.Module):
    def __init__(self, max_num_classes: int, input_size: int,
                 head_dim: int = 64, num_heads: int = 6,
                 softmax_scaling_layer: nn.Module | None = None) -> None:
        super().__init__()
        self.max_num_classes = max_num_classes
        self.input_size = input_size
        self.attention_size = head_dim * num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q_projection = nn.Linear(input_size, self.attention_size)
        self.k_projection = nn.Linear(input_size, self.attention_size)
        self.softmax_scaling_layer = softmax_scaling_layer

    def forward(self, train_embeddings: torch.Tensor, test_embeddings: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        B, M, _ = test_embeddings.shape
        q_BME = self.q_projection(test_embeddings)
        if train_embeddings.dtype != q_BME.dtype:
            train_embeddings = train_embeddings.to(q_BME.dtype)
        k_BNE = self.k_projection(train_embeddings)
        if M == 0:
            empty = test_embeddings.new_empty((0, B, self.max_num_classes))
            return empty + (q_BME.sum() + k_BNE.sum()) * 0.0
        one_hot_BNT = F.one_hot(targets.long(), num_classes=self.max_num_classes).to(q_BME.dtype).contiguous()
        q_BMHD = q_BME.view(B, M, self.num_heads, self.head_dim).contiguous()
        k_BNHD = k_BNE.view(B, -1, self.num_heads, self.head_dim).contiguous()
        one_hot_BNHT = one_hot_BNT.unsqueeze(2).expand(-1, -1, self.num_heads, -1).contiguous()
        test_output_BMHT = _chunked_class_attention(q_BMHD, k_BNHD, one_hot_BNHT, softmax_scaling_layer=self.softmax_scaling_layer)
        test_output_BMT = test_output_BMHT.mean(2)
        test_output_MBT = test_output_BMT.transpose(0, 1)
        return torch.log(torch.clamp(test_output_MBT, min=1e-5) + 3e-5)


# ---------------------------------------------------------------------------
# MLPClassDecoder
# ---------------------------------------------------------------------------


class MLPClassDecoder(nn.Module):
    """MLP-based class decoder — replaces attention with a direct linear head.

    The ICL transformer blocks already mix train-context information into the
    test-row embeddings, so a two-layer MLP mapping D → hidden_dim → T is
    sufficient to produce class logits.

    Accepts the same positional arguments as ClassNormalizedManyClassDecoder
    (train_embeddings, targets, key_padding_mask) so the call-site in model.py
    is unchanged; those arguments are unused here.
    ``softmax_scaling_layer`` is also accepted but ignored.
    """

    def __init__(self, max_num_classes: int, input_size: int,
                 hidden_dim: int = 384,
                 softmax_scaling_layer=None) -> None:
        super().__init__()
        self.max_num_classes = max_num_classes
        self.norm = nn.LayerNorm(input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_num_classes),
        )

    def forward(self, train_embeddings: torch.Tensor, test_embeddings: torch.Tensor,
                targets: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            train_embeddings: (B, N, D)  — unused; accepted for API compatibility
            test_embeddings:  (B, M, D)
            targets:          (B, N)     — unused
            key_padding_mask: (B, N)     — unused

        Returns:
            log_probs: (M, B, T)
        """
        B, M, _ = test_embeddings.shape

        if M == 0:
            T = self.max_num_classes
            empty = test_embeddings.new_empty((0, B, T))
            return empty + self.mlp[0].weight.sum() * 0.0

        logits_BMT = self.mlp(self.norm(test_embeddings))   # (B, M, T)
        log_probs_BMT = F.log_softmax(logits_BMT, dim=-1)
        return log_probs_BMT.transpose(0, 1)                # (M, B, T)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    def __init__(self, *, emsize: int, nhead: int, dim_feedforward: int,
                 norm_factory: Callable[[int], nn.Module],
                 softmax_scaling_layer: nn.Module | None = None,
                 mlp_factory: Callable | None = None,
                 post_norm_factory: Callable[[int], nn.Module] | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        assert emsize % nhead == 0
        kw: dict = {"device": device, "dtype": dtype}
        self.attn = CrossAttention(emsize, nhead, emsize // nhead, softmax_scaling_layer, **kw)
        mlp_cls = mlp_factory if mlp_factory is not None else MLP
        self.mlp = mlp_cls(emsize, dim_feedforward, **kw)
        self.layernorm_q = norm_factory(emsize)
        self.layernorm_kv = norm_factory(emsize)
        self.layernorm2 = norm_factory(emsize)
        self.post_norm_attn = post_norm_factory(emsize) if post_norm_factory is not None else None
        self.post_norm_mlp = post_norm_factory(emsize) if post_norm_factory is not None else None

    def forward(self, x_BQE: torch.Tensor, context_BVE: torch.Tensor) -> torch.Tensor:
        x_BQE = x_BQE + self.attn(self.layernorm_q(x_BQE), self.layernorm_kv(context_BVE))
        if self.post_norm_attn is not None:
            x_BQE = self.post_norm_attn(x_BQE)
        x_BQE = x_BQE + self.mlp(self.layernorm2(x_BQE))
        if self.post_norm_mlp is not None:
            x_BQE = self.post_norm_mlp(x_BQE)
        return x_BQE


class ICLTransformerBlock(nn.Module):
    def __init__(self, *, emsize: int, nhead: int, dim_feedforward: int,
                 norm_factory: Callable[[int], nn.Module],
                 softmax_scaling_layer: nn.Module | None = None,
                 num_kv_heads: int | None = None,
                 num_kv_heads_test: int | None = None,
                 post_norm_factory: Callable[[int], nn.Module] | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        assert emsize % nhead == 0
        kw: dict = {"device": device, "dtype": dtype}
        self.icl_attention = ICLAttention(emsize, nhead, emsize // nhead,
                                          softmax_scaling_layer, num_kv_heads,
                                          num_kv_heads_test, **kw)
        self.layernorm = norm_factory(emsize)
        self.layernorm_mlp = norm_factory(emsize)
        self.mlp = MLP(emsize, dim_feedforward, **kw)
        self.post_norm_attn = post_norm_factory(emsize) if post_norm_factory is not None else None
        self.post_norm_mlp = post_norm_factory(emsize) if post_norm_factory is not None else None

    def forward(self, x_BRE: torch.Tensor, single_eval_pos: int,
                save_peak_memory_factor: int | None = None, *,
                cached_kv: KVCacheEntry | None = None,
                return_kv: bool = False) -> tuple[torch.Tensor, KVCacheEntry | None]:
        kv_entry: KVCacheEntry | None = None
        if return_kv:
            attn_out, kv_entry = self.icl_attention(self.layernorm(x_BRE), single_eval_pos=single_eval_pos, return_kv=True)
            x_BRE = x_BRE + attn_out
        elif cached_kv is not None:
            def _attn_fn_cached(x: torch.Tensor, single_eval_pos: int | None = None) -> torch.Tensor:
                out, _ = self.icl_attention(self.layernorm(x), single_eval_pos=single_eval_pos, cached_kv=cached_kv)
                return out
            x_BRE = chunked_evaluate_maybe_inplace(_attn_fn_cached, x_BRE, save_peak_memory_factor, residual=True, batch_dims=1, single_eval_pos=single_eval_pos)
        else:
            def _attn_fn(x: torch.Tensor, single_eval_pos: int | None = None) -> torch.Tensor:
                out, _ = self.icl_attention(self.layernorm(x), single_eval_pos=single_eval_pos)
                return out
            x_BRE = chunked_evaluate_maybe_inplace(_attn_fn, x_BRE, save_peak_memory_factor, residual=True, batch_dims=1, single_eval_pos=single_eval_pos)
        if self.post_norm_attn is not None:
            x_BRE = self.post_norm_attn(x_BRE)
        x_BRE = chunked_evaluate_maybe_inplace(lambda x: self.mlp(self.layernorm_mlp(x)), x_BRE, save_peak_memory_factor, residual=True, batch_dims=2)
        if self.post_norm_mlp is not None:
            x_BRE = self.post_norm_mlp(x_BRE)
        return x_BRE, kv_entry


# ---------------------------------------------------------------------------
# Induced self-attention block (SetTransformer-style)
# ---------------------------------------------------------------------------


class InducedSelfAttentionBlock(nn.Module):
    def __init__(self, *, emsize: int, nhead: int, num_inducing_points: int,
                 dim_feedforward: int, norm_factory: Callable[[int], nn.Module],
                 softmax_scaling_layer: nn.Module | None = None,
                 mlp_factory: Callable | None = None,
                 post_norm_factory: Callable[[int], nn.Module] | None = None,
                 use_data_inducing: bool = False,
                 inducing_coreset_method: str = "random",
                 device=None, dtype=None) -> None:
        super().__init__()
        kw: dict = {"device": device, "dtype": dtype}
        block_kw = {"emsize": emsize, "nhead": nhead, "dim_feedforward": dim_feedforward,
                    "norm_factory": norm_factory, "mlp_factory": mlp_factory,
                    "post_norm_factory": post_norm_factory, **kw}
        self.cross_attn_block1 = CrossAttentionBlock(**block_kw, softmax_scaling_layer=softmax_scaling_layer)
        self.cross_attn_block2 = CrossAttentionBlock(**block_kw)
        self.num_inducing_points = num_inducing_points
        self.use_data_inducing = use_data_inducing
        self.inducing_coreset_method = inducing_coreset_method
        self.inducing_vectors = nn.Parameter(torch.empty(num_inducing_points, emsize))
        nn.init.trunc_normal_(self.inducing_vectors, std=0.02)

    def _get_inducing_queries(self, x_BcRE: torch.Tensor, N: int) -> torch.Tensor:
        """Return inducing query tensor (Bc, M, E).

        use_data_inducing=False (default): fixed learned vectors, unchanged.
        use_data_inducing=True: sample M rows from the N training rows.
          coreset_method="random":     uniform random sample (randperm[:M]).
          coreset_method="stratified": divide [0,N) into M equal buckets,
            sample one random row per bucket → guaranteed uniform N-coverage.
            O(M) extra cost vs random; no distance computation needed.
          Falls back to learned vectors when N < num_inducing_points.
        """
        if not self.use_data_inducing or N < self.num_inducing_points:
            return self.inducing_vectors.unsqueeze(0).expand(x_BcRE.shape[0], -1, -1)
        M = self.num_inducing_points
        if self.inducing_coreset_method == "stratified":
            stride = N // M
            bucket_starts = torch.arange(M, device=x_BcRE.device) * stride
            offsets = torch.randint(0, max(1, stride), (M,), device=x_BcRE.device)
            idx = (bucket_starts + offsets).clamp(0, N - 1)
        else:
            idx = torch.randperm(N, device=x_BcRE.device)[:M]
        return x_BcRE[:, idx]   # (Bc, M, E) — data-dependent, fresh each call

    def _induced_attention(self, x_BcRE: torch.Tensor,
                           single_eval_pos: int | None = None,
                           cached_hidden: torch.Tensor | None = None, *,
                           return_hidden: bool = False,
                           inducing_bias: torch.Tensor | None = None):
        if cached_hidden is not None:
            hidden = cached_hidden.to(x_BcRE.dtype)
        else:
            Bc, R, _ = x_BcRE.shape
            N = R if single_eval_pos is None else single_eval_pos
            ind = self._get_inducing_queries(x_BcRE, N)
            if inducing_bias is not None:
                ind = ind + inducing_bias
            hidden = self.cross_attn_block1(ind, x_BcRE[:, :N])
        out = self.cross_attn_block2(x_BcRE, hidden)
        if return_hidden:
            return out, hidden.detach()
        return out

    def forward(self, x_BRCE: torch.Tensor, single_eval_pos: int | None = None,
                save_peak_memory_factor: int | None = None, *,
                cached_hidden: torch.Tensor | None = None,
                return_hidden: bool = False,
                inducing_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, R, C, E = x_BRCE.shape
        x_BcRE = x_BRCE.transpose(1, 2).contiguous().reshape(B * C, R, E)
        if return_hidden or cached_hidden is not None or inducing_bias is not None:
            out_BcRE, hidden = self._induced_attention(
                x_BcRE, single_eval_pos=single_eval_pos,
                cached_hidden=cached_hidden, return_hidden=True,
                inducing_bias=inducing_bias,
            )
            if not return_hidden:
                hidden = None
        else:
            out_BcRE = chunked_evaluate_maybe_inplace(
                self._induced_attention, x_BcRE, save_peak_memory_factor,
                residual=False, batch_dims=1,
                single_eval_pos=single_eval_pos, cached_hidden=None,
            )
            hidden = None
        out_BRCE = out_BcRE.reshape(B, C, R, E).transpose(1, 2).contiguous()
        return out_BRCE, hidden


# ---------------------------------------------------------------------------
# PMA block (Pooling by Multihead Attention)
# ---------------------------------------------------------------------------


class PMABlock(nn.Module):
    """PMA applied per column: Pool → Refine (SAB) → Read.

    Three stages operating on the (B*C, R, E) per-column view:
      1. Pool   — K seed vectors cross-attend to train rows only.
      2. Refine — seeds self-attend (SAB) so slots share information before
                  broadcasting.  Key difference from ISAB: inter-seed
                  interaction before the read step.
      3. Read   — all rows (train + test) cross-attend to refined seeds,
                  receiving a rich distributional context.

    Input / output shape: (B, R, C, E) → (B, R, C, E)
    """

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        num_seeds: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer: nn.Module | None = None,
        post_norm_factory: Callable[[int], nn.Module] | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kw: dict = {"device": device, "dtype": dtype}
        block_kw = {
            "emsize": emsize,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "norm_factory": norm_factory,
            "post_norm_factory": post_norm_factory,
            **kw,
        }
        # Step 1: seeds pool from train rows
        self.pool_attn = CrossAttentionBlock(**block_kw, softmax_scaling_layer=softmax_scaling_layer)
        # Step 2: seeds self-attend (SAB)
        self.sab = CrossAttentionBlock(**block_kw)
        # Step 3: rows read from refined seeds
        self.read_attn = CrossAttentionBlock(**block_kw)
        self.seeds = nn.Parameter(torch.empty(num_seeds, emsize, **kw))
        nn.init.trunc_normal_(self.seeds, std=0.02)

    def forward(self, x_BRCE: torch.Tensor, num_train: int) -> torch.Tensor:
        B, R, C, E = x_BRCE.shape
        x_BcRE = x_BRCE.transpose(1, 2).contiguous().reshape(B * C, R, E)

        # Step 1: pool — seeds cross-attend to train rows only
        seeds = self.seeds.unsqueeze(0).expand(B * C, -1, -1)  # (B*C, K, E)
        pooled = self.pool_attn(seeds, x_BcRE[:, :num_train])  # (B*C, K, E)

        # Step 2: refine — SAB (seeds self-attend)
        refined = self.sab(pooled, pooled)                      # (B*C, K, E)

        # Step 3: read — all rows cross-attend to refined seeds
        out_BcRE = self.read_attn(x_BcRE, refined)             # (B*C, R, E)

        return out_BcRE.reshape(B, C, R, E).transpose(1, 2).contiguous()
