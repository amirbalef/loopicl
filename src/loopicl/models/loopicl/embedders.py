"""Axial embedder and cross-column block for LoopICL.

Contains:
  - _CrossColBlock: cross-column self-attention with RoPE + SoftmaxScaling
  - AxialEmbedderV3A: interleaved within-col / cross-col attention (no back-projection)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    Attention,
    CrossAttentionBlock,
    GatedMLP,
    InducedSelfAttentionBlock,
    MLP,
    RotaryEmbedding,
    SoftmaxScalingMLP,
    _DtypeMatchingRMSNorm,
    _batched_scaled_dot_product_attention,
)
from .conditioning import MaskedICLTransformerBlock


# ---------------------------------------------------------------------------
# _CrossColBlock
# ---------------------------------------------------------------------------


class _CrossColBlock(nn.Module):
    """Cross-column self-attention with RoPE and SoftmaxScaling.

    Applied per row over the C dimension. Takes (B, R, C, E) in, returns same shape.
    """

    def __init__(
        self,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        norm_factory,
        softmax_scaling_layer: nn.Module | None = None,
        rope_base: float = 100_000,
        mlp_factory=None,
        post_norm_factory=None,
    ) -> None:
        super().__init__()
        assert emsize % nhead == 0
        self.attention = Attention(emsize, nhead, emsize // nhead)
        self.softmax_scaling_layer = softmax_scaling_layer
        self.rope = RotaryEmbedding(dim=emsize // nhead, theta=int(rope_base))
        self.layernorm = norm_factory(emsize)
        self.layernorm_mlp = norm_factory(emsize)
        mlp_cls = mlp_factory if mlp_factory is not None else MLP
        self.mlp = mlp_cls(emsize, dim_feedforward)
        self.post_norm_attn = post_norm_factory(emsize) if post_norm_factory is not None else None
        self.post_norm_mlp = post_norm_factory(emsize) if post_norm_factory is not None else None

    def forward(
        self,
        x_BRCE: torch.Tensor,
        residual_alpha: torch.Tensor | None = None,
        residual_beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, R, C, E = x_BRCE.shape
        H, D = self.attention.num_heads, self.attention.head_dim
        x_flat = x_BRCE.flatten(0, 1)  # (B*R, C, E)

        def _residual(x, delta):
            if residual_alpha is None:
                return x + delta
            b = residual_beta if residual_beta is not None else (1 - residual_alpha)
            return residual_alpha * x + b * delta

        normed = self.layernorm(x_flat)
        q = self.attention.q_projection(normed).view(B * R, C, H, D)
        k = self.attention.k_projection(normed).view(B * R, C, H, D)
        v = self.attention.v_projection(normed).view(B * R, C, H, D)
        q = self.rope.rotate_queries_or_keys(q.transpose(1, 2)).transpose(1, 2)
        k = self.rope.rotate_queries_or_keys(k.transpose(1, 2)).transpose(1, 2)
        attn_out = _batched_scaled_dot_product_attention(
            q, k, v, softmax_scaling_layer=self.softmax_scaling_layer
        )
        attn_out = self.attention.out_projection(attn_out.reshape(B * R, C, H * D))
        x_flat = _residual(x_flat, attn_out)
        if self.post_norm_attn is not None:
            x_flat = self.post_norm_attn(x_flat)

        x_flat = _residual(x_flat, self.mlp(self.layernorm_mlp(x_flat)))
        if self.post_norm_mlp is not None:
            x_flat = self.post_norm_mlp(x_flat)

        return x_flat.view(B, R, C, E)


# ---------------------------------------------------------------------------
# AxialEmbedderV3A
# ---------------------------------------------------------------------------


class AxialEmbedderV3A(nn.Module):
    """Interleaved within-col / cross-col attention without back-projection.

    Per axial block:
      1. Label re-injection into training rows  (zero-init)
      2. Within-column induced self-attention   (B*C, R, E)
      3. Cross-column attention + RoPE           (B*R, C, E)
      4. Readout: CLS cross-attention           (B*R, C→Cl, E) → accumulate x_BRD
      5. pre_icl_ln on x_BRD
      6. ICL label injection                   (zero-init)
      7. num_icl_per_round MaskedICLTransformerBlocks
    """

    def __init__(
        self,
        emsize: int,
        nhead_within: int,
        nhead_cross: int,
        num_inducing_points: int,
        num_axial_blocks: int,
        num_within_per_round: int,
        num_cross_per_round: int,
        num_icl_per_round: int,
        num_cls_tokens: int,
        dim_feedforward: int,
        norm_factory,
        icl_block_factory,
        within_softmax_scaling_factory=None,
        cross_softmax_scaling_factory=None,
        rope_base: float = 100_000,
        mlp_factory=None,
        post_norm_factory=None,
        use_data_inducing: bool = False,
        inducing_coreset_method: str = "random",
    ) -> None:
        super().__init__()
        self.num_axial_blocks    = num_axial_blocks
        self.num_within_per_round = num_within_per_round
        self.num_cross_per_round  = num_cross_per_round
        self.num_icl_per_round   = num_icl_per_round
        self.num_cls_tokens      = num_cls_tokens

        # ---- cell stream ----
        self.within_col_blocks = nn.ModuleList([
            InducedSelfAttentionBlock(
                emsize=emsize, nhead=nhead_within,
                num_inducing_points=num_inducing_points,
                dim_feedforward=dim_feedforward, norm_factory=norm_factory,
                softmax_scaling_layer=(within_softmax_scaling_factory() if within_softmax_scaling_factory else None),
                mlp_factory=mlp_factory,
                post_norm_factory=post_norm_factory,
                use_data_inducing=use_data_inducing,
                inducing_coreset_method=inducing_coreset_method,
            )
            for _ in range(num_axial_blocks * num_within_per_round)
        ])
        self.cross_col_blocks = nn.ModuleList([
            _CrossColBlock(
                emsize=emsize, nhead=nhead_cross,
                dim_feedforward=dim_feedforward, norm_factory=norm_factory,
                softmax_scaling_layer=(cross_softmax_scaling_factory() if cross_softmax_scaling_factory else None),
                rope_base=rope_base,
                mlp_factory=mlp_factory,
                post_norm_factory=post_norm_factory,
            )
            for _ in range(num_axial_blocks * num_cross_per_round)
        ])
        # per-round cell-stream label re-injection (zero-init → no-op at init)
        self.label_inject_projs = nn.ModuleList([
            nn.Linear(emsize, emsize, bias=False) for _ in range(num_axial_blocks)
        ])
        for proj in self.label_inject_projs:
            nn.init.zeros_(proj.weight)

        # ---- readout: cell → row ----
        self.cls_tokens = nn.Parameter(torch.empty(num_cls_tokens, emsize))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.readout_blocks = nn.ModuleList([
            CrossAttentionBlock(emsize=emsize, nhead=nhead_cross,
                                dim_feedforward=dim_feedforward, norm_factory=norm_factory,
                                mlp_factory=mlp_factory, post_norm_factory=post_norm_factory)
            for _ in range(num_axial_blocks)
        ])
        self.row_ln     = norm_factory(num_cls_tokens * emsize)
        self.pre_icl_lns = nn.ModuleList([
            norm_factory(num_cls_tokens * emsize) for _ in range(num_axial_blocks)
        ])

        # ---- row stream ----
        D = num_cls_tokens * emsize
        self.icl_label_projs = nn.ModuleList([
            nn.Linear(emsize, D, bias=False) for _ in range(num_axial_blocks)
        ])
        for proj in self.icl_label_projs:
            nn.init.zeros_(proj.weight)

        total_icl = num_axial_blocks * num_icl_per_round
        self.icl_blocks = nn.ModuleList([icl_block_factory() for _ in range(total_icl)])
