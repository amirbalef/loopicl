"""LoopICL — looped axial in-context learning model for tabular classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson

from .conditioning import (
    AugmentedLogHistConditioner,
    ClassNormalizedManyClassDecoder,
    DiscriminativeHistConditioner,
    FourierQuantileEncoderWithOOD,
    MaskedICLTransformerBlock,
    _N_HIST_BINS,
    _adaptive_bin_geometry,
    _compute_moments,
    _soft_log_histogram,
)
from .embedders import AxialEmbedderV3A
from .layers import (
    CrossAttentionBlock,
    ManyClassDecoder,
    MLPClassDecoder,
    SoftmaxScalingMLP,
    SoftmaxScalingMLPv2,
    TrainableOrthogonalEmbedding,
    _DtypeMatchingRMSNorm,
)
from .kv_cache import LoopICLTrainCache
from .scaler import TorchStandardScaler

__all__ = [
    "LoopICL",
    "LoopICLConfig",
    "LoopICLTrainCache",
    "LoopICLUnrolled",
    "build_loopicl",
    "build_loopicl_oneway",
]

_NAN_INDICATOR      = -2.0
_INFINITY_INDICATOR =  2.0
_NEG_INF_INDICATOR  =  4.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoopICLConfig:
    """Flat configuration for LoopICL.
    """

    # ---- General ----
    max_num_classes: int = 10

    # ---- Cell embedding ----
    embed_dim: int = 128
    feature_group_size: int = 3

    # ---- Within-column attention ----
    dist_embed_num_heads: int = 8
    dist_embed_num_inducing_points: int = 128

    # ---- Cross-column attention ----
    feat_agg_num_heads: int = 8
    feat_agg_num_cls_tokens: int = 4
    feat_agg_rope_base: float = 100_000.0

    # ---- ICL blocks ----
    icl_num_heads: int = 8
    icl_num_kv_heads: Optional[int] = None
    icl_num_kv_heads_test: Optional[int] = None

    # ---- Decoder ----
    decoder_head_dim: int = 64
    decoder_num_heads: int = 6
    decoder_use_softmax_scaling: bool = False
    # MLP decoder (replaces attention-based ClassNormalizedManyClassDecoder)
    use_mlp_decoder: bool = False
    decoder_mlp_hidden_dim: int = 384
    use_class_normalized_decoder: bool = True

    # ---- Shared ----
    ff_factor: int = 2
    softmax_scaling_mlp_hidden_dim: int = 64
    use_v2_softmax_scaling: bool = False
    softmax_scaling_n_ref: int = 512
    softmax_scaling_use_tanh: bool = False  # applies tanh to log(n/n_ref) input (v2 only)
    attn_logit_softcap: Optional[float] = None
    use_data_inducing: bool = False  # sample training rows as within-col inducing queries
    layernorm_elementwise_affine: bool = True
    use_nan_indicators: bool = True
    # Sandwich normalisation (tabfm-style stable pre+post norm) — on by default:
    #   x = post_norm(x + sublayer(pre_norm(x)))
    use_sandwich_norm: bool = True

    # ---- Axial architecture ----
    axial_num_blocks: int = 1
    num_within_per_round: int = 1
    num_cross_per_round: int = 1
    num_icl_per_round: int = 1

    # ---- Outer loop ----
    num_icl_loops: int = 4
    icl_random_loops: bool = True
    icl_random_loops_mu: Optional[float] = None
    icl_random_loops_sigma: float = 0.5
    icl_random_loops_max: Optional[int] = 6
    iter_embed: bool = False
    inject_only_first_loop: bool = False

    # ---- Memory / speed ----
    loop_gradient_checkpointing: bool = False
    tbptt_last_k: Optional[int] = None  # truncated BPTT: only backprop through last k loops

    # ---- Loop residual scaling (arXiv 2606.18524) ----
    use_loop_residual_scaling: bool = False
    loop_residual_lambda: float = 1.0
    # Scaling function applied to n_loops.  All options keep the 1/sqrt(L)
    # layers-per-loop normalisation; only the n-dependent part differs:
    #   "inv_n"       — λ / (n · √L)                      [default / original]
    #   "inv_sqrt_n"  — λ / (√n · √L)
    #   "constant"    — λ / √L                             (no loop-count dependence)
    #   "inv_n2"      — λ / (n² · √L)
    #   "log_n"       — λ · log(n) / (n · √L)
    #   "cosine"      — (λ/√L) · ½(1 + cos(π·t/n))        (per-step, decays with loop_i)
    #   "custom"      — calls model.loop_residual_custom_fn(n_loops, loop_i) -> float
    #   "learned_mlp" — shared 2→1 MLP applied per embedding channel:
    #                   input  = [t, log_rms_channel]  (position + per-channel residual magnitude)
    #                   output = scale multiplier for that channel
    #                   init so multiplier=1 → identical to inv_n at the start of training.
    loop_residual_scaling_fn: str = "inv_n"

    # ---- One-way column→row attention ----
    use_oneway: bool = False

    # ---- Within-col inducing coreset method ----
    # "random":     uniform random sample (default, O(1) extra cost).
    # "stratified": divide [0,N) into M equal buckets, sample one per bucket.
    #   Guarantees uniform coverage of the training sequence at O(M) cost.
    #   Use with use_data_inducing=True for large N.
    inducing_coreset_method: str = "random"

    # ---- Long-context histogram conditioning ----
    # use_disc_hist_conditioning: disable DiscriminativeHistConditioner to save
    #   memory at very large N (the chunked path already handles N up to ~100k).
    # disc_hist_chunk_size: chunk size for DiscriminativeHistConditioner's N-loop.
    #   At N=100k, C=100, K=32: full (B,C,N,K) tensor ≈ 1.2 GB → chunked to O(chunk).
    use_log_hist_conditioning: bool = True
    use_disc_hist_conditioning: bool = True
    disc_hist_chunk_size: int = 2048
    use_fourier_ood: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LoopICL(nn.Module):
    """LoopICL — looped axial in-context learning model.

    Sandwich normalisation is enabled by default.  Fully standalone
    implementation: no dependency on other model packages.

    Example
    -------
    >>> cfg = LoopICLConfig(embed_dim=256, num_icl_loops=6)
    >>> model = LoopICL(cfg)

    >>> model = LoopICL(embed_dim=256, num_icl_loops=6)
    """

    def __init__(self, config: LoopICLConfig | None = None, **kwargs):
        super().__init__()
        if config is not None:
            merged = {**vars(config), **kwargs}
        else:
            merged = dict(kwargs)
        cfg = LoopICLConfig(**{k: v for k, v in merged.items()
                                if k in LoopICLConfig.__dataclass_fields__})
        self._build(cfg)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, cfg: LoopICLConfig) -> None:
        self.cfg = cfg
        T  = cfg.max_num_classes
        E  = cfg.embed_dim
        Cl = cfg.feat_agg_num_cls_tokens
        D  = Cl * E
        G  = cfg.feature_group_size
        n_hidden   = cfg.softmax_scaling_mlp_hidden_dim
        self.feature_group_size = G
        self.use_nan_indicators  = cfg.use_nan_indicators
        self.standard_scaler     = TorchStandardScaler()

        norm_factory = partial(
            _DtypeMatchingRMSNorm,
            elementwise_affine=cfg.layernorm_elementwise_affine,
        )
        post_norm_factory = norm_factory if cfg.use_sandwich_norm else None

        _scaling_cls = SoftmaxScalingMLPv2 if cfg.use_v2_softmax_scaling else SoftmaxScalingMLP
        _scaling_kw: dict = (
            {
                "n_ref": cfg.softmax_scaling_n_ref,
                "attn_logit_softcap": cfg.attn_logit_softcap,
                "use_tanh_log_ratio": cfg.softmax_scaling_use_tanh,
            }
            if cfg.use_v2_softmax_scaling else {}
        )
        def _make_scaling(num_heads: int, head_dim: int) -> nn.Module:
            return _scaling_cls(num_heads, head_dim, n_hidden, **_scaling_kw)

        # ---- cell embedding ----
        in_features = G * (2 if cfg.use_nan_indicators else 1)
        self.x_embed = nn.Linear(in_features, E)

        # ---- label encoder ----
        self.col_y_encoder = TrainableOrthogonalEmbedding(T, E)

        # ---- histogram conditioning (embedding stage) ----
        self._n_bins = _N_HIST_BINS
        self.use_log_hist_conditioning = cfg.use_log_hist_conditioning
        if cfg.use_log_hist_conditioning:
            self.log_hist_conditioner        = AugmentedLogHistConditioner(E, n_hidden, _N_HIST_BINS)
        self.discriminative_hist_conditioner = DiscriminativeHistConditioner(E, T, n_hidden, _N_HIST_BINS)
        self.use_disc_hist_conditioning = cfg.use_disc_hist_conditioning
        self._disc_hist_chunk_size      = cfg.disc_hist_chunk_size
        self.use_fourier_ood = cfg.use_fourier_ood
        if cfg.use_fourier_ood:
            self.fourier_quantile_encoder = FourierQuantileEncoderWithOOD(E, n_hidden=n_hidden)

        # ---- output norm + decoder ----
        self.output_norm = norm_factory(D)
        if cfg.use_mlp_decoder:
            self.many_class_decoder = MLPClassDecoder(
                max_num_classes=T, input_size=D,
                hidden_dim=cfg.decoder_mlp_hidden_dim,
            )
        else:
            decoder_scaling = (
                _make_scaling(cfg.decoder_num_heads, cfg.decoder_head_dim)
                if cfg.decoder_use_softmax_scaling else None
            )
            decoder_cls = ClassNormalizedManyClassDecoder if cfg.use_class_normalized_decoder else ManyClassDecoder
            self.many_class_decoder = decoder_cls(
                max_num_classes=T, input_size=D,
                head_dim=cfg.decoder_head_dim, num_heads=cfg.decoder_num_heads,
                softmax_scaling_layer=decoder_scaling,
            )

        # ---- axial embedder ----
        def _icl_block():
            return MaskedICLTransformerBlock(
                emsize=D, nhead=cfg.icl_num_heads,
                dim_feedforward=D * cfg.ff_factor,
                norm_factory=norm_factory,
                num_kv_heads=cfg.icl_num_kv_heads,
                num_kv_heads_test=cfg.icl_num_kv_heads_test,
                softmax_scaling_layer=_make_scaling(cfg.icl_num_heads, D // cfg.icl_num_heads),
                post_norm_factory=post_norm_factory,
            )

        self.axial_embedder = AxialEmbedderV3A(
            emsize=E,
            nhead_within=cfg.dist_embed_num_heads,
            nhead_cross=cfg.feat_agg_num_heads,
            num_inducing_points=cfg.dist_embed_num_inducing_points,
            num_axial_blocks=cfg.axial_num_blocks,
            num_within_per_round=cfg.num_within_per_round,
            num_cross_per_round=cfg.num_cross_per_round,
            num_icl_per_round=cfg.num_icl_per_round,
            num_cls_tokens=Cl,
            dim_feedforward=E * cfg.ff_factor,
            norm_factory=norm_factory,
            icl_block_factory=_icl_block,
            within_softmax_scaling_factory=lambda: _make_scaling(cfg.dist_embed_num_heads, E // cfg.dist_embed_num_heads),
            cross_softmax_scaling_factory=lambda: _make_scaling(cfg.feat_agg_num_heads, E // cfg.feat_agg_num_heads),
            rope_base=cfg.feat_agg_rope_base,
            post_norm_factory=post_norm_factory,
            use_data_inducing=cfg.use_data_inducing,
            inducing_coreset_method=cfg.inducing_coreset_method,
        )

        # ---- outer loop settings ----
        self.num_icl_loops       = cfg.num_icl_loops
        self.icl_random_loops    = cfg.icl_random_loops
        self.icl_random_loops_mu = (
            float(cfg.icl_random_loops_mu) if cfg.icl_random_loops_mu is not None
            else float(cfg.num_icl_loops - 1)
        )
        self.icl_random_loops_sigma = cfg.icl_random_loops_sigma
        self.icl_random_loops_max   = (
            cfg.icl_random_loops_max if cfg.icl_random_loops_max is not None
            else cfg.num_icl_loops
        )
        self.inject_only_first_loop      = cfg.inject_only_first_loop
        self.loop_gradient_checkpointing = cfg.loop_gradient_checkpointing
        self.tbptt_last_k                = cfg.tbptt_last_k
        self._D = D

        if cfg.iter_embed:
            self.loop_film = nn.Embedding(self.icl_random_loops_max, 2 * D)
            nn.init.zeros_(self.loop_film.weight)
        else:
            self.loop_film = None

        # ---- loop residual scaling ----
        self.use_loop_residual_scaling  = cfg.use_loop_residual_scaling
        self.loop_residual_lambda       = cfg.loop_residual_lambda
        self.loop_residual_scaling_fn   = cfg.loop_residual_scaling_fn
        self.loop_residual_custom_fn    = None   # set externally when using "custom"
        self._entropy_h0: float         = 0.0   # baseline entropy for "entropy_adaptive"
        self._layers_per_loop = cfg.axial_num_blocks * (
            cfg.num_within_per_round + cfg.num_cross_per_round + 1 + cfg.num_icl_per_round
        )

        # learned MLP: shared 2→1 MLP applied independently per embedding channel.
        # Input per channel: [t, log_rms_channel]
        #   t        = loop_i / max(n_loops-1, 1)  ∈ [0, 1]
        #   log_rms  = log √E[Δd²]  (per-channel RMS of the residual, detached)
        # Output: scale multiplier for that channel (shared weights across channels).
        # Init: last layer weight=0, bias=softplus_inv(1)≈0.541 → all scales=base at start,
        # identical to "inv_n".
        if cfg.loop_residual_scaling_fn == "learned_mlp":
            self.loop_scale_mlp: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(2, 16),
                nn.SiLU(),
                nn.Linear(16, 1),
            )
            nn.init.zeros_(self.loop_scale_mlp[2].weight)
            self.loop_scale_mlp[2].bias.data.fill_(math.log(math.e - 1))  # softplus_inv(1.0)
        else:
            self.loop_scale_mlp = None

        # ---- one-way column→row attention ----
        self.use_oneway = cfg.use_oneway
        if cfg.use_oneway:
            self.cls_cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    emsize=E,
                    nhead=cfg.feat_agg_num_heads,
                    dim_feedforward=E * cfg.ff_factor,
                    norm_factory=norm_factory,
                    softmax_scaling_layer=_make_scaling(cfg.feat_agg_num_heads, E // cfg.feat_agg_num_heads),
                    post_norm_factory=post_norm_factory,
                )
                for _ in range(cfg.axial_num_blocks)
            ])

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        x_RBC: torch.Tensor,
        num_train: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        nan_ind_BRC: torch.Tensor | None = None
        if self.use_nan_indicators:
            nan_ind_BRC = (
                torch.isnan(x_RBC) * _NAN_INDICATOR
                + torch.isposinf(x_RBC) * _INFINITY_INDICATOR
                + torch.isneginf(x_RBC) * _NEG_INF_INDICATOR
            ).to(x_RBC.dtype).transpose(0, 1)

        is_finite = torch.isfinite(x_RBC)
        x_train   = torch.where(is_finite[:num_train], x_RBC[:num_train], torch.nan)
        feature_means = torch.nan_to_num(torch.nanmean(x_train, dim=0), 0)
        x_RBC = torch.where(is_finite, x_RBC, feature_means.unsqueeze(0).expand_as(x_RBC))
        x_RBC = self.standard_scaler(x=x_RBC, num_train_rows=num_train)
        return x_RBC.transpose(0, 1), nan_ind_BRC

    def _group_features(
        self,
        x_BRC: torch.Tensor,
        nan_ind_BRC: torch.Tensor | None,
    ) -> torch.Tensor:
        size = self.feature_group_size
        groups = [torch.roll(x_BRC, shifts=-i, dims=2) for i in range(size)]
        x_grouped = torch.stack(groups, dim=-1)
        if nan_ind_BRC is not None:
            ind_groups = [torch.roll(nan_ind_BRC, shifts=-i, dims=2) for i in range(size)]
            x_grouped = torch.cat([x_grouped, torch.stack(ind_groups, dim=-1)], dim=-1)
        return x_grouped

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    def _sample_num_loops(self) -> int:
        mu    = self.icl_random_loops_mu
        sigma = self.icl_random_loops_sigma
        normal_mean = math.log(mu) - 0.5 * sigma ** 2
        tau = Normal(torch.tensor(normal_mean), torch.tensor(sigma)).sample().item()
        n   = int(Poisson(torch.tensor(math.exp(tau))).sample().item()) + 1
        return min(n, self.icl_random_loops_max)

    def _compute_loop_scale(self, n_loops: int, loop_i: int = 0) -> float:
        """Return the residual step size for loop_residual_scaling.

        All variants share the ``λ / sqrt(L)`` base factor; only the
        n-dependent multiplier differs.  ``L = _layers_per_loop``.
        ``loop_i`` is only used by schedule-based functions (e.g. "cosine").
        """
        lam   = self.loop_residual_lambda
        sqrtL = math.sqrt(self._layers_per_loop)
        fn    = self.loop_residual_scaling_fn
        n     = n_loops
        if fn == "inv_n":
            return lam / (n * sqrtL)
        elif fn == "inv_sqrt_n":
            return lam / (math.sqrt(n) * sqrtL)
        elif fn == "constant":
            return lam / sqrtL
        elif fn == "inv_n2":
            return lam / (n * n * sqrtL)
        elif fn == "log_n":
            return lam * math.log(max(n, 2)) / (n * sqrtL)
        elif fn == "cosine":
            # Cosine annealing: starts at λ/√L (t=0), decays to 0 (t=n).
            return (lam / sqrtL) * 0.5 * (1.0 + math.cos(math.pi * loop_i / n))
        elif fn == "custom":
            if self.loop_residual_custom_fn is None:
                raise ValueError(
                    "loop_residual_scaling_fn='custom' requires setting "
                    "model.loop_residual_custom_fn = callable(n_loops, loop_i) -> float"
                )
            return float(self.loop_residual_custom_fn(n_loops, loop_i))
        else:
            raise ValueError(
                f"Unknown loop_residual_scaling_fn: {fn!r}. "
                "Choose one of: 'inv_n', 'inv_sqrt_n', 'constant', 'inv_n2', 'log_n', 'cosine', 'custom', 'entropy_adaptive', 'learned_mlp'."
            )

    def _compute_learned_mlp_scale(
        self,
        n_loops: int,
        loop_i: int,
        delta_D: torch.Tensor,
        delta_E: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-embedding residual scales for the 'learned_mlp' variant.

        A shared 2→1 MLP is applied independently to each embedding channel:
          input:  [t, log_rms_channel]
            t           = loop_i / max(n_loops-1, 1)  ∈ [0, 1]
            log_rms     = log √E[Δd²]  per-channel RMS of the residual (detached)
          output: scale multiplier for that channel

        Shared weights across channels: the MLP learns a universal rule —
        "given position and how much this channel changed, how much to let through."

        Parameters
        ----------
        delta_D : (B, R, D)    — x_BRD_new  - x_BRD_pre
        delta_E : (B, R, C, E) — x_BRCE_new - x_BRCE_pre

        Returns
        -------
        scale_D : (D,) — per-channel scales for x_BRD
        scale_E : (E,) — per-channel scales for x_BRCE
        """
        lam   = self.loop_residual_lambda
        sqrtL = math.sqrt(self._layers_per_loop)
        base  = lam / (n_loops * sqrtL)
        dtype = delta_D.dtype
        device = self.loop_scale_mlp[0].weight.device

        t_val = loop_i / max(n_loops - 1, 1)

        # Per-channel log-RMS (detached — MLP gates the magnitude, does not optimise it)
        log_rms_D = delta_D.detach().pow(2).mean(dim=(0, 1)).clamp(min=1e-8).log()     # (D,)
        log_rms_E = delta_E.detach().pow(2).mean(dim=(0, 1, 2)).clamp(min=1e-8).log()  # (E,)

        D, E = log_rms_D.shape[0], log_rms_E.shape[0]

        t_D = log_rms_D.new_full((D, 1), t_val)
        t_E = log_rms_E.new_full((E, 1), t_val)

        inp_D = torch.cat([t_D, log_rms_D.unsqueeze(1)], dim=1)  # (D, 2)
        inp_E = torch.cat([t_E, log_rms_E.unsqueeze(1)], dim=1)  # (E, 2)

        scale_D = base * F.softplus(self.loop_scale_mlp(inp_D.to(device))).squeeze(1)  # (D,)
        scale_E = base * F.softplus(self.loop_scale_mlp(inp_E.to(device))).squeeze(1)  # (E,)
        return scale_D, scale_E

    def _compute_entropy_adaptive_scale(
        self,
        x_BRD: torch.Tensor,
        num_train: int,
        y_BN: torch.Tensor,
        n_loops: int,
        loop_i: int,
    ) -> float:
        """Loop-adaptive residual scale based on prediction entropy.

        Interpolates between ``inv_n`` (α=1, conservative) and ``inv_sqrt_n``
        (α=0.5, light) depending on how quickly the model gains confidence:

        - Fast entropy drop → easy task → α→0.5 (lighter damping, let loops through)
        - Slow entropy drop → hard task → α stays near 1.0 (stronger damping)

        Falls back to ``inv_n`` during training to avoid the per-loop decoder
        overhead and keep the gradient graph clean.
        """
        lam   = self.loop_residual_lambda
        sqrtL = math.sqrt(self._layers_per_loop)

        if self.training:
            return lam / (n_loops * sqrtL)

        test_emb_len = x_BRD.shape[1] - num_train
        if test_emb_len == 0:
            return lam / (n_loops * sqrtL)

        with torch.no_grad():
            normed    = self.output_norm(x_BRD.detach())
            train_emb = normed[:, :num_train]
            test_emb  = normed[:, num_train:]
            logits    = self.many_class_decoder(train_emb, test_emb, y_BN)
            probs     = torch.softmax(logits.float(), dim=-1).clamp(min=1e-9)
            entropy   = -(probs * probs.log()).sum(dim=-1).mean().item()

        if loop_i == 0:
            self._entropy_h0 = entropy

        h0              = self._entropy_h0 + 1e-9
        normalized_drop = max(0.0, min(1.0, (h0 - entropy) / h0))
        alpha           = 1.0 - normalized_drop   # [0, 1]: 0=constant (easy), 1=inv_n (hard)
        return lam / (n_loops ** alpha * sqrtL)

    def _apply_film(self, x_BRD: torch.Tensor, loop_i: int) -> torch.Tensor:
        assert self.loop_film is not None
        idx    = torch.tensor(min(loop_i, self.loop_film.num_embeddings - 1), device=x_BRD.device)
        params = self.loop_film(idx)
        D      = self._D
        gamma, beta = params[:D], params[D:]
        return (1.0 + gamma) * x_BRD + beta

    def _init_x_brd(self, B: int, R: int, x_BRCE: torch.Tensor, Cl: int, E: int) -> torch.Tensor:
        """Initialise the row-stream tensor x_BRD before the outer loop."""
        if self.use_oneway:
            return (
                self.axial_embedder.cls_tokens
                .to(dtype=x_BRCE.dtype)
                .reshape(1, 1, Cl * E)
                .expand(B, R, -1)
                .clone()
            )
        return x_BRCE.new_zeros(B, R, Cl * E)

    # ------------------------------------------------------------------
    # Axial step
    # ------------------------------------------------------------------

    def _axial_step_chunked(
        self,
        x_BRCE: torch.Tensor,
        x_BRD: torch.Tensor,
        num_train: int,
        y_col_emb: torch.Tensor,
        skip_cell_label_inject: bool = False,
        test_chunk_size: int = 512,
        offload_x_brce: bool = False,
        residual_alpha: torch.Tensor | None = None,
        residual_beta: torch.Tensor | None = None,
        return_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Like _axial_step but processes test rows in chunks to reduce peak GPU memory.

        At each sub-block, train rows are processed first to produce a fixed
        cached state; test rows are then processed in chunks of ``test_chunk_size``
        that reuse this cached state:

        - Within-col  (InducedSelfAttentionBlock): inducing hidden is computed
          once from train rows (``return_hidden=True``) then shared across all
          test chunks via ``cached_hidden``.
        - Cross-col   (_CrossColBlock): row-independent; chunked directly.
        - Readout     (CrossAttentionBlock or cls_cross_attn_blocks): row-
          independent; chunked directly.
        - ICL         (MaskedICLTransformerBlock): train KV is computed once
          (``return_kv=True``) then shared across all test chunks via
          ``cached_kv``.

        The result is numerically equivalent to ``_axial_step`` at inference.

        Constraints
        -----------
        - Must be used inside ``torch.no_grad()`` (inference only; no gradients
          propagate through cached states).
        - Incompatible with ``use_loop_residual_scaling`` (need_pre=True path in ``forward``).
          Both flags are False by default, so this constraint is rarely relevant.
        """
        assert not self.training, "_axial_step_chunked is for inference only"
        emb = self.axial_embedder
        B, R, C, E = x_BRCE.shape
        Cl = emb.num_cls_tokens
        D  = Cl * E
        num_test   = R - num_train
        icl_idx    = 0
        within_idx = 0
        cross_idx  = 0
        _within_hiddens: list[torch.Tensor]   = []
        _icl_kvs:        list                 = []

        for i in range(emb.num_axial_blocks):

            # ---- cell stream: label inject (train only) ----
            # Creates a new x_BRCE tensor → breaks any external aliasing.
            if not skip_cell_label_inject:
                inject = emb.label_inject_projs[i](y_col_emb)
                pad    = inject.new_zeros(B, num_test, E)
                x_BRCE = x_BRCE + torch.cat([inject, pad], dim=1).unsqueeze(2)

            # ---- within-col blocks ----
            for _ in range(emb.num_within_per_round):
                # Compute inducing hidden from train rows; update train cells.
                x_train_new, hidden = emb.within_col_blocks[within_idx](
                    x_BRCE[:, :num_train],
                    single_eval_pos=num_train,
                    return_hidden=True,
                )
                x_BRCE[:, :num_train] = x_train_new
                if return_cache:
                    _within_hiddens.append(hidden)
                # Process test rows in chunks using the cached inducing hidden.
                for start in range(0, num_test, test_chunk_size):
                    end = min(start + test_chunk_size, num_test)
                    x_chunk_new, _ = emb.within_col_blocks[within_idx](
                        x_BRCE[:, num_train + start: num_train + end],
                        cached_hidden=hidden,
                    )
                    x_BRCE[:, num_train + start: num_train + end] = x_chunk_new
                within_idx += 1

            # ---- cross-col blocks (row-independent) ----
            for _ in range(emb.num_cross_per_round):
                x_BRCE[:, :num_train] = emb.cross_col_blocks[cross_idx](
                    x_BRCE[:, :num_train],
                    residual_alpha=residual_alpha,
                    residual_beta=residual_beta,
                )
                for start in range(0, num_test, test_chunk_size):
                    end = min(start + test_chunk_size, num_test)
                    x_BRCE[:, num_train + start: num_train + end] = emb.cross_col_blocks[cross_idx](
                        x_BRCE[:, num_train + start: num_train + end],
                        residual_alpha=residual_alpha,
                        residual_beta=residual_beta,
                    )
                cross_idx += 1

            # ---- readout: cell → row (row-independent) ----
            if self.use_oneway:
                x_BRD_new = x_BRD.new_empty(B, R, D)
                xd_train   = x_BRD[:, :num_train].view(B * num_train, Cl, E)
                cols_train = x_BRCE[:, :num_train].flatten(0, 1)
                x_BRD_new[:, :num_train] = (
                    self.cls_cross_attn_blocks[i](xd_train, cols_train).view(B, num_train, D)
                )
                for start in range(0, num_test, test_chunk_size):
                    end = min(start + test_chunk_size, num_test)
                    chunk_len  = end - start
                    xd_chunk   = x_BRD[:, num_train + start: num_train + end].view(B * chunk_len, Cl, E)
                    cols_chunk = x_BRCE[:, num_train + start: num_train + end].flatten(0, 1)
                    x_BRD_new[:, num_train + start: num_train + end] = (
                        self.cls_cross_attn_blocks[i](xd_chunk, cols_chunk).view(B, chunk_len, D)
                    )
                x_BRD = x_BRD_new
            else:
                cls = emb.cls_tokens.view(1, 1, Cl, E)
                cls_train     = cls.expand(B, num_train, -1, -1)
                cls_out_train = emb.readout_blocks[i](
                    cls_train.flatten(0, 1), x_BRCE[:, :num_train].flatten(0, 1)
                )
                x_BRD[:, :num_train] = (
                    x_BRD[:, :num_train] + emb.row_ln(cls_out_train.view(B, num_train, D))
                )
                for start in range(0, num_test, test_chunk_size):
                    end = min(start + test_chunk_size, num_test)
                    chunk_len     = end - start
                    cls_chunk     = cls.expand(B, chunk_len, -1, -1)
                    cls_out_chunk = emb.readout_blocks[i](
                        cls_chunk.flatten(0, 1),
                        x_BRCE[:, num_train + start: num_train + end].flatten(0, 1),
                    )
                    x_BRD[:, num_train + start: num_train + end] = (
                        x_BRD[:, num_train + start: num_train + end]
                        + emb.row_ln(cls_out_chunk.view(B, chunk_len, D))
                    )

            x_BRD = emb.pre_icl_lns[i](x_BRD)

            # ---- ICL label injection (train only) ----
            if offload_x_brce:
                x_BRCE = x_BRCE.cpu()
            icl_label = emb.icl_label_projs[i](y_col_emb)
            x_BRD[:, :num_train] = x_BRD[:, :num_train] + icl_label

            # ---- ICL blocks ----
            for _ in range(emb.num_icl_per_round):
                # Process train rows; cache their KV for reuse by test rows.
                x_BRD_train_new, kv = emb.icl_blocks[icl_idx](
                    x_BRD[:, :num_train],
                    num_train,
                    return_kv=True,
                    residual_alpha=residual_alpha,
                    residual_beta=residual_beta,
                )
                x_BRD[:, :num_train] = x_BRD_train_new
                if return_cache:
                    _icl_kvs.append(kv)
                # Process test rows in chunks using the cached train KV.
                for start in range(0, num_test, test_chunk_size):
                    end = min(start + test_chunk_size, num_test)
                    x_chunk_new, _ = emb.icl_blocks[icl_idx](
                        x_BRD[:, num_train + start: num_train + end],
                        num_train,
                        cached_kv=kv,
                        residual_alpha=residual_alpha,
                        residual_beta=residual_beta,
                    )
                    x_BRD[:, num_train + start: num_train + end] = x_chunk_new
                icl_idx += 1

            if offload_x_brce:
                x_BRCE = x_BRCE.to(x_BRD.device)

        if return_cache:
            return x_BRCE, x_BRD, _within_hiddens, _icl_kvs
        return x_BRCE, x_BRD

    def _axial_step_test_only(
        self,
        x_BRCE_test: torch.Tensor,
        x_BRD_test:  torch.Tensor,
        within_hiddens: list[torch.Tensor],
        icl_kvs:        list,
        num_train:      int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one axial step on test rows only, using cached train-side state.

        Mirrors ``_axial_step_chunked`` but skips all train-only computation
        (label injection, KV/hidden recomputation).  All test rows are processed
        in a single batch (no chunking).

        Parameters
        ----------
        x_BRCE_test     : (B, num_test, C, E)
        x_BRD_test      : (B, num_test, D)
        within_hiddens  : inducing hidden tensors from the corresponding prefill step,
                          one per within-col block (length = num_axial_blocks * num_within_per_round).
        icl_kvs         : KVCacheEntry list from the corresponding prefill step,
                          one per ICL block (length = num_axial_blocks * num_icl_per_round).
        num_train       : number of training rows (used by softmax-scaling in ICL attention).
        """
        assert not self.training, "_axial_step_test_only is for inference only"
        emb = self.axial_embedder
        B, num_test, C, E = x_BRCE_test.shape
        Cl = emb.num_cls_tokens
        D  = Cl * E

        within_idx = 0
        cross_idx  = 0
        icl_idx    = 0

        for i in range(emb.num_axial_blocks):
            # label inject: test rows always receive zeros — skip

            # ---- within-col: use cached inducing hidden ----
            for _ in range(emb.num_within_per_round):
                x_BRCE_test, _ = emb.within_col_blocks[within_idx](
                    x_BRCE_test,
                    cached_hidden=within_hiddens[within_idx],
                )
                within_idx += 1

            # ---- cross-col (row-independent) ----
            for _ in range(emb.num_cross_per_round):
                x_BRCE_test = emb.cross_col_blocks[cross_idx](x_BRCE_test)
                cross_idx += 1

            # ---- readout: cell → row ----
            if self.use_oneway:
                xd_test   = x_BRD_test.view(B * num_test, Cl, E)
                cols_test = x_BRCE_test.flatten(0, 1)
                x_BRD_test = self.cls_cross_attn_blocks[i](xd_test, cols_test).view(B, num_test, D)
            else:
                cls     = emb.cls_tokens.view(1, 1, Cl, E).expand(B, num_test, -1, -1)
                cls_out = emb.readout_blocks[i](cls.flatten(0, 1), x_BRCE_test.flatten(0, 1))
                x_BRD_test = x_BRD_test + emb.row_ln(cls_out.view(B, num_test, D))

            x_BRD_test = emb.pre_icl_lns[i](x_BRD_test)

            # ICL label inject: test rows receive zeros — skip

            # ---- ICL blocks: use cached KV ----
            for _ in range(emb.num_icl_per_round):
                x_BRD_test, _ = emb.icl_blocks[icl_idx](
                    x_BRD_test,
                    num_train,
                    cached_kv=icl_kvs[icl_idx],
                )
                icl_idx += 1

        return x_BRCE_test, x_BRD_test

    def _axial_block_i(
        self,
        x_BRCE: torch.Tensor,
        x_BRD: torch.Tensor,
        block_i: int,
        within_start: int,
        cross_start: int,
        icl_start: int,
        num_train: int,
        y_col_emb: torch.Tensor,
        skip_cell_label_inject: bool = False,
        residual_alpha: torch.Tensor | None = None,
        residual_beta: torch.Tensor | None = None,
        offload_x_brce: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single axial block step — extracted for per-block gradient checkpointing."""
        emb = self.axial_embedder
        B, R, C, E = x_BRCE.shape
        Cl = emb.num_cls_tokens
        D  = Cl * E

        # ---- cell stream ----
        if not skip_cell_label_inject:
            inject = emb.label_inject_projs[block_i](y_col_emb)
            pad    = inject.new_zeros(B, R - num_train, E)
            x_BRCE = x_BRCE + torch.cat([inject, pad], dim=1).unsqueeze(2)

        wi = within_start
        for _ in range(emb.num_within_per_round):
            x_BRCE, _ = emb.within_col_blocks[wi](x_BRCE, single_eval_pos=num_train)
            wi += 1

        ci = cross_start
        for _ in range(emb.num_cross_per_round):
            x_BRCE = emb.cross_col_blocks[ci](x_BRCE,
                                               residual_alpha=residual_alpha,
                                               residual_beta=residual_beta)
            ci += 1

        # ---- readout: cell → row ----
        if self.use_oneway:
            xd_flat   = x_BRD.view(B * R, Cl, E)
            cols_flat = x_BRCE.flatten(0, 1)
            xd_out    = self.cls_cross_attn_blocks[block_i](xd_flat, cols_flat)
            x_BRD     = xd_out.view(B, R, D)
        else:
            cls     = emb.cls_tokens.view(1, 1, Cl, E).expand(B, R, -1, -1)
            cls_out = emb.readout_blocks[block_i](cls.flatten(0, 1), x_BRCE.flatten(0, 1))
            x_BRD   = x_BRD + emb.row_ln(cls_out.view(B, R, D))
        x_BRD   = emb.pre_icl_lns[block_i](x_BRD)

        # ---- row stream (ICL) ----
        if offload_x_brce:
            x_BRCE = x_BRCE.cpu()
        icl_label = emb.icl_label_projs[block_i](y_col_emb)
        # Out-of-place label inject — avoids in-place op on checkpoint input tensor
        x_BRD = x_BRD + torch.cat(
            [icl_label, icl_label.new_zeros(B, x_BRD.shape[1] - num_train, icl_label.shape[-1])],
            dim=1,
        )

        ii = icl_start
        for _ in range(emb.num_icl_per_round):
            x_BRD, _ = emb.icl_blocks[ii](
                x_BRD, num_train,
                residual_alpha=residual_alpha,
                residual_beta=residual_beta,
            )
            ii += 1

        if offload_x_brce:
            x_BRCE = x_BRCE.to(x_BRD.device)

        return x_BRCE, x_BRD

    def _axial_step(
        self,
        x_BRCE: torch.Tensor,
        x_BRD: torch.Tensor,
        num_train: int,
        y_col_emb: torch.Tensor,
        skip_cell_label_inject: bool = False,
        offload_x_brce: bool = False,
        residual_alpha: torch.Tensor | None = None,
        residual_beta: torch.Tensor | None = None,
        gradient_checkpointing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.axial_embedder
        within_idx = 0
        cross_idx  = 0
        icl_idx    = 0

        for i in range(emb.num_axial_blocks):
            wi, ci, ii = within_idx, cross_idx, icl_idx

            if self.training and gradient_checkpointing:
                def _block(_xe, _xd, _i=i, _wi=wi, _ci=ci, _ii=ii):
                    return self._axial_block_i(
                        _xe, _xd, _i, _wi, _ci, _ii,
                        num_train, y_col_emb,
                        skip_cell_label_inject, residual_alpha, residual_beta,
                    )
                x_BRCE, x_BRD = torch.utils.checkpoint.checkpoint(
                    _block, x_BRCE, x_BRD, use_reentrant=False)
            else:
                x_BRCE, x_BRD = self._axial_block_i(
                    x_BRCE, x_BRD, i, wi, ci, ii,
                    num_train, y_col_emb,
                    skip_cell_label_inject, residual_alpha, residual_beta,
                    offload_x_brce=offload_x_brce,
                )

            within_idx += emb.num_within_per_round
            cross_idx  += emb.num_cross_per_round
            icl_idx    += emb.num_icl_per_round

        return x_BRCE, x_BRD

    # ------------------------------------------------------------------
    # Prefill / predict_cached
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_loops: int | None = None,
    ) -> LoopICLTrainCache:
        """Run the train-side forward pass and return a reusable cache.

        Parameters
        ----------
        X_train  : (B, N_train, C) raw training features.
        y_train  : (B, N_train)   integer training labels.
        num_loops: override the default number of outer ICL loops.

        Returns
        -------
        LoopICLTrainCache  — pass to ``predict_cached`` for fast test queries.
        """
        self.eval()
        B, N, C = X_train.shape
        num_train = N

        # ---- Stage 0: preprocess (train rows only) ----
        x_RBC = X_train.transpose(0, 1)          # (N, B, C)

        # NaN indicator (computed on raw features before imputation)
        nan_ind_BRC: torch.Tensor | None = None
        if self.use_nan_indicators:
            nan_ind_BRC = (
                torch.isnan(x_RBC) * _NAN_INDICATOR
                + torch.isposinf(x_RBC) * _INFINITY_INDICATOR
                + torch.isneginf(x_RBC) * _NEG_INF_INDICATOR
            ).to(x_RBC.dtype).transpose(0, 1)    # (B, N, C)

        # NaN imputation using train means
        is_finite     = torch.isfinite(x_RBC)
        feature_means = torch.nan_to_num(
            torch.nanmean(torch.where(is_finite, x_RBC, torch.nan), dim=0), 0
        )                                         # (B, C)
        x_RBC = torch.where(is_finite, x_RBC, feature_means.unsqueeze(0))

        # Fit scaler on train rows and transform
        scaler_params = self.standard_scaler.fit(x_RBC)
        x_BRC = self.standard_scaler.transform(x_RBC, scaler_params).transpose(0, 1)
        # (B, N, C) — normalised

        # Feature grouping & initial cell embedding
        x_BRCG   = self._group_features(x_BRC, nan_ind_BRC)
        x_BRCE   = self.x_embed(x_BRCG)           # (B, N, C, E)

        # ---- Stage 1: conditioning ----
        y_BN      = y_train
        y_col_emb = self.col_y_encoder(y_BN)      # (B, N, E)
        x_BRCE    = x_BRCE + y_col_emb.unsqueeze(2)

        x_train_BCN  = x_BRC.permute(0, 2, 1)     # (B, C, N)
        x_sorted_BCN = x_train_BCN.sort(dim=-1).values

        centers, widths  = _adaptive_bin_geometry(x_sorted_BCN, self._n_bins)
        log_hist_BCK     = _soft_log_histogram(x_train_BCN, centers, widths, num_train)
        moments_BC4      = _compute_moments(x_train_BCN)

        log_hist_cond_BCE: torch.Tensor | None = None
        if self.use_log_hist_conditioning:
            log_hist_cond_BCE = self.log_hist_conditioner(log_hist_BCK, moments_BC4)  # (B, C, E)
            x_BRCE = x_BRCE + log_hist_cond_BCE.unsqueeze(1)

        disc_hist_cond_BCE: torch.Tensor | None = None
        if self.use_disc_hist_conditioning:
            disc_hist_cond_BCE = self.discriminative_hist_conditioner(
                x_train_BCN, centers, widths, log_hist_BCK, y_BN, num_train,
                chunk_size=self._disc_hist_chunk_size,
            )                                      # (B, C, E)
            x_BRCE = x_BRCE + disc_hist_cond_BCE.unsqueeze(1)

        if self.use_fourier_ood:
            x_BRCE = x_BRCE + self.fourier_quantile_encoder(x_BRC, x_sorted_BCN, num_train)

        # ---- Stage 2: outer loop (train rows only, collect hiddens + KVs) ----
        E  = x_BRCE.shape[-1]
        Cl = self.axial_embedder.num_cls_tokens
        x_BRD = self._init_x_brd(B, num_train, x_BRCE, Cl, E)

        n_loops = num_loops if num_loops is not None else self.num_icl_loops

        all_hiddens: list[torch.Tensor] = []
        all_kvs:     list               = []

        for loop_i in range(n_loops):
            if self.loop_film is not None:
                x_BRD = self._apply_film(x_BRD, loop_i)

            skip_inject = self.inject_only_first_loop and loop_i > 0

            if self.use_loop_residual_scaling:
                x_BRCE_pre, x_BRD_pre = x_BRCE, x_BRD

            # Pass train-only tensors (num_test=0 → test loops are no-ops)
            x_BRCE, x_BRD, step_hiddens, step_kvs = self._axial_step_chunked(
                x_BRCE, x_BRD,
                num_train=num_train,
                y_col_emb=y_col_emb,
                skip_cell_label_inject=skip_inject,
                return_cache=True,
            )

            if self.use_loop_residual_scaling:
                delta_D = x_BRD  - x_BRD_pre
                delta_E = x_BRCE - x_BRCE_pre
                if self.loop_residual_scaling_fn == "learned_mlp":
                    scale_D, scale_E = self._compute_learned_mlp_scale(n_loops, loop_i, delta_D, delta_E)
                    x_BRD  = x_BRD_pre  + scale_D * delta_D
                    x_BRCE = x_BRCE_pre + scale_E * delta_E
                else:
                    loop_scale = self._compute_loop_scale(n_loops, loop_i)
                    x_BRD  = x_BRD_pre  + loop_scale * delta_D
                    x_BRCE = x_BRCE_pre + loop_scale * delta_E

            all_hiddens.extend(step_hiddens)
            all_kvs.extend(step_kvs)

        x_BRD_normed = self.output_norm(x_BRD)    # (B, N, D)

        return LoopICLTrainCache(
            scaler_params=scaler_params,
            feature_means=feature_means,
            x_sorted_BCN=x_sorted_BCN,
            log_hist_cond=log_hist_cond_BCE,
            disc_hist_cond=disc_hist_cond_BCE,
            y_col_emb=y_col_emb,
            within_hiddens=all_hiddens,
            icl_kvs=all_kvs,
            x_BRD_train_normed=x_BRD_normed,
            y_BN=y_BN,
            num_train=num_train,
            num_loops=n_loops,
        )

    @torch.no_grad()
    def predict_cached(
        self,
        X_test: torch.Tensor,
        cache: LoopICLTrainCache,
    ) -> torch.Tensor:
        """Predict on new test rows using a precomputed train cache.

        No train-side computation is repeated: within-col hidden states and ICL
        KV caches are reused directly from ``cache``.

        Parameters
        ----------
        X_test : (B, M, C) raw test features.
        cache  : LoopICLTrainCache returned by ``prefill()``.

        Returns
        -------
        Tensor (B, M, T)  — class logits, same format as ``forward()``.
        """
        self.eval()
        B, M, C = X_test.shape

        # ---- Preprocess test features using cached train stats ----
        x_RBC_test = X_test.transpose(0, 1)       # (M, B, C)

        nan_ind_BRC_test: torch.Tensor | None = None
        if self.use_nan_indicators:
            nan_ind_BRC_test = (
                torch.isnan(x_RBC_test) * _NAN_INDICATOR
                + torch.isposinf(x_RBC_test) * _INFINITY_INDICATOR
                + torch.isneginf(x_RBC_test) * _NEG_INF_INDICATOR
            ).to(x_RBC_test.dtype).transpose(0, 1)

        is_finite_test = torch.isfinite(x_RBC_test)
        x_RBC_test = torch.where(
            is_finite_test, x_RBC_test,
            cache.feature_means.unsqueeze(0).expand(M, -1, -1),
        )
        x_BRC_test = self.standard_scaler.transform(
            x_RBC_test, cache.scaler_params
        ).transpose(0, 1)                          # (B, M, C)

        # ---- Stage 1: embed test rows ----
        x_BRCG_test = self._group_features(x_BRC_test, nan_ind_BRC_test)
        x_BRCE_test = self.x_embed(x_BRCG_test)   # (B, M, C, E)

        # Add per-column conditioning (same broadcast as train, cached)
        if cache.log_hist_cond is not None:
            x_BRCE_test = x_BRCE_test + cache.log_hist_cond.unsqueeze(1)
        if cache.disc_hist_cond is not None:
            x_BRCE_test = x_BRCE_test + cache.disc_hist_cond.unsqueeze(1)
        if self.use_fourier_ood:
            x_BRCE_test = x_BRCE_test + self.fourier_quantile_encoder(
                x_BRC_test, cache.x_sorted_BCN, cache.num_train
            )

        # ---- Stage 2: outer loop (test rows only, using cached states) ----
        E  = x_BRCE_test.shape[-1]
        Cl = self.axial_embedder.num_cls_tokens
        x_BRD_test = self._init_x_brd(B, M, x_BRCE_test, Cl, E)

        # Each loop consumes one slice of the cached hiddens/KVs.
        blocks_per_step = (
            self.axial_embedder.num_axial_blocks
            * self.axial_embedder.num_within_per_round
        )
        kvs_per_step = (
            self.axial_embedder.num_axial_blocks
            * self.axial_embedder.num_icl_per_round
        )

        for loop_i in range(cache.num_loops):
            if self.loop_film is not None:
                x_BRD_test = self._apply_film(x_BRD_test, loop_i)

            if self.use_loop_residual_scaling:
                x_BRCE_test_pre, x_BRD_test_pre = x_BRCE_test, x_BRD_test

            h_start = loop_i * blocks_per_step
            k_start = loop_i * kvs_per_step

            x_BRCE_test, x_BRD_test = self._axial_step_test_only(
                x_BRCE_test,
                x_BRD_test,
                within_hiddens=cache.within_hiddens[h_start: h_start + blocks_per_step],
                icl_kvs=cache.icl_kvs[k_start: k_start + kvs_per_step],
                num_train=cache.num_train,
            )

            if self.use_loop_residual_scaling:
                delta_D_test = x_BRD_test  - x_BRD_test_pre
                delta_E_test = x_BRCE_test - x_BRCE_test_pre
                if self.loop_residual_scaling_fn == "learned_mlp":
                    scale_D, scale_E = self._compute_learned_mlp_scale(cache.num_loops, loop_i, delta_D_test, delta_E_test)
                    x_BRD_test  = x_BRD_test_pre  + scale_D * delta_D_test
                    x_BRCE_test = x_BRCE_test_pre + scale_E * delta_E_test
                else:
                    loop_scale = self._compute_loop_scale(cache.num_loops, loop_i)
                    x_BRD_test  = x_BRD_test_pre  + loop_scale * delta_D_test
                    x_BRCE_test = x_BRCE_test_pre + loop_scale * delta_E_test

        # ---- Stage 4: decode ----
        test_emb = self.output_norm(x_BRD_test)   # (B, M, D)
        logits   = self.many_class_decoder(
            cache.x_BRD_train_normed, test_emb, cache.y_BN
        )
        return torch.nan_to_num(logits.transpose(0, 1), nan=0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        single_eval_pos: int | None = None,
        train_size: int | None = None,
        inference_num_blocks: int | None = None,
        num_loops: int | None = None,
        return_all_logits: bool = False,
        return_embedding: bool = False,
        test_chunk_size: int | None = None,
        offload_x_brce: bool = False,
        gradient_checkpointing: bool | None = None,
        **_kwargs,
    ) -> torch.Tensor | List[torch.Tensor]:
        """Multi-view looped forward pass.

        Parameters
        ----------
        X : Tensor (B, R, C)
        y_train : Tensor (B, R)  — labels for all rows (test rows can be any value)
        single_eval_pos : int    — number of training rows; rows[single_eval_pos:] are test
        train_size : int, optional — alias for single_eval_pos
        inference_num_blocks : int, optional
        num_loops : int, optional  — direct override
        return_all_logits : bool   — return per-loop logits list

        Returns
        -------
        Tensor (B, M, T) or List[Tensor (B, M, T)]
        """
        if gradient_checkpointing is None:
            gradient_checkpointing = self.loop_gradient_checkpointing

        x_RBC = X.transpose(0, 1)
        y_NB  = y_train.transpose(0, 1)
        num_train = (
            int(single_eval_pos) if single_eval_pos is not None
            else int(train_size) if train_size is not None
            else int(y_NB.shape[0])
        )

        # Stage 0
        x_BRC, nan_ind_BRC = self._preprocess(x_RBC, num_train)
        x_BRCG = self._group_features(x_BRC, nan_ind_BRC)

        x_train_BCN  = x_BRC[:, :num_train].permute(0, 2, 1)
        x_sorted_BCN = x_train_BCN.sort(dim=-1).values

        centers, widths = _adaptive_bin_geometry(x_sorted_BCN, self._n_bins)
        log_hist_BCK    = _soft_log_histogram(x_train_BCN, centers, widths, num_train)
        moments_BC4     = _compute_moments(x_train_BCN)

        # Stage 1
        x_BRCE    = self.x_embed(x_BRCG)
        y_BN      = y_NB.transpose(0, 1)[:, :num_train]
        y_col_emb = self.col_y_encoder(y_BN)
        x_BRCE[:, :num_train] = x_BRCE[:, :num_train] + y_col_emb.unsqueeze(2)

        if self.use_log_hist_conditioning:
            x_BRCE = x_BRCE + self.log_hist_conditioner(log_hist_BCK, moments_BC4).unsqueeze(1)
        if self.use_disc_hist_conditioning:
            x_BRCE = x_BRCE + self.discriminative_hist_conditioner(
                x_train_BCN, centers, widths, log_hist_BCK, y_BN, num_train,
                chunk_size=self._disc_hist_chunk_size,
            ).unsqueeze(1)
        if self.use_fourier_ood:
            x_BRCE = x_BRCE + self.fourier_quantile_encoder(x_BRC, x_sorted_BCN, num_train)

        # Stage 2: outer loop
        B, R, _, E = x_BRCE.shape
        Cl = self.axial_embedder.num_cls_tokens
        D  = Cl * E
        x_BRD = self._init_x_brd(B, R, x_BRCE, Cl, E)

        n_loops = (
            num_loops if num_loops is not None
            else inference_num_blocks if inference_num_blocks is not None
            else self._sample_num_loops() if (self.training and self.icl_random_loops)
            else self.num_icl_loops
        )

        all_logits: List[torch.Tensor] = []

        use_lrs = self.use_loop_residual_scaling

        for loop_i in range(n_loops):

            # Truncated BPTT: stop gradient flow before the last-k window.
            if (
                self.training
                and self.tbptt_last_k is not None
                and loop_i == n_loops - self.tbptt_last_k
            ):
                x_BRCE = x_BRCE.detach()
                x_BRD  = x_BRD.detach()

            if self.loop_film is not None:
                x_BRD = self._apply_film(x_BRD, loop_i)

            skip_inject = self.inject_only_first_loop and loop_i > 0

            need_pre = use_lrs
            if need_pre:
                x_BRCE_pre = x_BRCE
                x_BRD_pre  = x_BRD

            use_chunked = (
                test_chunk_size is not None
                and not self.training
                and not need_pre
            )
            if use_chunked:
                x_BRCE, x_BRD = self._axial_step_chunked(
                    x_BRCE, x_BRD, num_train, y_col_emb,
                    skip_cell_label_inject=skip_inject,
                    test_chunk_size=test_chunk_size,
                    offload_x_brce=offload_x_brce,
                )
            else:
                x_BRCE, x_BRD = self._axial_step(
                    x_BRCE, x_BRD, num_train, y_col_emb,
                    skip_cell_label_inject=skip_inject,
                    offload_x_brce=offload_x_brce,
                    gradient_checkpointing=gradient_checkpointing,
                )

            if use_lrs:
                if self.loop_residual_scaling_fn == "entropy_adaptive":
                    loop_scale = self._compute_entropy_adaptive_scale(
                        x_BRD, num_train, y_BN, n_loops, loop_i
                    )
                    x_BRD  = x_BRD_pre  + loop_scale * (x_BRD  - x_BRD_pre)
                    x_BRCE = x_BRCE_pre + loop_scale * (x_BRCE - x_BRCE_pre)
                elif self.loop_residual_scaling_fn == "learned_mlp":
                    delta_D = x_BRD  - x_BRD_pre
                    delta_E = x_BRCE - x_BRCE_pre
                    scale_D, scale_E = self._compute_learned_mlp_scale(n_loops, loop_i, delta_D, delta_E)
                    x_BRD  = x_BRD_pre  + scale_D * delta_D
                    x_BRCE = x_BRCE_pre + scale_E * delta_E
                else:
                    loop_scale = self._compute_loop_scale(n_loops, loop_i)
                    x_BRD  = x_BRD_pre  + loop_scale * (x_BRD  - x_BRD_pre)
                    x_BRCE = x_BRCE_pre + loop_scale * (x_BRCE - x_BRCE_pre)

            if return_all_logits:
                normed    = self.output_norm(x_BRD)
                train_emb = normed[:, :num_train]
                test_emb  = normed[:, num_train:]
                logits    = self.many_class_decoder(train_emb, test_emb, y_BN)
                all_logits.append(torch.nan_to_num(logits.transpose(0, 1), nan=0.0))

        if return_all_logits:
            return all_logits

        if return_embedding:
            return x_BRD, y_BN, num_train

        # Stage 4
        x_BRD     = self.output_norm(x_BRD)
        train_emb = x_BRD[:, :num_train]
        test_emb  = x_BRD[:, num_train:]
        test_out  = self.many_class_decoder(train_emb, test_emb, y_BN)
        return torch.nan_to_num(test_out.transpose(0, 1), nan=0.0)


# ---------------------------------------------------------------------------
# LoopICLUnrolled — 6 fixed iterations, no weight sharing
# ---------------------------------------------------------------------------


class LoopICLUnrolled(LoopICL):
    """LoopICL with 6 fixed unrolled iterations — no weight sharing.

    In the base :class:`LoopICL` the outer loop reuses the *same*
    ``AxialEmbedderV3A`` weights on every iteration (looped / weight-shared
    transformer).  This class "unrolls" those iterations by setting
    ``axial_num_blocks=NUM_UNROLLED`` and ``num_icl_loops=1``:

    * ``AxialEmbedderV3A`` allocates ``NUM_UNROLLED`` independent block sets
      (within-col, cross-col, readout, ICL — each with distinct parameters).
    * The outer loop runs exactly once — there is no recurrent weight reuse.
    * Random-loop sampling and loop residual scaling are disabled.
    """

    NUM_UNROLLED: int = 6

    def __init__(self, config: LoopICLConfig | None = None, **kwargs) -> None:
        merged: dict = dict(vars(config)) if config is not None else {}
        merged.update(kwargs)
        merged.update(
            axial_num_blocks=self.NUM_UNROLLED,
            num_icl_loops=1,
            icl_random_loops=False,
            iter_embed=False,
            use_loop_residual_scaling=False,
        )
        super().__init__(**merged)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_loopicl(config: LoopICLConfig) -> LoopICL:
    """Construct LoopICL from a LoopICLConfig."""
    return LoopICL(config)


def build_loopicl_oneway(config: LoopICLConfig) -> LoopICL:
    """Construct a LoopICL with one-way column→row attention (use_oneway=True)."""
    config.use_oneway = True
    return LoopICL(config)
