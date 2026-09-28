"""LoopICLEarlyExit — LoopICL with a test-set-agnostic exit gate.

Ported from loopicl/models/loopicl/model_early_exit.py.

Design principles
-----------------
1. Test-set agnostic: the gate observes only training-row embeddings.
   No decoder forward pass is needed inside the loop during inference, so
   the decoder is called exactly once — at the loop where the model exits.
   During training (return_exit_info=True) the decoder is still called
   at every step to produce the per-loop logits needed for the BCE/Stage-I loss.

2. Task-agnostic: the same gate architecture and training procedure work
   for both classification (CE loss) and regression (pinball loss).

Gate signals (all train-set only):
  row_compressor   MLP(D → max(4*c,32) → c) per row, mean-pooled → state(c).
  gate_use_delta   Append (state_t − state_{t−1}).
  gate_use_loop_index  sinusoidal loop-position embedding.
  gate_history_len     past λ scalars.
  gate_train_size_dim  sinusoidal embedding of log(num_train).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .conditioning import (
    _N_HIST_BINS,
    _adaptive_bin_geometry,
    _compute_moments,
    _soft_log_histogram,
)
from .model import LoopICL, LoopICLConfig

__all__ = ["EarlyExitGate", "LoopICLEarlyExitConfig", "LoopICLEarlyExit"]


# ---------------------------------------------------------------------------
# Gate module  (must match checkpoint key layout exactly)
# ---------------------------------------------------------------------------


class EarlyExitGate(nn.Module):
    """Test-set-agnostic exit gate.

    Maps per-loop training-row CLS embeddings to a scalar exit probability
    λₜ ∈ (0, 1) via a 2-layer MLP.  No test-set signals are used.

    Module layout (checkpoint keys):
      row_compressor.{0,2}.weight  — Linear(D, hidden_c) → ReLU → Linear(hidden_c, c)
                                     both bias=False; hidden_c = max(4*c, 32)
      mlp.{0,2}.{weight,bias}      — Linear(gate_in, H) → ReLU → Linear(H, 1)
      _loop_freqs                  — buffer (if gate_use_loop_index)
      _train_size_freqs            — buffer (if gate_train_size_dim > 0)
    """

    def __init__(
        self,
        D: int,
        gate_hidden_dim: int = 64,
        gate_use_delta: bool = True,
        gate_delta_out_dim: int = 8,
        gate_use_loop_index: bool = False,
        gate_loop_index_dim: int = 16,
        gate_history_len: int = 0,
        gate_train_size_dim: int = 8,
    ) -> None:
        super().__init__()

        self.use_delta      = gate_use_delta
        self.compress_dim   = gate_delta_out_dim
        self.use_loop_index = gate_use_loop_index
        self.loop_index_dim = gate_loop_index_dim
        self.history_len    = gate_history_len
        self.train_size_dim = gate_train_size_dim

        # Per-row MLP compressor: D → hidden_c → compress_dim, then mean-pool.
        hidden_c = max(gate_delta_out_dim * 4, 32)
        self.row_compressor = nn.Sequential(
            nn.Linear(D, hidden_c, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_c, gate_delta_out_dim, bias=False),
        )

        # Gate input: [state, (delta), loop_emb, history, train_size_emb]
        gate_in  = gate_delta_out_dim
        gate_in += gate_delta_out_dim  if gate_use_delta      else 0
        gate_in += gate_loop_index_dim if gate_use_loop_index else 0
        gate_in += gate_history_len
        gate_in += gate_train_size_dim

        H = gate_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(gate_in, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )
        # Neutral start: sigmoid(0) = 0.5
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        if gate_use_loop_index:
            dim   = gate_loop_index_dim
            freqs = torch.exp(
                -torch.arange(0, dim, 2, dtype=torch.float32)
                * (math.log(10_000.0) / dim)
            )
            self.register_buffer("_loop_freqs", freqs)

        if gate_train_size_dim > 0:
            dim   = gate_train_size_dim
            freqs = torch.exp(
                -torch.arange(0, dim, 2, dtype=torch.float32)
                * (math.log(1_000.0) / dim)
            )
            self.register_buffer("_train_size_freqs", freqs)

    def forward(
        self,
        x_BRD: torch.Tensor,                  # (B, R, D) full sequence
        num_train: int,
        loop_i: int,
        prev_compressed: Optional[torch.Tensor],  # (B, compress_dim) or None
        lambda_list: List[torch.Tensor],           # past λ tensors, each (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(lambda_t, curr_compressed)``."""
        x_train = x_BRD[:, :num_train]           # (B, N_train, D)
        B = x_train.shape[0]

        curr_compressed = self.row_compressor(x_train).mean(dim=1)  # (B, c)
        feats = curr_compressed

        if self.use_delta:
            if prev_compressed is not None:
                delta_feat = curr_compressed - prev_compressed
            else:
                delta_feat = curr_compressed.new_zeros(B, self.compress_dim)
            feats = torch.cat([feats, delta_feat], dim=-1)

        if self.use_loop_index:
            t      = torch.tensor(loop_i, dtype=feats.dtype, device=feats.device)
            angles = t * self._loop_freqs
            loop_emb = torch.cat([angles.sin(), angles.cos()])
            feats = torch.cat([feats, loop_emb.unsqueeze(0).expand(B, -1)], dim=-1)

        if self.history_len > 0:
            n   = min(len(lambda_list), self.history_len)
            pad = self.history_len - n
            hist = torch.stack(lambda_list[-n:], dim=1).detach() if n > 0 else feats.new_empty(B, 0)
            if pad > 0:
                hist = torch.cat([feats.new_zeros(B, pad), hist], dim=1)
            feats = torch.cat([feats, hist], dim=-1)

        if self.train_size_dim > 0:
            log_n  = feats.new_tensor(math.log(max(num_train, 1)))
            angles = log_n * self._train_size_freqs
            size_emb = torch.cat([angles.sin(), angles.cos()])
            feats = torch.cat([feats, size_emb.unsqueeze(0).expand(B, -1)], dim=-1)

        lambda_t = torch.sigmoid(self.mlp(feats)).squeeze(-1)  # (B,)
        return lambda_t, curr_compressed.detach()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoopICLEarlyExitConfig(LoopICLConfig):
    """LoopICLConfig plus v2 early-exit gate parameters."""

    exit_q:              float = 0.5
    per_sample_exit:     bool  = False
    gate_hidden_dim:     int   = 64
    gate_use_delta:      bool  = True
    gate_delta_out_dim:  int   = 8
    gate_use_loop_index: bool  = False
    gate_loop_index_dim: int   = 16
    gate_history_len:    int   = 0
    gate_train_size_dim: int   = 8


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LoopICLEarlyExit(LoopICL):
    """LoopICL with test-set-agnostic early-exit gate v2.

    At each outer loop t an EarlyExitGate produces λ_t from training-row
    embeddings only.  At inference the decoder is called once, at the exit
    loop.  The gate module is named ``exit_gate`` so that the trainer can
    freeze all other parameters by checking ``name.startswith("exit_gate")``.
    """

    def __init__(
        self,
        config: LoopICLEarlyExitConfig | None = None,
        **kwargs,
    ) -> None:
        # Merge config + extra kwargs
        merged = dict(vars(config)) if config is not None else {}
        merged.update(kwargs)

        def _g(field, default):
            return merged.get(field, default)

        # Stash gate params before LoopICL.__init__ (which silently ignores them)
        gate_kwargs = dict(
            gate_hidden_dim     = _g("gate_hidden_dim",     64),
            gate_use_delta      = _g("gate_use_delta",      True),
            gate_delta_out_dim  = _g("gate_delta_out_dim",  8),
            gate_use_loop_index = _g("gate_use_loop_index", False),
            gate_loop_index_dim = _g("gate_loop_index_dim", 16),
            gate_history_len    = _g("gate_history_len",    0),
            gate_train_size_dim = _g("gate_train_size_dim", 8),
        )
        self._init_exit_q          = _g("exit_q",          0.5)
        self._init_per_sample_exit = _g("per_sample_exit", False)

        super().__init__(**merged)  # LoopICL filters to LoopICLConfig fields

        self.exit_q          = self._init_exit_q
        self.per_sample_exit = self._init_per_sample_exit

        # Build exit gate using the resolved row-stream dim
        self.exit_gate = EarlyExitGate(D=self._D, **gate_kwargs)

        # Expose max_classes for the classifier's many-class check
        self.max_classes: int = self.cfg.max_num_classes

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
        return_exit_info: bool = False,
        return_all_logits: bool = False,
        exit_q: float | None = None,
        per_sample_exit: bool | None = None,
        gradient_checkpointing: bool | None = None,
        **_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor],
               Tuple[List[torch.Tensor], torch.Tensor]]:
        """Looped forward with test-set-agnostic exit gate.

        Parameters
        ----------
        X : (B, R, C)
        y_train : (B, N_train)
        single_eval_pos / train_size : number of training rows
        num_loops : override number of outer loops
        return_exit_info : return (all_logits, all_lambdas) for gate training
        exit_q : per-call inference threshold override
        """
        if gradient_checkpointing is None:
            gradient_checkpointing = self.loop_gradient_checkpointing

        x_RBC = X.transpose(0, 1)
        y_NB  = y_train.transpose(0, 1)
        num_train = (
            int(single_eval_pos) if single_eval_pos is not None else
            int(train_size)      if train_size      is not None else
            int(y_NB.shape[0])
        )

        # ---- Stage 0: preprocess ----
        x_BRC, nan_ind_BRC = self._preprocess(x_RBC, num_train)
        x_BRCG = self._group_features(x_BRC, nan_ind_BRC)

        x_train_BCN  = x_BRC[:, :num_train].permute(0, 2, 1)
        x_sorted_BCN = x_train_BCN.sort(dim=-1).values

        centers, widths = _adaptive_bin_geometry(x_sorted_BCN, self._n_bins)
        log_hist_BCK    = _soft_log_histogram(x_train_BCN, centers, widths, num_train)
        moments_BC4     = _compute_moments(x_train_BCN)

        # ---- Stage 1: embedding + conditioning ----
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

        # ---- Stage 2: outer loop with gate ----
        B, R, _, E = x_BRCE.shape
        Cl = self.axial_embedder.num_cls_tokens
        x_BRD = self._init_x_brd(B, R, x_BRCE, Cl, E)

        n_loops = (
            num_loops            if num_loops            is not None else
            inference_num_blocks if inference_num_blocks is not None else
            self._sample_num_loops() if (self.training and self.icl_random_loops) else
            self.num_icl_loops
        )

        use_lrs = self.use_loop_residual_scaling

        all_logits:      List[torch.Tensor]     = []
        lambda_list:     List[torch.Tensor]     = []
        prev_compressed: Optional[torch.Tensor] = None

        # Inference early-exit state
        _exit_q     = exit_q        if exit_q        is not None else self.exit_q
        _per_sample = per_sample_exit if per_sample_exit is not None else self.per_sample_exit
        use_early_exit = (
            not self.training
            and not return_exit_info
            and _exit_q < 1.0
        )
        survival = x_BRD.new_ones(B)
        if use_early_exit and _per_sample:
            final_x_BRD = torch.empty_like(x_BRD)
            exited = torch.zeros(B, dtype=torch.bool, device=x_BRD.device)
        else:
            final_x_BRD = exited = None

        for loop_i in range(n_loops):
            # Truncated BPTT
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
            if use_lrs:
                x_BRCE_pre, x_BRD_pre = x_BRCE, x_BRD

            x_BRCE, x_BRD = self._axial_step(
                x_BRCE, x_BRD, num_train, y_col_emb,
                skip_cell_label_inject=skip_inject,
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
                    scale_D, scale_E = self._compute_learned_mlp_scale(
                        n_loops, loop_i, delta_D, delta_E
                    )
                    x_BRD  = x_BRD_pre  + scale_D * delta_D
                    x_BRCE = x_BRCE_pre + scale_E * delta_E
                else:
                    loop_scale = self._compute_loop_scale(n_loops, loop_i)
                    x_BRD  = x_BRD_pre  + loop_scale * (x_BRD  - x_BRD_pre)
                    x_BRCE = x_BRCE_pre + loop_scale * (x_BRCE - x_BRCE_pre)

            # Gate: train-row embeddings only (no test-set signals)
            lambda_t, prev_compressed = self.exit_gate(
                x_BRD, num_train, loop_i, prev_compressed, lambda_list
            )
            lambda_list.append(lambda_t)

            # Decoder only when training loss needs per-loop logits
            if return_exit_info or return_all_logits:
                normed    = self.output_norm(x_BRD)
                train_emb = normed[:, :num_train]
                test_emb  = normed[:, num_train:]
                logits    = self.many_class_decoder(train_emb, test_emb, y_BN)
                all_logits.append(torch.nan_to_num(logits.transpose(0, 1), nan=0.0))

            # Inference: CDF-based exit
            if use_early_exit:
                survival = survival * (1.0 - lambda_t.detach())
                cdf = 1.0 - survival
                if _per_sample:
                    newly = (cdf > _exit_q) & ~exited
                    if newly.any():
                        final_x_BRD[newly] = x_BRD[newly]
                        exited |= newly
                    if exited.all():
                        break
                else:
                    if cdf.mean().item() > _exit_q:
                        break

        # Fold remaining per-sample exits
        if use_early_exit and _per_sample:
            remaining = ~exited
            if remaining.any():
                final_x_BRD[remaining] = x_BRD[remaining]
            x_BRD = final_x_BRD

        if return_exit_info:
            return all_logits, torch.stack(lambda_list, dim=0)  # (T, B)
        if return_all_logits:
            return all_logits

        # Stage 4: decoder called ONCE at the exit loop
        x_BRD     = self.output_norm(x_BRD)
        train_emb = x_BRD[:, :num_train]
        test_emb  = x_BRD[:, num_train:]
        test_out  = self.many_class_decoder(train_emb, test_emb, y_BN)
        return torch.nan_to_num(test_out.transpose(0, 1), nan=0.0)
