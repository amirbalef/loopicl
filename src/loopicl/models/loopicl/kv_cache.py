"""KV cache data structures for LoopICL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass
class KVCacheEntry:
    """A single key-value cache entry for one attention layer.

    Attributes:
        key:   Cached key projections,   shape (B, N_train, num_kv_heads, head_dim).
        value: Cached value projections, shape (B, N_train, num_kv_heads, head_dim).
    """

    key: Tensor | None = None
    value: Tensor | None = None

    def is_valid(self) -> bool:
        return self.key is not None and self.value is not None

    def to(self, device: torch.device | str) -> KVCacheEntry:
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key.to(device), value=self.value.to(device))


@dataclass
class KVCache:
    """Maps layer indices to KVCacheEntry objects."""

    kv: dict[int, KVCacheEntry] = field(default_factory=dict)

    def is_populated(self) -> bool:
        return any(entry.is_valid() for entry in self.kv.values())

    def to(self, device: torch.device | str) -> KVCache:
        return KVCache(kv={idx: entry.to(device) for idx, entry in self.kv.items()})


@dataclass
class LoopICLTrainCache:
    """Cached train-side state for fast test-time predictions.

    Created by ``LoopICL.prefill()``; consumed by ``LoopICL.predict_cached()``.

    The cache stores everything needed to process new test rows without
    re-running any train-side computation:

    - Preprocessing stats (scaler, NaN-imputation means, sorted train features)
    - Per-column conditioning outputs already applied to x_BRCE
      (log-hist and discriminative-hist conditioners produce the same vector
       for every row, so they are computed once and cached here)
    - Per-loop inducing hidden states from each within-col block
    - Per-loop KV entries from each ICL attention block
    - Final normalised train embeddings for the decoder

    Attributes
    ----------
    scaler_params       : dict with "mean" and "std" tensors (shape (B, C) each)
                          used to normalise raw test features.
    feature_means       : (B, C) per-feature finite mean for NaN imputation.
    x_sorted_BCN        : (B, C, N_train) sorted training features used by the
                          Fourier-quantile encoder to rank test values.
    log_hist_cond       : (B, C, E) cached output of AugmentedLogHistConditioner,
                          or None when ``use_log_hist_conditioning=False``.
    disc_hist_cond      : (B, C, E) or None. Cached output of
                          DiscriminativeHistConditioner (None when disabled).
    y_col_emb           : (B, N_train, E) label column embedding kept for
                          within-axial-block label-inject projections.
    within_hiddens      : flat list of (B*C, M, E) inducing hidden tensors,
                          one entry per within-col block call per outer loop.
                          Length = num_loops * num_axial_blocks * num_within_per_round.
    icl_kvs             : flat list of KVCacheEntry, one per ICL block call per
                          outer loop.
                          Length = num_loops * num_axial_blocks * num_icl_per_round.
    x_BRD_train_normed  : (B, N_train, D) output_norm applied to the final train
                          row-stream tensor; fed directly to the decoder.
    y_BN                : (B, N_train) integer training labels.
    num_train           : number of training rows.
    num_loops           : number of outer ICL loops used during prefill.
    """

    scaler_params:       dict
    feature_means:       Tensor
    x_sorted_BCN:        Tensor
    log_hist_cond:       Tensor | None
    disc_hist_cond:      Tensor | None
    y_col_emb:           Tensor
    within_hiddens:      list[Tensor]
    icl_kvs:             list[KVCacheEntry]
    x_BRD_train_normed:  Tensor
    y_BN:                Tensor
    num_train:           int
    num_loops:           int

    def to(self, device: torch.device | str) -> LoopICLTrainCache:
        """Move all tensors in the cache to ``device``."""
        def _t(x: Tensor) -> Tensor:
            return x.to(device)

        return LoopICLTrainCache(
            scaler_params={k: _t(v) for k, v in self.scaler_params.items()},
            feature_means=_t(self.feature_means),
            x_sorted_BCN=_t(self.x_sorted_BCN),
            log_hist_cond=_t(self.log_hist_cond) if self.log_hist_cond is not None else None,
            disc_hist_cond=_t(self.disc_hist_cond) if self.disc_hist_cond is not None else None,
            y_col_emb=_t(self.y_col_emb),
            within_hiddens=[_t(h) for h in self.within_hiddens],
            icl_kvs=[kv.to(device) for kv in self.icl_kvs],
            x_BRD_train_normed=_t(self.x_BRD_train_normed),
            y_BN=_t(self.y_BN),
            num_train=self.num_train,
            num_loops=self.num_loops,
        )
