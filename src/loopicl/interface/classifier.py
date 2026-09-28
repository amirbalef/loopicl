"""Sklearn-compatible classifier interface for the LoopICL tabular foundation model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
_DEFAULT_WEIGHTS    = str(_WEIGHTS_DIR / "loopicl-cls-stage2-step-30000.ckpt")
_DEFAULT_EE_WEIGHTS = str(_WEIGHTS_DIR / "loopicl-cls-ee-step-4000.ckpt")

from loopicl.interface._preprocessing import (
    EnsembleGenerator,
    XEncoder,
    _has_prefill,
    _quantize_cache,
    _wrapper_prefill,
    _wrapper_predict_cached,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    x = x / max(temperature, 1e-8)
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _compute_mr_bases(n_classes: int, max_classes: int) -> list:
    """Compute mixed-radix bases to cover n_classes with each base <= max_classes.

    Returns the smallest list of bases (each <= max_classes) whose product
    is >= n_classes.  Used for the many-class ECOC strategy.
    """
    import math
    if n_classes <= max_classes:
        return [n_classes]
    n_digits = math.ceil(math.log(n_classes) / math.log(max_classes))
    base = min(math.ceil(n_classes ** (1.0 / n_digits)), max_classes)
    bases = [base] * n_digits
    while int(np.prod(bases)) < n_classes:
        bases[0] = min(bases[0] + 1, max_classes)
        if int(np.prod(bases)) < n_classes:
            # need another digit
            n_digits += 1
            base = min(math.ceil(n_classes ** (1.0 / n_digits)), max_classes)
            bases = [base] * n_digits
    return bases


def _make_digit_map(n_classes: int, bases: list) -> np.ndarray:
    """Map each class index to its mixed-radix digit values.

    Returns an array of shape (n_classes, n_digits) where
    digit_map[c, d] = digit d of class c (least-significant digit first).
    """
    n_digits = len(bases)
    digit_map = np.zeros((n_classes, n_digits), dtype=np.int64)
    for c in range(n_classes):
        remainder = c
        for d in range(n_digits):
            digit_map[c, d] = remainder % bases[d]
            remainder //= bases[d]
    return digit_map


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Binary defaults (probability averaging)
_BINARY_NORM_METHODS = ["none", "power", "quantile", "robust"]
_BINARY_CLASS_SHUFFLE = "balanced_shuffle"

# Multiclass defaults (logit averaging)
_MULTI_NORM_METHODS = ["none", "power"]
_MULTI_CLASS_SHUFFLE = "shift"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class LoopICLClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible classifier wrapping the LoopICL tabular foundation model."""

    def __init__(
        self,
        model=None,
        weights_path: Optional[str] = None,
        use_early_exit: bool = False,
        device=None,
        dtype: str = "float32",
        num_loops: Optional[int] = None,
        mrl_dim: Optional[int] = None,
        loop_residual_scaling_fn: Optional[str] = None,
        exit_q: Optional[float] = None,
        n_estimators: int = 8,
        norm_methods: Optional[str | list] = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: Optional[str] = None,
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        batch_size: Optional[int] = 8,
        test_chunk_size: Optional[int] = None,
        cache_context: bool = False,
        quantize_kv_cache: bool = False,
        keep_cache_on_cpu: bool = False,
        subsample_samples: Optional[int] = None,
        max_features: Optional[int] = None,
        multiclass_strategy: str = "prob_avg",
        random_state: Optional[int] = 42,
    ):
        if model is None:
            if weights_path is None:
                weights_path = _DEFAULT_EE_WEIGHTS if use_early_exit else _DEFAULT_WEIGHTS
            model = self._load_checkpoint(weights_path, device)
            if use_early_exit and exit_q is None:
                exit_q = 0.5
        self.model = model
        self.weights_path = weights_path
        self.use_early_exit = use_early_exit
        self.device = device
        self.dtype = dtype
        self.num_loops = num_loops
        self.mrl_dim = mrl_dim
        self.loop_residual_scaling_fn = loop_residual_scaling_fn
        self.exit_q = exit_q
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.batch_size = batch_size
        self.test_chunk_size = test_chunk_size
        self.cache_context = cache_context
        self.quantize_kv_cache = quantize_kv_cache
        self.keep_cache_on_cpu = keep_cache_on_cpu
        self.subsample_samples = subsample_samples
        self.max_features = max_features
        self.multiclass_strategy = multiclass_strategy
        self.random_state = random_state

    @staticmethod
    def _load_checkpoint(weights_path: str, device=None):
        import dataclasses
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["config"])
        if "max_classes" in cfg and "max_num_classes" not in cfg:
            cfg["max_num_classes"] = cfg.pop("max_classes")

        if "gate_delta_out_dim" in cfg:
            from loopicl.models.loopicl.model_early_exit import (
                LoopICLEarlyExit,
                LoopICLEarlyExitConfig,
            )
            valid = set(LoopICLEarlyExitConfig.__dataclass_fields__)
            cfg = {k: v for k, v in cfg.items() if k in valid}
            model = LoopICLEarlyExit(**cfg)
        else:
            from loopicl.models.loopicl.model import LoopICL, LoopICLConfig
            valid = {f.name for f in dataclasses.fields(LoopICLConfig)}
            cfg = {k: v for k, v in cfg.items() if k in valid}
            model = LoopICL(LoopICLConfig(**cfg))

        state_key = "ema_state_dict" if "ema_state_dict" in ckpt else "state_dict"
        model.load_state_dict(ckpt[state_key])
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)
        model.eval()
        return model

    def _resolve_device(self):
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def _call_model(self, X_np: np.ndarray, y_np: np.ndarray, eval_pos: int) -> np.ndarray:
        device = self._resolve_device()
        X_t = torch.as_tensor(X_np.astype(np.float32), device=device)
        y_t = torch.as_tensor(y_np.astype(np.float32), device=device)

        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        amp_dtype = _dtype_map.get(self.dtype)

        kwargs: dict = {"return_logits": True}
        if self.num_loops is not None:
            kwargs["num_loops"] = self.num_loops
        if self.mrl_dim is not None:
            kwargs["mrl_dim"] = self.mrl_dim
        if self.exit_q is not None:
            kwargs["exit_q"] = self.exit_q

        orig_fn = None
        if self.loop_residual_scaling_fn is not None and hasattr(self.model, "loop_residual_scaling_fn"):
            orig_fn = self.model.loop_residual_scaling_fn
            self.model.loop_residual_scaling_fn = self.loop_residual_scaling_fn

        try:
            with torch.no_grad():
                if amp_dtype is not None and "cuda" in str(device):
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        if X_t.ndim == 3:
                            out = self.model(X_t, y_t, **kwargs)
                        else:
                            out = self.model(X_t.unsqueeze(0), y_t.unsqueeze(0), **kwargs).squeeze(0)
                else:
                    if X_t.ndim == 3:
                        out = self.model(X_t, y_t, **kwargs)
                    else:
                        out = self.model(X_t.unsqueeze(0), y_t.unsqueeze(0), **kwargs).squeeze(0)
        finally:
            if orig_fn is not None:
                self.model.loop_residual_scaling_fn = orig_fn

        return out.float().cpu().numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoopICLClassifier":
        """Fit preprocessing and ensemble configurations on the training data."""
        self.X_encoder_ = XEncoder()
        X_enc = self.X_encoder_.fit_transform(X)

        self.y_encoder_ = LabelEncoder()
        y_enc = self.y_encoder_.fit_transform(y).astype(np.float32)

        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_enc.shape[1]
        self._n_train = X_enc.shape[0]

        _max_classes = getattr(self.model, "max_classes", None)
        if _max_classes is not None and self.n_classes_ > _max_classes:
            self._mr_bases = _compute_mr_bases(self.n_classes_, _max_classes)
            self._mr_digit_map = _make_digit_map(self.n_classes_, self._mr_bases)
        else:
            self._mr_bases = None
            self._mr_digit_map = None

        is_binary = self.n_classes_ == 2
        use_prob_avg = is_binary or self.multiclass_strategy == "prob_avg"

        norm_methods = self.norm_methods
        class_shuffle_method = self.class_shuffle_method

        if norm_methods is None:
            norm_methods = _BINARY_NORM_METHODS if use_prob_avg else _MULTI_NORM_METHODS

        if class_shuffle_method is None:
            class_shuffle_method = _BINARY_CLASS_SHUFFLE if use_prob_avg else _MULTI_CLASS_SHUFFLE

        fingerprint_feature = not self.cache_context

        self.ensemble_generator_ = EnsembleGenerator(
            n_estimators=self.n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            subsample_samples=self.subsample_samples,
            max_features=self.max_features,
            fingerprint_feature=fingerprint_feature,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X_enc, y_enc)

        if self.cache_context and _has_prefill(self.model):
            self.context_caches_, self._context_cache_batch_size_ = (
                self._build_context_cache()
            )

        return self

    def _build_context_cache(self) -> tuple:
        all_X_train, all_y = self.ensemble_generator_.transform_train_only()
        n_estimators = len(all_X_train)
        batch_size = self.batch_size if self.batch_size is not None else n_estimators

        caches: dict = {}
        for start in range(0, n_estimators, batch_size):
            end = min(start + batch_size, n_estimators)
            batch_X_train = np.stack(all_X_train[start:end], axis=0)
            batch_y = np.stack(all_y[start:end], axis=0)

            cache = _wrapper_prefill(self.model, self._resolve_device(), self.dtype, batch_X_train, batch_y)
            if self.quantize_kv_cache:
                cache = _quantize_cache(cache)
            if self.keep_cache_on_cpu:
                cache = cache.to("cpu")
            caches[start] = cache

        return caches, batch_size

    def predict_proba(self, X: np.ndarray, num_loops: Optional[int] = None) -> np.ndarray:
        """Return class probabilities for X."""
        _prev_num_loops = self.num_loops
        if num_loops is not None:
            self.num_loops = num_loops
        try:
            return self._predict_proba_impl(X)
        finally:
            self.num_loops = _prev_num_loops

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["ensemble_generator_"])

        X_enc = self.X_encoder_.transform(X)

        if getattr(self, "_mr_bases", None) is not None:
            return self._predict_proba_mixed_radix(X_enc)

        n_test = X_enc.shape[0]
        n_train = self.ensemble_generator_.effective_n_train_
        is_binary = self.n_classes_ == 2
        use_prob_avg = is_binary or self.multiclass_strategy == "prob_avg"

        ensemble_data = self.ensemble_generator_.transform(X_enc)

        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_shuffles: list[np.ndarray] = []

        for norm_method, (X_ens, y_ens) in ensemble_data.items():
            shuffles = self.ensemble_generator_.class_shuffles_[norm_method]
            for i in range(X_ens.shape[0]):
                all_X.append(X_ens[i])
                all_y.append(y_ens[i])
                all_shuffles.append(np.asarray(shuffles[i], dtype=np.intp))

        n_estimators = len(all_X)
        accumulated = np.zeros((n_test, self.n_classes_), dtype=np.float32)
        test_chunk_size = self.test_chunk_size if self.test_chunk_size is not None else n_test

        use_cache_context = self.cache_context and hasattr(self, "context_caches_")
        use_two_phase = _has_prefill(self.model) and (
            self.test_chunk_size is not None or use_cache_context
        )

        batch_size = (
            self._context_cache_batch_size_
            if use_cache_context
            else (self.batch_size if self.batch_size is not None else n_estimators)
        )

        for start in range(0, n_estimators, batch_size):
            end = min(start + batch_size, n_estimators)

            if use_cache_context:
                cache = self.context_caches_[start]
            elif use_two_phase:
                batch_X_train = np.stack(
                    [x[:n_train] for x in all_X[start:end]], axis=0
                )
                batch_y = np.stack(all_y[start:end], axis=0)
                cache = _wrapper_prefill(self.model, self._resolve_device(), self.dtype, batch_X_train, batch_y)
                if self.quantize_kv_cache:
                    cache = _quantize_cache(cache)
                if self.keep_cache_on_cpu:
                    cache = cache.to("cpu")
            else:
                batch_X_train = np.stack(
                    [x[:n_train] for x in all_X[start:end]], axis=0
                )
                batch_y = np.stack(all_y[start:end], axis=0)

            for t_start in range(0, n_test, test_chunk_size):
                t_end = min(t_start + test_chunk_size, n_test)

                batch_X_chunk = np.stack(
                    [x[n_train + t_start: n_train + t_end] for x in all_X[start:end]],
                    axis=0,
                )

                if use_two_phase:
                    if self.keep_cache_on_cpu:
                        device = self._resolve_device()
                        cache = cache.to(device)
                    logits = _wrapper_predict_cached(
                        self.model, self._resolve_device(), self.dtype, batch_X_chunk, cache
                    )
                    if self.keep_cache_on_cpu:
                        cache = cache.to("cpu")
                else:
                    batch_X = np.concatenate(
                        [batch_X_train, batch_X_chunk], axis=1
                    )
                    logits = self._call_model(batch_X, batch_y, eval_pos=n_train)
                for i in range(end - start):
                    perm = all_shuffles[start + i]
                    out = logits[i][..., perm]

                    if use_prob_avg:
                        accumulated[t_start:t_end] += _softmax(
                            out, temperature=self.softmax_temperature
                        )
                    else:
                        accumulated[t_start:t_end] += out

        accumulated /= n_estimators

        if not use_prob_avg:
            accumulated = _softmax(accumulated, temperature=self.softmax_temperature)

        accum64 = accumulated.astype(np.float64)
        accum64 /= accum64.sum(axis=1, keepdims=True)
        return accum64.astype(np.float32)

    def _predict_proba_mixed_radix(self, X_enc: np.ndarray) -> np.ndarray:
        import torch

        n_test = X_enc.shape[0]
        n_train = self.ensemble_generator_.effective_n_train_
        batch_size = self.batch_size if self.batch_size is not None else 1

        # Collect all ensemble views (X with feature/norm diversity)
        ensemble_data = self.ensemble_generator_.transform(X_enc)

        all_X = []
        all_y_orig = []

        for norm_method, (X_ens, y_ens) in ensemble_data.items():
            shuffles = self.ensemble_generator_.class_shuffles_[norm_method]
            for i, perm in enumerate(shuffles):
                all_X.append(X_ens[i])
                if self.ensemble_generator_.class_shuffle_method == "none":
                    all_y_orig.append(y_ens[i].astype(np.int64))
                else:
                    inv_perm = np.argsort(perm)
                    all_y_orig.append(inv_perm[y_ens[i].astype(np.intp)])

        n_estimators = len(all_X)

        log_proba = np.zeros((n_test, self.n_classes_), dtype=np.float64)

        for d, base_d in enumerate(self._mr_bases):
            digit_map_d = self._mr_digit_map[:, d]  # (n_classes,): class → digit value

            accum_d = np.zeros((n_test, base_d), dtype=np.float64)

            for start in range(0, n_estimators, batch_size):
                end = min(start + batch_size, n_estimators)
                bs = end - start

                X_batch = np.stack(all_X[start:end], axis=0)  # (bs, n_total, n_feat)
                y_digit_batch = np.stack(
                    [digit_map_d[all_y_orig[j]] for j in range(start, end)],
                    axis=0,
                ).astype(np.float32)  # (bs, n_train)

                logits = self._call_model(X_batch, y_digit_batch, eval_pos=n_train)

                accum_d += logits[:, :, :base_d].astype(np.float64).sum(axis=0)

            accum_d /= n_estimators  # average logits

            # Log-softmax over digit classes
            accum_d -= accum_d.max(axis=-1, keepdims=True)
            log_probs_d = accum_d - np.log(np.exp(accum_d).sum(axis=-1, keepdims=True))

            log_proba += log_probs_d[:, digit_map_d]

        log_proba -= log_proba.max(axis=-1, keepdims=True)
        proba = np.exp(log_proba)          # float64
        proba /= proba.sum(axis=-1, keepdims=True)
        return proba.astype(np.float32)

    def predict(self, X: np.ndarray, num_loops: Optional[int] = None) -> np.ndarray:
        """Predict class labels for X."""
        proba = self.predict_proba(X, num_loops=num_loops)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
