"""Preprocessing and ensemble utilities for LoopICLClassifier."""

from __future__ import annotations

import itertools
import random
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as _SKPipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


# ---------------------------------------------------------------------------
# Feature encoding (DataFrame → float32 numpy)
# ---------------------------------------------------------------------------


def encode_X(X) -> np.ndarray:
    """Convert a DataFrame or array to a float32 numpy array."""
    if not hasattr(X, "columns"):
        return np.asarray(X, dtype=np.float32)

    cat_cols = make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)
    cat_pos = [X.columns.get_loc(c) for c in cat_cols]
    num_cols = make_column_selector(dtype_include="number")(X)
    num_pos = [X.columns.get_loc(c) for c in num_cols]

    tfm = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    dtype=np.float32,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_pos,
            ),
            ("num", SimpleImputer(strategy="median"), num_pos),
        ]
    )
    return tfm.fit_transform(X).astype(np.float32)


class XEncoder:
    """Fit-once feature encoder for DataFrames and numpy arrays."""

    def fit_transform(self, X) -> np.ndarray:
        if hasattr(X, "columns"):
            cat_cols = make_column_selector(
                dtype_include=["string", "object", "category", "boolean"]
            )(X)
            cat_pos = [X.columns.get_loc(c) for c in cat_cols]
            num_cols = make_column_selector(dtype_include="number")(X)
            num_pos = [X.columns.get_loc(c) for c in num_cols]
            self._col_tfm = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OrdinalEncoder(
                            dtype=np.int64,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                        cat_pos,
                    ),
                    ("num", SimpleImputer(), num_pos),
                ]
            )
            X_enc = self._col_tfm.fit_transform(X)
            self._imputer = None
        else:
            self._col_tfm = None
            X_enc = np.asarray(X)
            self._imputer = SimpleImputer(strategy="mean")
            X_enc = self._imputer.fit_transform(X_enc)

        return X_enc

    def transform(self, X) -> np.ndarray:
        if self._col_tfm is not None:
            return self._col_tfm.transform(X)
        X_enc = np.asarray(X)
        if self._imputer is not None:
            X_enc = self._imputer.transform(X_enc)
        return X_enc


# ---------------------------------------------------------------------------
# Unique-feature filter
# ---------------------------------------------------------------------------


class UniqueFeatureFilter:
    """Drop features whose training-set unique-value count is ≤ threshold."""

    def __init__(self, threshold: int = 1):
        self.threshold = threshold

    def fit(self, X: np.ndarray) -> "UniqueFeatureFilter":
        if X.shape[0] <= self.threshold:
            self._mask = np.ones(X.shape[1], dtype=bool)
        else:
            self._mask = np.array(
                [len(np.unique(X[:, j])) > self.threshold for j in range(X.shape[1])]
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self._mask]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


class _CustomStandardScaler:
    """Z-score scaler with symmetric clipping to [-100, 100]."""

    def fit(self, X: np.ndarray) -> "_CustomStandardScaler":
        self._mean = np.mean(X, axis=0)
        self._scale = np.std(X, axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self._mean) / self._scale, -100.0, 100.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _OutlierClipper:
    """Clip values using a two-stage Z-score method with log-soft bounds."""

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, X: np.ndarray) -> "_OutlierClipper":
        means = np.nanmean(X, axis=0)
        stds = np.maximum(np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0), 1e-6)

        X_clean = X.copy()
        X_clean[np.abs(X - means) > self.threshold * stds] = np.nan

        self._means = np.nanmean(X_clean, axis=0)
        self._stds = np.maximum(
            np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0), 1e-6
        )
        self._lo = self._means - self.threshold * self._stds
        self._hi = self._means + self.threshold * self._stds
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.maximum(-np.log1p(np.abs(X)) + self._lo, X)
        X = np.minimum(np.log1p(np.abs(X)) + self._hi, X)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _RTDLQuantileTransformer:
    """Quantile transformer with training-noise injection."""

    def __init__(
        self,
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "_RTDLQuantileTransformer":
        n_q = max(min(X.shape[0] // 30, self.n_quantiles), 10)
        rng = np.random.default_rng(self.random_state)
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        X_noisy = X + noise_std * rng.standard_normal(X.shape)
        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_q,
            subsample=1_000_000_000,
            random_state=self.random_state,
        )
        self._ss = StandardScaler()
        self._qt.fit(X_noisy)
        self._ss.fit(self._qt.transform(X_noisy))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._ss.transform(self._qt.transform(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PreprocessingPipeline:
    """Standardise → normalise → clip outliers."""

    def __init__(
        self,
        normalization_method: str = "power",
        outlier_threshold: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "PreprocessingPipeline":
        if self.normalization_method in ("raw", "squashing_scaler_default"):
            self._scaler = None
            X_scaled = np.asarray(X, dtype=np.float64)
        else:
            self._scaler = _CustomStandardScaler()
            X_scaled = self._scaler.fit_transform(X)

        if self.normalization_method in ("raw", "none"):
            self._normalizer = None
            X_norm = X_scaled
        elif self.normalization_method == "squashing_scaler_default":
            from tabpfn.preprocessing.steps.squashing_scaler_transformer import (
                SquashingScaler,
            )
            self._normalizer = SquashingScaler(max_absolute_value=3.0)
            X_norm = self._normalizer.fit_transform(X_scaled)
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=3.0, neginf=-3.0)
        else:
            self._X_min = X_scaled.min(axis=0, keepdims=True)
            self._X_max = X_scaled.max(axis=0, keepdims=True)

            if self.normalization_method in ("power", "safepower"):
                self._normalizer = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                self._normalizer = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, X_scaled.shape[0]),
                    random_state=self.random_state,
                )
            elif self.normalization_method in ("quantile_uni", "quantile_uni_append"):
                self._normalizer = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=min(1000, X_scaled.shape[0]),
                    random_state=self.random_state,
                )
            elif self.normalization_method == "quantile_rtdl":
                self._normalizer = _RTDLQuantileTransformer(random_state=self.random_state)
            elif self.normalization_method == "robust":
                self._normalizer = RobustScaler(unit_variance=True)
            else:
                raise ValueError(
                    f"Unknown normalization_method: {self.normalization_method!r}. "
                    "Use 'raw', 'none', 'power', 'safepower', 'quantile', 'quantile_uni', "
                    "'quantile_uni_append', 'quantile_rtdl', 'robust', or "
                    "'squashing_scaler_default'."
                )
            X_norm = self._normalizer.fit_transform(X_scaled)
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

            if self.normalization_method == "quantile_uni_append":
                X_norm = np.concatenate([X_norm, X_scaled], axis=1)

        if self.normalization_method == "squashing_scaler_default" and X_norm.shape[1] >= 2:
            n_s, n_f = X_norm.shape
            n_components = max(1, min(n_s // 10 + 1, n_f // 4))
            self._svd = _SKPipeline([
                ("ss", StandardScaler(with_mean=False)),
                ("svd", TruncatedSVD(
                    n_components=n_components,
                    algorithm="arpack",
                    random_state=self.random_state,
                )),
            ])
            svd_feats = self._svd.fit_transform(X_norm)
            X_norm = np.concatenate([X_norm, svd_feats], axis=1)
        else:
            self._svd = None

        if self.outlier_threshold is not None:
            self._clipper = _OutlierClipper(threshold=self.outlier_threshold)
            self.X_transformed_ = self._clipper.fit_transform(X_norm)
        else:
            self._clipper = None
            self.X_transformed_ = X_norm
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        else:
            X = np.asarray(X, dtype=np.float64)

        if self._normalizer is not None:
            if self.normalization_method == "squashing_scaler_default":
                X = self._normalizer.transform(X)
                X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
            elif self.normalization_method == "quantile_uni_append":
                X_std = X.copy()  # save standardised features for appending
                try:
                    X = self._normalizer.transform(X)
                except ValueError:
                    X = np.clip(X, self._X_min, self._X_max)
                    X = self._normalizer.transform(X)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = np.concatenate([X, X_std], axis=1)
            else:
                try:
                    X = self._normalizer.transform(X)
                except ValueError:
                    X = np.clip(X, self._X_min, self._X_max)
                    X = self._normalizer.transform(X)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self._svd is not None:
            X = np.concatenate([X, self._svd.transform(X)], axis=1)

        if self._clipper is not None:
            X = self._clipper.transform(X)
        return X


# ---------------------------------------------------------------------------
# Fingerprint feature
# ---------------------------------------------------------------------------


def _compute_fingerprint(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Return a per-row fingerprint column for the concatenated train+test array."""
    import hashlib
    from collections import defaultdict

    _CONST = 2 ** 64 - 1
    _ROUND = 12

    n_train = X_train.shape[0]
    X_all = np.concatenate([X_train, X_test], axis=0)
    X_rounded = np.ascontiguousarray(np.round(X_all, decimals=_ROUND))

    salt = X_rounded.shape[0] * X_rounded.shape[1]
    salt_bytes = salt.to_bytes(8, "little", signed=False)

    def _hash(row_bytes: bytes, offset: int) -> float:
        ob = (salt + offset).to_bytes(8, "little", signed=False)
        h = int(hashlib.sha256(row_bytes + ob).hexdigest(), 16)
        return (h & _CONST) / _CONST

    fingerprints = np.zeros(len(X_all), dtype=np.float32)

    seen: set[float] = set()
    hash_counter: dict[bytes, int] = defaultdict(int)
    for i in range(n_train):
        row = X_rounded[i].tobytes()
        h_base = _hash(row, 0)
        offset = hash_counter[row]
        h = h_base if offset == 0 else _hash(row, offset)
        while h in seen:
            offset += 1
            h = _hash(row, offset)
        fingerprints[i] = h
        seen.add(h)
        hash_counter[row] = offset + 1

    for i in range(n_train, len(X_all)):
        row = X_rounded[i].tobytes()
        fingerprints[i] = _hash(row, 0)

    return fingerprints


# ---------------------------------------------------------------------------
# Shuffler
# ---------------------------------------------------------------------------


class Shuffler:
    """Generate permutation patterns for ensemble diversity."""

    _MAX_LATIN = 4000  # fall back to random above this
    _OVERSAMPLE_FACTOR = 4  # for balanced_shuffle

    def __init__(
        self,
        n_elements: int,
        method: str = "latin",
        random_state: Optional[int] = None,
    ):
        self.n_elements = n_elements
        self.method = method
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> List[List[int]]:
        indices = list(range(self.n_elements))

        method = self.method
        if self.n_elements > self._MAX_LATIN and method == "latin":
            method = "random"

        if method == "none" or n_estimators == 1:
            return [indices]

        if method == "shift":
            return [indices[-i:] + indices[:-i] for i in range(self.n_elements)]

        if method == "balanced_shuffle":
            return self._balanced_shuffle(n_estimators)

        rng = random.Random(self.random_state)

        if method == "random":
            if self.n_elements <= 5:
                all_perms = [list(p) for p in itertools.permutations(indices)]
                return rng.sample(all_perms, min(n_estimators, len(all_perms)))
            return [rng.sample(indices, self.n_elements) for _ in range(n_estimators)]

        if method == "latin":
            return self._latin_squares(rng)

        raise ValueError(
            f"Unknown method: {method!r}. "
            "Use 'none', 'shift', 'latin', 'random', or 'balanced_shuffle'."
        )

    def _balanced_shuffle(self, n_estimators: int) -> List[List[int]]:
        rng = np.random.default_rng(self.random_state)
        noise = rng.random((n_estimators * self._OVERSAMPLE_FACTOR, self.n_elements))
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)  # (n_unique, n_elements)

        balance_count = n_estimators // len(uniqs)
        result = [list(uniqs[i]) for i in range(len(uniqs))] * balance_count

        remainder = n_estimators % len(uniqs)
        if remainder:
            chosen = rng.choice(len(uniqs), size=remainder, replace=False)
            result += [list(uniqs[i]) for i in chosen]

        rng_py = random.Random(self.random_state)
        rng_py.shuffle(result)
        return result

    def _latin_squares(self, rng: random.Random) -> List[List[int]]:
        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols]
            sym = rng.choice(symbols)
            symbols = [s for s in symbols if s != sym]
            square = _rls(symbols)
            square.append(square[0].copy())
            for i in range(n):
                square[i].insert(i, sym)
            return square

        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix)
            rng.shuffle(square)
            trans = list(zip(*square))
            rng.shuffle(trans)
            return trans

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100_000)
        try:
            symbols = list(range(self.n_elements))
            square = _rls(symbols)
            shuffles = _shuffle_transpose_shuffle(square)
        finally:
            sys.setrecursionlimit(old_limit)

        return [list(row) for row in shuffles]


# ---------------------------------------------------------------------------
# Row subsampling
# ---------------------------------------------------------------------------


def _stratified_subsample(
    y: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ``n_samples`` row indices stratified by class label.

    Each class gets a share proportional to its frequency (minimum 1 per class).
    """
    classes, counts = np.unique(y, return_counts=True)
    indices = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_cls = max(1, round(n_samples * count / len(y)))
        n_cls = min(n_cls, len(cls_idx))
        indices.append(rng.choice(cls_idx, n_cls, replace=False))
    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return selected[:n_samples]


# ---------------------------------------------------------------------------
# EnsembleGenerator
# ---------------------------------------------------------------------------


class EnsembleGenerator:
    """Generate diverse ensemble configurations for LoopICL inference."""

    def __init__(
        self,
        n_estimators: int,
        norm_methods=None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: Optional[float] = None,
        subsample_samples: Optional[int] = None,
        max_features: Optional[int] = None,
        fingerprint_feature: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.subsample_samples = subsample_samples
        self.max_features = max_features
        self.fingerprint_feature = fingerprint_feature
        self.random_state = random_state

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleGenerator":
        """Fit preprocessing pipelines and generate shuffle + row configurations."""
        if self.norm_methods is None:
            norm_methods = ["none", "power"]
        elif isinstance(self.norm_methods, str):
            norm_methods = [self.norm_methods]
        else:
            norm_methods = list(self.norm_methods)
        self._norm_methods = norm_methods

        # Drop constant features
        self._unique_filter = UniqueFeatureFilter()
        X = self._unique_filter.fit_transform(X)
        self._X_train = X
        self._y_train = y
        self._n_train = X.shape[0]
        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))

        # Resolve effective training size (after subsampling)
        use_subsample = (
            self.subsample_samples is not None
            and self.subsample_samples < self._n_train
        )
        self.effective_n_train_ = self.subsample_samples if use_subsample else self._n_train

        # Generate per-estimator row indices for subsampling
        if use_subsample:
            rng_np = np.random.default_rng(self.random_state)
            self._row_indices: list[np.ndarray] | None = [
                _stratified_subsample(y.astype(int), self.subsample_samples, rng_np)
                for _ in range(self.n_estimators)
            ]
        else:
            self._row_indices = None

        use_col_subsample = (
            self.max_features is not None and self.max_features < self._n_features
        )
        self.effective_n_features_ = self.max_features if use_col_subsample else self._n_features
        if use_col_subsample:
            rng_py = random.Random(self.random_state)
            all_cols = list(range(self._n_features))
            self._col_indices: list[np.ndarray] | None = [
                np.array(rng_py.sample(all_cols, self.max_features))
                for _ in range(self.n_estimators)
            ]
        else:
            self._col_indices = None

        feat_shuffler = Shuffler(
            n_elements=self._n_features,
            method=self.feat_shuffle_method,
            random_state=self.random_state,
        )
        class_shuffler = Shuffler(
            n_elements=self._n_classes,
            method=self.class_shuffle_method,
            random_state=self.random_state,
        )
        feat_patterns = feat_shuffler.shuffle(self.n_estimators)
        class_patterns = class_shuffler.shuffle(self.n_estimators)

        rng = random.Random(self.random_state)
        n_feat = len(feat_patterns)
        n_cls  = len(class_patterns)
        feat_cycled = [feat_patterns[i % n_feat] for i in range(self.n_estimators)]
        rng.shuffle(feat_cycled)
        cls_cycled  = [class_patterns[i % n_cls]  for i in range(self.n_estimators)]
        combos = list(zip(feat_cycled, cls_cycled))
        rng.shuffle(combos)
        norm_combos = [
            (combo, norm_methods[i % len(norm_methods)])
            for i, combo in enumerate(combos)
        ]

        used_methods = list(dict.fromkeys(nc[1] for nc in norm_combos))
        self._ensemble_configs: OrderedDict[str, list] = OrderedDict()
        self.class_shuffles_: OrderedDict[str, list] = OrderedDict()
        # Map each (method, local_index) → global estimator index for row lookup
        self._global_indices: OrderedDict[str, list[int]] = OrderedDict()

        global_order = [nc[1] for nc in norm_combos]  # norm_method per estimator slot
        for method in used_methods:
            configs = [nc[0] for nc in norm_combos if nc[1] == method]
            self._ensemble_configs[method] = configs
            self.class_shuffles_[method] = [cfg[1] for cfg in configs]
            self._global_indices[method] = [
                i for i, m in enumerate(global_order) if m == method
            ]

        self._preprocessors: dict[str, PreprocessingPipeline] = {}
        for method in used_methods:
            pp = PreprocessingPipeline(
                normalization_method=method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            pp.fit(X)
            self._preprocessors[method] = pp

        return self

    def transform(self, X_test: np.ndarray) -> OrderedDict:
        X_test = self._unique_filter.transform(X_test)
        y = self._y_train

        data: OrderedDict = OrderedDict()
        for method, configs in self._ensemble_configs.items():
            pp = self._preprocessors[method]
            X_test_pp = pp.transform(X_test)
            global_ids = self._global_indices[method]

            X_ensemble, y_ensemble = [], []
            for local_i, (feat_shuffle, class_shuffle) in enumerate(configs):
                g = global_ids[local_i]

                if self._row_indices is not None:
                    row_idx = self._row_indices[g]
                    X_train_pp = pp.X_transformed_[row_idx]
                    y_sub = y[row_idx]
                else:
                    X_train_pp = pp.X_transformed_
                    y_sub = y

                X_full = np.concatenate([X_train_pp, X_test_pp], axis=0)
                n_orig = self._n_features
                n_pp = X_full.shape[1]
                if self._col_indices is not None:
                    base_sel = self._col_indices[g]
                else:
                    base_sel = feat_shuffle
                if n_pp > n_orig:
                    extra = list(range(n_orig, n_pp))
                    col_sel = list(base_sel) + extra
                else:
                    col_sel = base_sel
                X_selected = X_full[:, col_sel]
                if self.fingerprint_feature:
                    fp = _compute_fingerprint(X_train_pp, X_test_pp).reshape(-1, 1)
                    X_selected = np.concatenate([X_selected, fp], axis=1)
                X_ensemble.append(X_selected)
                if self.class_shuffle_method == "none":
                    y_ensemble.append(y_sub.astype(np.float32))
                else:
                    y_perm = np.asarray(class_shuffle, dtype=np.float32)[y_sub.astype(np.intp)]
                    y_ensemble.append(y_perm)

            data[method] = (
                np.stack(X_ensemble, axis=0),
                np.stack(y_ensemble, axis=0),
            )

        return data

    def transform_train_only(self) -> tuple:
        all_X_train: list = []
        all_y: list = []
        y = self._y_train

        for method, configs in self._ensemble_configs.items():
            pp = self._preprocessors[method]
            global_ids = self._global_indices[method]

            for local_i, (feat_shuffle, class_shuffle) in enumerate(configs):
                g = global_ids[local_i]

                if self._row_indices is not None:
                    row_idx = self._row_indices[g]
                    X_train_pp = pp.X_transformed_[row_idx]
                    y_sub = y[row_idx]
                else:
                    X_train_pp = pp.X_transformed_
                    y_sub = y

                n_orig = self._n_features
                n_pp = X_train_pp.shape[1]
                base_sel = (
                    self._col_indices[g]
                    if self._col_indices is not None
                    else feat_shuffle
                )
                col_sel = (
                    list(base_sel) + list(range(n_orig, n_pp))
                    if n_pp > n_orig
                    else base_sel
                )
                X_selected = X_train_pp[:, col_sel]
                all_X_train.append(X_selected)
                if self.class_shuffle_method == "none":
                    all_y.append(y_sub.astype(np.float32))
                else:
                    y_perm = np.asarray(class_shuffle, dtype=np.float32)[
                        y_sub.astype(np.intp)
                    ]
                    all_y.append(y_perm)

        return all_X_train, all_y


# ---------------------------------------------------------------------------
# Two-phase (prefill / predict_cached) helpers
# ---------------------------------------------------------------------------


def _has_prefill(model) -> bool:
    """Return True if the model supports prefill + predict_cached."""
    return (
        callable(getattr(model, "prefill", None))
        and callable(getattr(model, "predict_cached", None))
    )


def _wrapper_prefill(model, device, dtype: str, batch_X_train_np: np.ndarray, batch_y_np: np.ndarray):
    import torch

    X_t = torch.as_tensor(batch_X_train_np.astype(np.float32), device=device)
    y_t = torch.as_tensor(batch_y_np.astype(np.float32), device=device)

    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    amp_dtype  = _dtype_map.get(dtype)

    with torch.no_grad():
        if amp_dtype is not None and "cuda" in str(device):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model.prefill(X_t, y_t)
        return model.prefill(X_t, y_t)


def _wrapper_predict_cached(model, device, dtype: str, batch_X_test_np: np.ndarray, cache) -> np.ndarray:
    import torch

    X_t = torch.as_tensor(batch_X_test_np.astype(np.float32), device=device)

    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    amp_dtype  = _dtype_map.get(dtype)

    with torch.no_grad():
        if amp_dtype is not None and "cuda" in str(device):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model.predict_cached(X_t, cache)
        else:
            out = model.predict_cached(X_t, cache)

    return out.float().cpu().numpy()


def _quantize_cache(cache):
    """Int8-quantize a KV cache if the cache object supports it."""
    if hasattr(cache, "quantize") and callable(cache.quantize):
        return cache.quantize()
    return cache
