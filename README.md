# LoopICL

A looped in-context learning (ICL) foundation model for tabular classification. Provides a scikit-learn compatible interface that works out of the box with pre-bundled weights.

## Installation

```bash
pip install .
```

Or in editable mode for development:

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loopicl.interface import LoopICLClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LoopICLClassifier()
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)  # (n_test, n_classes)
preds = clf.predict(X_test)        # class labels
```

## Checkpoints

Two checkpoints are bundled in the package:

| Checkpoint | Description |
|---|---|
| `loopicl-cls-stage2-step-30000.ckpt` | Standard classifier (default) |
| `loopicl-cls-ee-step-4000.ckpt` | Early-exit classifier |

## Features

### Number of Loops

`num_loops` controls how many outer ICL iterations the model runs. More loops improve accuracy at the cost of compute. It can be set at construction time or overridden per call:

```python
clf = LoopICLClassifier(num_loops=4)

# or override at predict time
proba = clf.predict_proba(X_test, num_loops=6)
```

### Early Exit

The early-exit model learns a per-loop gate that stops inference once predictions converge. `exit_q` is the CDF quantile threshold — lower values exit earlier.

```python
clf = LoopICLClassifier(use_early_exit=True, exit_q=0.5)  # balanced (default)
clf = LoopICLClassifier(use_early_exit=True, exit_q=0.3)  # aggressive, faster
clf = LoopICLClassifier(use_early_exit=True, exit_q=1.0)  # no early exit
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `use_early_exit` | `False` | Load the early-exit checkpoint |
| `exit_q` | `None` | Early-exit quantile threshold (0–1) |
| `num_loops` | `None` | Override number of ICL iterations |
| `n_estimators` | `8` | Ensemble size |
| `device` | auto | Torch device (`"cuda"`, `"cpu"`) |
| `dtype` | `"float32"` | Compute dtype (`"float16"`, `"bfloat16"`) |
| `batch_size` | `8` | Ensemble members per forward call |
| `test_chunk_size` | `None` | Max test rows per forward call |
| `cache_context` | `False` | Cache training KV at fit time |
| `subsample_samples` | `None` | Training rows per estimator |
| `max_features` | `None` | Features per estimator |
| `random_state` | `42` | Reproducibility seed |

## Notebook

See [`notebooks/classification_example.ipynb`](notebooks/classification_example.ipynb) for a worked example covering basic usage, loop sweeps, and early-exit inference.
