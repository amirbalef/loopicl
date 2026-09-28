"""Standard scaler for LoopICL."""

from __future__ import annotations

import torch


def torch_nanmean(x: torch.Tensor, axis: int = 0, *, include_inf: bool = False) -> torch.Tensor:
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))
    num_valid = torch.where(nan_mask, torch.zeros_like(x), torch.ones_like(x)).sum(dim=axis)
    value_sum = torch.where(nan_mask, torch.zeros_like(x), x).sum(dim=axis)
    return value_sum / num_valid.clamp(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    nan_mask = torch.isnan(x)
    num_valid = torch.where(nan_mask, torch.zeros_like(x), torch.ones_like(x)).sum(dim=axis)
    value_sum = torch.where(nan_mask, torch.zeros_like(x), x).sum(dim=axis)
    mean = value_sum / num_valid.clamp(min=1.0)
    mean_broadcast = mean.unsqueeze(axis).expand_as(x)
    sq_diff = torch.where(
        nan_mask,
        torch.zeros_like(x),
        torch.square(x - mean_broadcast),
    ).sum(dim=axis)
    variance = sq_diff / (num_valid - 1).clamp(min=1.0)
    return torch.sqrt(variance)


class TorchStandardScaler:
    """Standard scaler for PyTorch tensors with NaN handling."""

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = torch_nanmean(x, axis=0)
        std = torch_nanstd(x, axis=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        if x.shape[0] == 1:
            std = torch.ones_like(std)
        return {"mean": mean, "std": std}

    def transform(self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]) -> torch.Tensor:
        mean = fitted_cache["mean"]
        std = fitted_cache["std"]
        x = (x - mean) / (std + torch.finfo(std.dtype).eps)
        return torch.clip(x, min=-100, max=100)

    def __call__(self, x: torch.Tensor, num_train_rows: int | None = None) -> torch.Tensor:
        fit_data = x[:num_train_rows] if (num_train_rows is not None and num_train_rows > 0) else x
        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)
