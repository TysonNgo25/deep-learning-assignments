"""Metrics and visualizations for the BTL01 function-approximation MLP."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from src.data import gaussian_2d


def _as_numpy(values: Any) -> np.ndarray:
    """Convert tensors or array-like values to NumPy safely."""
    if isinstance(values, Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _model_device(model: nn.Module) -> torch.device:
    """Find a module's parameter or buffer device, defaulting to CPU."""
    for value in (*model.parameters(), *model.buffers()):
        return value.device
    return torch.device("cpu")


def _validated_features(X: Any) -> Tensor:
    if isinstance(X, Tensor):
        features = X.detach().to(device="cpu", dtype=torch.float32)
    else:
        features = torch.tensor(X, dtype=torch.float32, device="cpu")
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must have shape (n, 2)")
    if len(features) == 0:
        raise ValueError("cannot evaluate an empty dataset")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    return features


def _validated_targets(y: Any, expected_count: int) -> Tensor:
    if isinstance(y, Tensor):
        targets = y.detach().to(device="cpu", dtype=torch.float32)
    else:
        targets = torch.tensor(y, dtype=torch.float32, device="cpu")
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    if targets.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("targets must have shape (n,) or (n, 1)")
    if len(targets) != expected_count:
        raise ValueError("features and targets must have the same number of rows")
    if not torch.isfinite(targets).all():
        raise ValueError("targets must contain only finite values")
    return targets


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MSE, MAE and RMSE for equally sized, finite scalar values.

    Flattening permits targets shaped ``(n,)`` and predictions shaped
    ``(n, 1)`` without accidental NumPy broadcasting.
    """
    truth = _as_numpy(y_true).astype(np.float64, copy=False).reshape(-1)
    prediction = _as_numpy(y_pred).astype(np.float64, copy=False).reshape(-1)
    if truth.size == 0 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same non-zero number of values")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("y_true and y_pred must contain only finite values")

    errors = prediction - truth
    mse = float(np.mean(np.square(errors)))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(mse)),
    }


def predict(model: nn.Module, X: Any) -> np.ndarray:
    """Predict scalar outputs and return a NumPy array shaped ``(n, 1)``."""
    features = _validated_features(X)
    device = _model_device(model)
    previous_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            predictions = model(features.to(device))
    finally:
        model.train(previous_mode)

    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    if predictions.ndim != 2 or predictions.shape != (len(features), 1):
        raise ValueError(
            "model must return one scalar per input with shape (n,) or (n, 1)"
        )
    predictions = predictions.detach().cpu().numpy()
    if not np.isfinite(predictions).all():
        raise ValueError("model predictions must contain only finite values")
    return predictions


def evaluate_model(model: nn.Module, X: Any, y: Any) -> dict[str, float]:
    """Evaluate one split and return MSE, MAE and RMSE.

    The caller chooses the split. Use validation while developing and reserve
    test evaluation for after the model/configuration has been selected.
    """
    features = _validated_features(X)
    targets = _validated_targets(y, len(features))
    predictions = predict(model, features)
    return regression_metrics(targets.numpy(), predictions)


def plot_y_vs_yhat(
    y_true: Any,
    y_pred: Any,
    save_path: str | Path,
) -> dict[str, float]:
    """Save a true-versus-predicted scatter plot and return its metrics."""
    truth = _as_numpy(y_true).astype(np.float64, copy=False).reshape(-1)
    prediction = _as_numpy(y_pred).astype(np.float64, copy=False).reshape(-1)
    scores = regression_metrics(truth, prediction)

    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    if lower == upper:
        margin = max(abs(lower) * 0.05, 1e-8)
        lower -= margin
        upper += margin

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(truth, prediction, alpha=0.75, edgecolors="none")
    axis.plot([lower, upper], [lower, upper], "r--", label="y = ŷ")
    axis.set(
        xlabel="Giá trị thật y",
        ylabel="Giá trị dự đoán ŷ",
        title="Giá trị thật và dự đoán",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return scores


def plot_surface_comparison(
    model: nn.Module,
    mu: Any,
    cov: Any,
    x1_range: tuple[float, float] = (-2.0, 2.0),
    x2_range: tuple[float, float] = (-4.0, 4.0),
    *,
    resolution: int = 80,
    save_path: str | Path,
) -> None:
    """Save side-by-side 3D surfaces for the true Gaussian and trained MLP."""
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer of at least 2")
    for name, bounds in (("x1_range", x1_range), ("x2_range", x2_range)):
        if (
            len(bounds) != 2
            or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"{name} must contain finite, increasing bounds")

    x1 = np.linspace(*x1_range, resolution)
    x2 = np.linspace(*x2_range, resolution)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    true_surface = gaussian_2d(points, mu, cov).reshape(grid_x1.shape)
    predicted_surface = predict(model, points).reshape(grid_x1.shape)

    fig = plt.figure(figsize=(13, 5))
    for index, (surface, title) in enumerate(
        ((true_surface, "Hàm Gaussian thật"), (predicted_surface, "MLP xấp xỉ")),
        start=1,
    ):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        axis.plot_surface(grid_x1, grid_x2, surface, cmap="viridis", linewidth=0)
        axis.set(xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)", title=title)
        # Five ticks per axis keep the x₁ labels from overlapping on the short side.
        axis.set_xticks(np.linspace(*x1_range, 5))
        axis.set_yticks(np.linspace(*x2_range, 5))
    fig.tight_layout()

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
