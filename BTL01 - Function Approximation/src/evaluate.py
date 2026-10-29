"""Metrics and plots for evaluating a trained function-approximation model."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import gaussian_2d


def _as_numpy(values: Any) -> np.ndarray:
    """Move tensors off-device before converting them to NumPy."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MSE, MAE, and RMSE after checking the arrays are valid.

    Flattening accepts either ``(n,)`` targets or model outputs shaped
    ``(n, 1)`` without introducing unintended NumPy broadcasting.
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


def evaluate_model(model: torch.nn.Module, X: Any, y: Any) -> dict[str, float]:
    """Evaluate a model in inference mode and return regression metrics.

    The caller chooses the split to evaluate. Use validation during
    development and reserve the test split for the final, post-selection run.
    """
    features = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must have shape (n, 2)")
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    if targets.ndim != 2 or targets.shape[1] != 1 or len(features) != len(targets):
        raise ValueError("targets must have shape (n,) or (n, 1) and match the feature count")
    if len(features) == 0:
        raise ValueError("cannot evaluate an empty dataset")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        predictions = model(features.to(device)).detach().cpu().numpy()
    return regression_metrics(targets.detach().cpu().numpy(), predictions)


def predict(model: torch.nn.Module, X: Any) -> np.ndarray:
    """Predict values and return a NumPy array with shape ``(n, 1)``."""
    features = torch.tensor(X, dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must have shape (n, 2)")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        return model(features.to(device)).detach().cpu().numpy()


def plot_y_vs_yhat(
    y_true: Any,
    y_pred: Any,
    save_path: str | Path,
) -> dict[str, float]:
    """Save a measured-versus-predicted scatter plot and return its metrics.

    Points close to the dashed ``y = ŷ`` line indicate accurate predictions.
    """
    truth = _as_numpy(y_true).astype(np.float64, copy=False).reshape(-1)
    prediction = _as_numpy(y_pred).astype(np.float64, copy=False).reshape(-1)
    scores = regression_metrics(truth, prediction)

    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(truth, prediction, alpha=0.75, edgecolors="none")
    axis.plot([lower, upper], [lower, upper], "r--", label="y = ŷ")
    axis.set(
        xlabel="Giá trị thật y",
        ylabel="Giá trị dự đoán ŷ",
        title="Giá trị thật và dự đoán",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return scores


def plot_surface_comparison(
    model: torch.nn.Module,
    mu: Any,
    cov: Any,
    x1_range: tuple[float, float] = (-2.0, 2.0),
    x2_range: tuple[float, float] = (-4.0, 4.0),
    *,
    resolution: int = 80,
    save_path: str | Path,
) -> None:
    """Save side-by-side 3D surfaces for the true function and the MLP.

    The regular grid makes the overall shape comparison easier than a
    scatter of the randomly sampled training points.
    """
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if len(x1_range) != 2 or x1_range[0] >= x1_range[1]:
        raise ValueError("x1_range must contain increasing lower and upper bounds")
    if len(x2_range) != 2 or x2_range[0] >= x2_range[1]:
        raise ValueError("x2_range must contain increasing lower and upper bounds")

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
    fig.tight_layout()
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
