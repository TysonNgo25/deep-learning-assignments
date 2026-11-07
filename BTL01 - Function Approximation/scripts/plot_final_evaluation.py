"""Create validation-only plots for the selected final checkpoint."""

import json

import numpy as np
import torch

from src.data import gaussian_2d, load_config, load_split
from src.evaluate import (
    plot_surface_comparison,
    plot_y_vs_yhat,
    predict,
    regression_metrics,
)
from src.paths import RESULTS_DIR
from src.train import load_checkpoint

CHECKPOINT_PATH = RESULTS_DIR / "final" / "selected_model.pt"
FIGURES_DIR = RESULTS_DIR / "final" / "figures"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "final" / "validation_error_analysis.json"


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint does not exist: {CHECKPOINT_PATH}"
        )

    torch.set_num_threads(1)
    config = load_config()
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH)
    val_x, val_y = load_split("val")
    val_predictions = predict(model, val_x)
    truth = val_y.reshape(-1)
    predictions = val_predictions.reshape(-1)
    errors = predictions - truth
    squared_errors = np.square(errors)

    region_masks = {
        "y_below_0_01": truth < 0.01,
        "y_at_least_0_1": truth >= 0.1,
    }
    region_analysis = {}
    total_squared_error = float(squared_errors.sum())
    for name, mask in region_masks.items():
        region_analysis[name] = {
            "count": int(mask.sum()),
            "mae": float(np.abs(errors[mask]).mean()),
            "mse": float(squared_errors[mask].mean()),
            "sum_squared_error_share": (
                float(squared_errors[mask].sum() / total_squared_error)
                if total_squared_error > 0
                else 0.0
            ),
        }

    max_error_index = int(np.argmax(np.abs(errors)))
    error_analysis = {
        "split": "validation",
        "n_samples": len(truth),
        "thresholds": {
            "low_density": "y < 0.01",
            "high_density": "y >= 0.1",
        },
        "regions": region_analysis,
        "max_absolute_error": {
            "index": max_error_index,
            "x1": float(val_x[max_error_index, 0]),
            "x2": float(val_x[max_error_index, 1]),
            "y": float(truth[max_error_index]),
            "y_hat": float(predictions[max_error_index]),
            "absolute_error": float(abs(errors[max_error_index])),
        },
        "protocol": (
            "Computed on validation only from the selected checkpoint; "
            "no test samples are read."
        ),
    }

    scatter_path = FIGURES_DIR / "selected_model_validation_y_yhat.png"
    surface_path = FIGURES_DIR / "selected_model_surface_comparison.png"
    validation_metrics = plot_y_vs_yhat(val_y, val_predictions, scatter_path)
    plot_surface_comparison(
        model,
        config["data"]["mu"],
        config["data"]["cov"],
        config["data"]["x1_range"],
        config["data"]["x2_range"],
        save_path=surface_path,
    )

    grid_x1 = torch.linspace(
        config["data"]["x1_range"][0],
        config["data"]["x1_range"][1],
        80,
    )
    grid_x2 = torch.linspace(
        config["data"]["x2_range"][0],
        config["data"]["x2_range"][1],
        80,
    )
    mesh_x1, mesh_x2 = torch.meshgrid(grid_x1, grid_x2, indexing="xy")
    grid_points = torch.column_stack((mesh_x1.reshape(-1), mesh_x2.reshape(-1)))
    exact_grid = gaussian_2d(
        grid_points.numpy(),
        config["data"]["mu"],
        config["data"]["cov"],
    )
    grid_metrics = regression_metrics(exact_grid, predict(model, grid_points.numpy()))
    ERROR_ANALYSIS_PATH.write_text(
        json.dumps(error_analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}; seed={checkpoint['seed']}, "
        f"best_epoch={checkpoint['best_epoch']}"
    )
    print(f"Validation metrics: {validation_metrics}")
    print(f"Validation error regions: {region_analysis}")
    print(f"Maximum absolute validation error: {error_analysis['max_absolute_error']}")
    print(f"80x80 analytic-grid diagnostic (not a held-out split): {grid_metrics}")
    print(f"Saved validation scatter: {scatter_path}")
    print(f"Saved surface comparison: {surface_path}")
    print(f"Saved validation error analysis: {ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
