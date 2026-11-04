"""Create validation-only plots for the selected final checkpoint."""

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

    print(
        f"Checkpoint: {CHECKPOINT_PATH}; seed={checkpoint['seed']}, "
        f"best_epoch={checkpoint['best_epoch']}"
    )
    print(f"Validation metrics: {validation_metrics}")
    print(f"80x80 analytic-grid diagnostic (not a held-out split): {grid_metrics}")
    print(f"Saved validation scatter: {scatter_path}")
    print(f"Saved surface comparison: {surface_path}")


if __name__ == "__main__":
    main()
