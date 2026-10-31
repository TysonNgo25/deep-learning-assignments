"""Run experiment C: compare 1–4 hidden layers, with 16 neurons per layer."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import pandas as pd
import torch

from src.data import load_config, load_split
from src.paths import RESULTS_DIR
from src.plots import make_group_figures
from src.train import train_model


SUMMARY_METRICS = ("train_mse", "val_mse", "test_mse", "test_mae", "test_rmse", "train_time_s", "best_epoch")


def _evaluate(model: torch.nn.Module, x, y) -> dict[str, float]:
    """Compute the requested regression metrics without depending on evaluate.py."""
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy().reshape(-1)
    target = torch.tensor(y, dtype=torch.float32).numpy().reshape(-1)
    error = prediction - target
    mse = float((error**2).mean())
    return {"mse": mse, "mae": float(abs(error).mean()), "rmse": float(mse**0.5)}


def _training_options(config: dict) -> dict:
    return {
        key: config["train"][key]
        for key in ("optimizer", "lr", "batch_size", "max_epochs", "patience")
    }


def run_experiment(
    depths: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Train each configured depth/seed pair and write per-run logs and summaries."""
    config = load_config()
    train_config = config["train"]
    configured_hidden = config["experiments"]["C"]["hidden"]
    if any(not hidden or any(width != 16 for width in hidden) for hidden in configured_hidden):
        raise ValueError("Experiment C requires 16 neurons in every hidden layer")
    configured_depths = [len(hidden) for hidden in configured_hidden]
    values = list(depths if depths is not None else configured_depths)
    run_seeds = list(seeds if seeds is not None else train_config["seeds"])
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value not in configured_depths for value in values):
        raise ValueError(f"depths must be selected from {configured_depths}")
    if not run_seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in run_seeds):
        raise ValueError("seeds must contain integers")

    pool_x, pool_y = load_split("train_pool")
    n_train = train_config["n_train"]
    if not 0 < n_train <= len(pool_x):
        raise ValueError(f"baseline n_train must be between 1 and {len(pool_x)}")
    train_x, train_y = pool_x[:n_train], pool_y[:n_train]
    val_x, val_y = load_split("val")

    result_dir = RESULTS_DIR / "C"
    logs_dir = result_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    options = _training_options(config)
    runs = []
    for depth in values:
        hidden = [16] * depth
        for seed in run_seeds:
            run_config = {
                "experiment": "C",
                "depth": depth,
                "n_train": n_train,
                "hidden": hidden,
                "activation": train_config["activation"],
                **options,
                "seed": seed,
            }
            training = train_model(
                train_x,
                train_y,
                val_x,
                val_y,
                hidden=hidden,
                activation=train_config["activation"],
                config=options,
                seed=seed,
            )

            train_metrics = _evaluate(training.model, train_x, train_y)
            val_metrics = _evaluate(training.model, val_x, val_y)
            # Test is loaded and evaluated only after training and model selection are complete.
            test_x, test_y = load_split("test")
            test_metrics = _evaluate(training.model, test_x, test_y)
            row = {
                "group": "C",
                "value": depth,
                "seed": seed,
                "hidden": ",".join(map(str, hidden)),
                "n_train": n_train,
                "best_epoch": training.best_epoch,
                "train_time_s": training.elapsed_seconds,
                **{f"train_{name}": score for name, score in train_metrics.items()},
                **{f"val_{name}": score for name, score in val_metrics.items()},
                **{f"test_{name}": score for name, score in test_metrics.items()},
            }
            runs.append(row)

            log = pd.DataFrame(training.history)
            log["epoch"] = log["epoch"].astype(int)
            log["seed"] = seed
            log["config_json"] = json.dumps(run_config, sort_keys=True)
            log.to_csv(logs_dir / f"C_{depth}_seed{seed}.csv", index=False)
            print(
                f"C depth={depth}, seed={seed}: "
                f"test_mse={test_metrics['mse']:.6e}, "
                f"time={training.elapsed_seconds:.2f}s, epoch={training.best_epoch}"
            )

    summary = pd.DataFrame(runs)
    summary.to_csv(result_dir / "summary.csv", index=False)
    # Keep the per-run summary consumed by plots.py; also export the requested mean ± std table.
    grouped = summary.groupby("value")[list(SUMMARY_METRICS)].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped.reset_index().to_csv(result_dir / "summary_mean_std.csv", index=False)
    make_group_figures("C", "C", "Số tầng ẩn (C)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=int, nargs="+", help="Optional subset of hidden-layer counts")
    parser.add_argument("--seeds", type=int, nargs="+", help="Optional subset of configured seeds")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_experiment(args.values, args.seeds)


if __name__ == "__main__":
    main()
