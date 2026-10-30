"""Run experiment A: vary the number of training samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data import load_config, load_split
from src.evaluate import evaluate_model
from src.paths import RESULTS_DIR
from src.train import train_model
from experiments.common import run_seeded_experiments, summarize_seed_runs


METRICS = ("mse", "mae", "rmse")


def _training_options(config: dict) -> dict:
    return {
        key: config["train"][key]
        for key in ("optimizer", "lr", "batch_size", "max_epochs", "patience")
    }


def _plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.5))
    for axis, metric in zip(axes, METRICS):
        axis.errorbar(
            summary["n_train"],
            summary[f"val_{metric}_mean"],
            yerr=summary[f"val_{metric}_std"],
            marker="o",
            capsize=3,
        )
        axis.set(xlabel="Số mẫu train", ylabel=metric.upper(), title=f"Validation {metric.upper()}")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    fig.suptitle("Thực nghiệm A: ảnh hưởng của số mẫu train")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(
    n_train_values: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Train each configured sample-size/seed pair and write logs and summary."""
    config = load_config()
    train_config = config["train"]
    values = list(n_train_values if n_train_values is not None else config["experiments"]["A"]["n_train"])
    run_seeds = list(seeds if seeds is not None else train_config["seeds"])
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("n_train_values must contain positive integers")
    if not run_seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in run_seeds):
        raise ValueError("seeds must contain integers")

    pool_x, pool_y = load_split("train_pool")
    val_x, val_y = load_split("val")
    max_train = len(pool_x)
    if any(value > max_train for value in values):
        raise ValueError(f"A sample sizes must not exceed the train pool size ({max_train})")

    result_dir = RESULTS_DIR / "A"
    logs_dir = result_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    options = _training_options(config)
    def run_one(n_train: int, seed: int) -> dict:
        run_config = {
            "experiment": "A",
            "n_train": n_train,
            "hidden": list(train_config["hidden"]),
            "activation": train_config["activation"],
            **options,
            "seed": seed,
        }
        training = train_model(
            pool_x[:n_train],
            pool_y[:n_train],
            val_x,
            val_y,
            hidden=train_config["hidden"],
            activation=train_config["activation"],
            config=options,
            seed=seed,
        )
        train_metrics = evaluate_model(training.model, pool_x[:n_train], pool_y[:n_train])
        val_metrics = evaluate_model(training.model, val_x, val_y)
        run_row = {
            **{f"train_{name}": score for name, score in train_metrics.items()},
            **{f"val_{name}": score for name, score in val_metrics.items()},
            "best_epoch": training.best_epoch,
            "elapsed_seconds": training.elapsed_seconds,
        }

        log = pd.DataFrame(training.history)
        for name, value in run_row.items():
            log[name] = value
        log["n_train"] = n_train
        log["seed"] = seed
        log["config_json"] = json.dumps(run_config, sort_keys=True)
        log["epoch"] = log["epoch"].astype(int)
        log.to_csv(logs_dir / f"A_{n_train}_seed{seed}.csv", index=False)
        print(
            f"A n_train={n_train:4d}, seed={seed}: "
            f"val_mse={val_metrics['mse']:.6e}, epoch={training.best_epoch}"
        )
        return run_row

    runs = run_seeded_experiments(
        values,
        run_seeds,
        value_column="n_train",
        run_one=run_one,
    )
    metric_columns = [
        *(f"train_{name}" for name in METRICS),
        *(f"val_{name}" for name in METRICS),
        "best_epoch",
        "elapsed_seconds",
    ]
    summary = summarize_seed_runs(runs, group_by="n_train", metrics=metric_columns)
    summary["hidden"] = ",".join(map(str, train_config["hidden"]))
    summary["activation"] = train_config["activation"]
    summary["optimizer"] = train_config["optimizer"]
    summary["lr"] = train_config["lr"]
    summary["batch_size"] = train_config["batch_size"]
    summary["max_epochs"] = train_config["max_epochs"]
    summary["patience"] = train_config["patience"]
    summary["seeds"] = ",".join(map(str, run_seeds))
    summary.to_csv(result_dir / "summary.csv", index=False)
    _plot_summary(summary, result_dir / "figures" / "A_validation_metrics.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=int, nargs="+", help="Optional subset of configured train sizes")
    parser.add_argument("--seeds", type=int, nargs="+", help="Optional subset of configured seeds")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_experiment(args.values, args.seeds)


if __name__ == "__main__":
    main()
