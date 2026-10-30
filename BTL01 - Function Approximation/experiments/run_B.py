"""Run experiment B: vary the width of a single hidden layer."""

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for split in ("train", "val"):
        axes[0].errorbar(
            summary["n_neurons"],
            summary[f"{split}_mse_mean"],
            yerr=summary[f"{split}_mse_std"],
            marker="o",
            capsize=3,
            label=split.capitalize(),
        )
    axes[0].set(xlabel="Số nơ-ron tầng ẩn", ylabel="MSE", title="MSE theo số nơ-ron")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for axis, metric in zip(axes[1:], METRICS[1:]):
        axis.errorbar(
            summary["n_neurons"],
            summary[f"val_{metric}_mean"],
            yerr=summary[f"val_{metric}_std"],
            marker="o",
            capsize=3,
        )
        axis.set(xlabel="Số nơ-ron tầng ẩn", ylabel=metric.upper(), title=f"Validation {metric.upper()}")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    fig.suptitle("Thực nghiệm B: một tầng ẩn")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(
    widths: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Train each configured width/seed pair and write logs and summary."""
    config = load_config()
    train_config = config["train"]
    configured_hidden = config["experiments"]["B"]["hidden"]
    if any(len(hidden) != 1 for hidden in configured_hidden):
        raise ValueError("Experiment B configurations must contain exactly one hidden layer")
    configured_widths = [int(hidden[0]) for hidden in configured_hidden]
    values = list(widths if widths is not None else configured_widths)
    run_seeds = list(seeds if seeds is not None else train_config["seeds"])
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("widths must contain positive integers")
    if not run_seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in run_seeds):
        raise ValueError("seeds must contain integers")

    pool_x, pool_y = load_split("train_pool")
    n_train = train_config["n_train"]
    if not 0 < n_train <= len(pool_x):
        raise ValueError(f"baseline n_train must be between 1 and {len(pool_x)}")
    val_x, val_y = load_split("val")
    train_x, train_y = pool_x[:n_train], pool_y[:n_train]

    result_dir = RESULTS_DIR / "B"
    logs_dir = result_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    options = _training_options(config)
    def run_one(n_neurons: int, seed: int) -> dict:
        run_config = {
            "experiment": "B",
            "n_train": n_train,
            "hidden": [n_neurons],
            "activation": train_config["activation"],
            **options,
            "seed": seed,
        }
        training = train_model(
            train_x,
            train_y,
            val_x,
            val_y,
            hidden=[n_neurons],
            activation=train_config["activation"],
            config=options,
            seed=seed,
        )
        train_metrics = evaluate_model(training.model, train_x, train_y)
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
        log["n_neurons"] = n_neurons
        log["seed"] = seed
        log["config_json"] = json.dumps(run_config, sort_keys=True)
        log["epoch"] = log["epoch"].astype(int)
        log.to_csv(logs_dir / f"B_{n_neurons}_seed{seed}.csv", index=False)
        print(
            f"B neurons={n_neurons:2d}, seed={seed}: "
            f"val_mse={val_metrics['mse']:.6e}, epoch={training.best_epoch}"
        )
        return run_row

    runs = run_seeded_experiments(
        values,
        run_seeds,
        value_column="n_neurons",
        run_one=run_one,
    )
    metric_columns = [
        *(f"train_{name}" for name in METRICS),
        *(f"val_{name}" for name in METRICS),
        "best_epoch",
        "elapsed_seconds",
    ]
    summary = summarize_seed_runs(runs, group_by="n_neurons", metrics=metric_columns)
    summary["n_train"] = n_train
    summary["hidden"] = summary["n_neurons"].astype(str)
    summary["activation"] = train_config["activation"]
    summary["optimizer"] = train_config["optimizer"]
    summary["lr"] = train_config["lr"]
    summary["batch_size"] = train_config["batch_size"]
    summary["max_epochs"] = train_config["max_epochs"]
    summary["patience"] = train_config["patience"]
    summary["seeds"] = ",".join(map(str, run_seeds))
    summary.to_csv(result_dir / "summary.csv", index=False)
    _plot_summary(summary, result_dir / "figures" / "B_metrics_vs_neurons.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=int, nargs="+", help="Optional subset of configured neuron counts")
    parser.add_argument("--seeds", type=int, nargs="+", help="Optional subset of configured seeds")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_experiment(args.values, args.seeds)


if __name__ == "__main__":
    main()
