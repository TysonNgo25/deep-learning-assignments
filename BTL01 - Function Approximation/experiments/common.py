"""Shared helpers for reproducible multi-seed experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
import pandas as pd


Value = TypeVar("Value")


def run_seeded_experiments(
    values: Sequence[Value],
    seeds: Sequence[int],
    *,
    value_column: str,
    run_one: Callable[[Value, int], Mapping[str, Any]],
) -> pd.DataFrame:
    """Run each value/seed combination once and collect one row per run."""
    if not value_column or value_column == "seed":
        raise ValueError("value_column must be a non-empty name other than 'seed'")
    if not values:
        raise ValueError("values must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("values must not contain duplicates")
    if not seeds:
        raise ValueError("seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must contain integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")

    rows = []
    for value in values:
        for seed in seeds:
            result = run_one(value, seed)
            if not isinstance(result, Mapping):
                raise TypeError("run_one must return a mapping of run results")
            if value_column in result or "seed" in result:
                raise ValueError("run_one results must not override the value or seed columns")
            rows.append({value_column: value, **result, "seed": seed})
    return pd.DataFrame(rows)


def summarize_seed_runs(
    runs: pd.DataFrame,
    *,
    group_by: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Aggregate run metrics by configuration using arithmetic mean and sample std."""
    if not isinstance(runs, pd.DataFrame) or runs.empty:
        raise ValueError("runs must be a non-empty DataFrame")
    if not group_by or group_by not in runs.columns:
        raise ValueError(f"group column {group_by!r} is missing")
    if "seed" not in runs.columns:
        raise ValueError("runs must include a seed column")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a non-empty sequence without duplicates")
    missing = [column for column in metrics if column not in runs.columns]
    if missing:
        raise ValueError(f"run metrics are missing: {', '.join(missing)}")
    if runs.duplicated([group_by, "seed"]).any():
        raise ValueError(f"runs contain duplicate seeds for {group_by}")

    numeric_metrics = runs.loc[:, metrics].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_metrics.to_numpy(dtype=float)).all():
        raise ValueError("run metrics must contain only finite numeric values")

    aggregated = runs.groupby(group_by, sort=True)[list(metrics)].agg(["mean", "std"])
    aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    return aggregated.reset_index()
