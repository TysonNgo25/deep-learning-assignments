"""Vẽ đường loss từ log CSV, đồ thị so sánh giữa các cấu hình, thời gian train.

Phụ trách: Nhân (hạn 30/10).

Định dạng đầu ra của experiments/run_C.py:

* Log từng lần chạy ``results/<nhóm>/logs/<nhóm>_<giá trị>_seed<k>.csv``
  có các cột: ``epoch, train_loss, val_loss``.
* ``results/<nhóm>/summary.csv`` mỗi dòng là một lần chạy, có các cột:
  ``group, value, seed, best_epoch, train_time_s, train_mse, val_mse,
  test_mse, test_mae, test_rmse``.

Dùng: python -m src.plots C
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # không cần màn hình, chạy được trên máy chấm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# --- Tên cột (sửa tại đây nếu log của Khang đặt tên khác) -------------------
COL_EPOCH = "epoch"
COL_TRAIN_LOSS = "train_loss"
COL_VAL_LOSS = "val_loss"
COL_GROUP = "group"
COL_VALUE = "value"
COL_SEED = "seed"
COL_TIME = "train_time_s"
COL_TEST_MSE = "test_mse"
COL_TEST_MAE = "test_mae"
COL_TEST_RMSE = "test_rmse"

_DPI = 200


def results_dir(group: str) -> Path:
    return ROOT / "results" / group


def load_log(path: str | Path) -> pd.DataFrame:
    """Đọc một file log CSV của một lần chạy."""
    return pd.read_csv(path)


def load_summary(group: str) -> pd.DataFrame:
    return pd.read_csv(results_dir(group) / "summary.csv")


def aggregate(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Gộp các seed: trả về bảng mean và std theo từng giá trị của biến khảo sát."""
    grouped = summary.groupby(COL_VALUE)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.sort_index().reset_index()


def _mean_curve(logs: list[pd.DataFrame], col: str) -> np.ndarray:
    """Trung bình theo epoch qua các seed. Early stopping làm độ dài khác nhau,
    nên đệm NaN rồi dùng nanmean (epoch nào chỉ còn 1 seed thì lấy seed đó)."""
    length = max(len(df) for df in logs)
    mat = np.full((len(logs), length), np.nan)
    for i, df in enumerate(logs):
        mat[i, : len(df)] = df[col].to_numpy()
    return np.nanmean(mat, axis=0)


def _finish(fig: plt.Figure, out_path: str | Path | None) -> None:
    fig.tight_layout()
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def plot_loss_curves(
    logs_by_value: dict,
    out_path: str | Path | None = None,
    title: str = "Đường loss theo epoch",
    var_label: str = "C",
) -> None:
    """Đường train loss (nét đứt) và val loss (nét liền) của từng cấu hình.

    ``logs_by_value``: {giá trị biến: [DataFrame seed 0, seed 1, ...]}.
    Mỗi đường là trung bình qua các seed. Trục y dùng thang log vì loss
    giảm qua nhiều bậc độ lớn.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(logs_by_value)))
    for color, (value, logs) in zip(colors, sorted(logs_by_value.items())):
        epochs = np.arange(1, max(len(d) for d in logs) + 1)
        ax.plot(epochs, _mean_curve(logs, COL_VAL_LOSS), color=color,
                label=f"val, {var_label}={value}")
        ax.plot(epochs, _mean_curve(logs, COL_TRAIN_LOSS), color=color,
                linestyle="--", alpha=0.7, label=f"train, {var_label}={value}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, thang log)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    _finish(fig, out_path)


def _errorbar(
    agg: pd.DataFrame,
    metric: str,
    ylabel: str,
    xlabel: str,
    title: str,
    out_path: str | Path | None,
    log_y: bool = False,
    integer_ticks: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = agg[COL_VALUE].to_numpy()
    ax.errorbar(x, agg[f"{metric}_mean"], yerr=agg[f"{metric}_std"].fillna(0),
                marker="o", capsize=4, linewidth=1.5)
    if log_y:
        ax.set_yscale("log")
    if integer_ticks:
        ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    _finish(fig, out_path)


def plot_error_vs_param(
    summary: pd.DataFrame,
    out_path: str | Path | None = None,
    metric: str = COL_TEST_MSE,
    xlabel: str = "Số tầng ẩn (C)",
    ylabel: str | None = None,
    title: str = "Sai số test theo số tầng ẩn (mean ± std)",
    log_y: bool = True,
) -> None:
    """Sai số test theo biến khảo sát, mean ± std qua các seed."""
    agg = aggregate(summary, [metric])
    _errorbar(agg, metric, ylabel or metric.upper().replace("_", " "),
              xlabel, title, out_path, log_y=log_y)


def plot_train_time(
    summary: pd.DataFrame,
    out_path: str | Path | None = None,
    xlabel: str = "Số tầng ẩn (C)",
    title: str = "Thời gian huấn luyện theo số tầng ẩn (mean ± std)",
) -> None:
    agg = aggregate(summary, [COL_TIME])
    _errorbar(agg, COL_TIME, "Thời gian train (giây)", xlabel, title,
              out_path, log_y=False)


def plot_error_and_time(
    summary: pd.DataFrame,
    out_path: str | Path | None = None,
    metric: str = COL_TEST_RMSE,
    xlabel: str = "Số tầng ẩn (C)",
) -> None:
    """Một hình hai panel (sai số + thời gian), tiện đưa vào slide."""
    agg = aggregate(summary, [metric, COL_TIME])
    x = agg[COL_VALUE].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, label in (
        (axes[0], metric, metric.upper().replace("_", " ")),
        (axes[1], COL_TIME, "Thời gian train (giây)"),
    ):
        ax.errorbar(x, agg[f"{col}_mean"], yerr=agg[f"{col}_std"].fillna(0),
                    marker="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title("Sai số trên tập test")
    axes[1].set_title("Thời gian huấn luyện")
    _finish(fig, out_path)


def collect_logs(group: str, summary: pd.DataFrame) -> dict:
    """Đọc đúng các log được liệt kê trong summary của lần chạy hiện tại.

    Tên file: <nhóm>_<giá trị>_seed<k>.csv (ví dụ C_3_seed1.csv).
    """
    logs: dict = {}
    for value, seed in summary[[COL_VALUE, COL_SEED]].itertuples(index=False, name=None):
        path = results_dir(group) / "logs" / f"{group}_{int(value)}_seed{int(seed)}.csv"
        logs.setdefault(int(value), []).append(load_log(path))
    return logs


def make_group_figures(group: str, var_label: str, xlabel: str) -> None:
    """Sinh toàn bộ hình của một nhóm thực nghiệm vào results/<group>/figures."""
    fig_dir = results_dir(group) / "figures"
    summary = load_summary(group)
    logs = collect_logs(group, summary)

    plot_loss_curves(logs, fig_dir / f"{group}_loss_curves.png",
                     title=f"Nhóm {group}: đường loss theo epoch (trung bình các seed)",
                     var_label=var_label)
    plot_error_vs_param(summary, fig_dir / f"{group}_test_mse.png",
                        metric=COL_TEST_MSE, xlabel=xlabel,
                        title=f"Nhóm {group}: MSE trên test (mean ± std)")
    plot_error_vs_param(summary, fig_dir / f"{group}_test_rmse.png",
                        metric=COL_TEST_RMSE, xlabel=xlabel, log_y=False,
                        title=f"Nhóm {group}: RMSE trên test (mean ± std)")
    plot_train_time(summary, fig_dir / f"{group}_train_time.png", xlabel=xlabel,
                    title=f"Nhóm {group}: thời gian train (mean ± std)")
    plot_error_and_time(summary, fig_dir / f"{group}_error_and_time.png",
                        xlabel=xlabel)
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    grp = sys.argv[1] if len(sys.argv) > 1 else "C"
    if grp != "C":
        raise SystemExit("plots.py currently supports experiment C only")
    make_group_figures(grp, "C", "Số tầng ẩn (C)")
