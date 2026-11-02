"""Audit committed C logs, export report tables and redraw figures without training.

Run from the project directory: python -m experiments.export_C
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.common import summarize_seed_runs
from src.data import load_config
from src.paths import RESULTS_DIR
from src.plots import make_group_figures


METRICS = [f"{split}_{metric}" for split in ("train", "val", "test")
           for metric in ("mse", "mae", "rmse")] + ["train_time_s", "best_epoch"]


def audit_runs(runs: pd.DataFrame, config: dict) -> list[dict]:
    """Check coverage, baseline metadata, epoch history and validation checkpoint."""
    train = config["train"]
    depths = [len(hidden) for hidden in config["experiments"]["C"]["hidden"]]
    expected = {(depth, seed) for depth in depths for seed in train["seeds"]}
    if len(runs) != len(expected) or set(zip(runs["value"], runs["seed"])) != expected:
        raise ValueError("C summary must contain exactly the configured depth/seed pairs")
    if not runs["group"].eq("C").all():
        raise ValueError("C summary contains another experiment group")
    records = []
    for row in runs.itertuples(index=False):
        name = f"C_{row.value}_seed{row.seed}.csv"
        log = pd.read_csv(RESULTS_DIR / "C" / "logs" / name)
        expected_config = {
            "experiment": "C", "depth": row.value, "seed": row.seed,
            "hidden": [16] * row.value, "n_train": train["n_train"],
            "activation": train["activation"],
            **{key: train[key] for key in ("optimizer", "lr", "batch_size", "max_epochs", "patience")},
        }
        if log.empty or not log["epoch"].eq(np.arange(1, len(log) + 1)).all():
            raise ValueError(f"{name}: missing or nonconsecutive epochs")
        if not log["seed"].eq(row.seed).all():
            raise ValueError(f"{name}: seed mismatch")
        if not all(json.loads(value) == expected_config for value in log["config_json"].unique()):
            raise ValueError(f"{name}: baseline metadata mismatch")
        if row.n_train != train["n_train"] or str(row.hidden) != ",".join(["16"] * row.value):
            raise ValueError(f"{name}: summary architecture/sample count mismatch")
        if len(log) > train["max_epochs"] or not np.isfinite(log[["train_loss", "val_loss"]]).all().all():
            raise ValueError(f"{name}: invalid loss history")
        best = log.loc[log["val_loss"].idxmin()]
        if row.best_epoch != best["epoch"] or not np.isclose(row.val_mse, best["val_loss"], rtol=2e-6, atol=1e-12):
            raise ValueError(f"{name}: best validation epoch/loss mismatch")
        if row.train_time_s <= 0:
            raise ValueError(f"{name}: invalid training duration")
        records.append({"log": name, "epochs": len(log), "best_epoch": row.best_epoch})
    return records


def export_tables() -> None:
    directory = RESULTS_DIR / "C"
    runs = pd.read_csv(directory / "summary.csv")
    records = audit_runs(runs, load_config())
    table = summarize_seed_runs(runs, group_by="value", metrics=METRICS)
    previous = pd.read_csv(directory / "summary_mean_std.csv").sort_values("value")
    if not np.array_equal(previous["value"], table["value"]):
        raise ValueError("Existing aggregate depth values do not match")
    if not np.allclose(previous.drop(columns="value"), table[previous.columns[1:]], rtol=1e-10, atol=1e-15):
        raise ValueError("Existing mean/std table does not match per-run summary")

    text = ["# Bảng kết quả C — bàn giao ngày 02/11/2026", "",
            "Nguồn: `summary.csv`, 12 log `logs/C_<tầng>_seed<seed>.csv`, baseline `configs/baseline.yaml`.",
            "Mỗi giá trị là trung bình ± độ lệch chuẩn mẫu (ddof=1) của 3 seed 0, 1, 2.",
            "Cố định 1000 mẫu train; 16 nơ-ron/tầng; tanh; Adam; lr=1e-3; batch=32; tối đa 1000 epoch; patience=100; y gốc.", ""]
    for split in ("train", "val", "test"):
        text += [f"## {split}", "", "| Số tầng | MSE | MAE | RMSE |", "| ---: | ---: | ---: | ---: |"]
        for _, row in table.iterrows():
            values = [f"{row[f'{split}_{metric}_mean']:.6e} ± {row[f'{split}_{metric}_std']:.6e}"
                      for metric in ("mse", "mae", "rmse")]
            text.append(f"| {int(row['value'])} | " + " | ".join(values) + " |")
        text.append("")
    text += ["## Thời gian và epoch tốt nhất", "", "| Số tầng | Thời gian train (giây) | Best epoch |",
             "| ---: | ---: | ---: |"]
    for _, row in table.iterrows():
        text.append(f"| {int(row['value'])} | {row['train_time_s_mean']:.2f} ± {row['train_time_s_std']:.2f} | {row['best_epoch_mean']:.2f} ± {row['best_epoch_std']:.2f} |")
    text += ["", "## Đối chiếu log", "", "| Log | Epoch thực chạy | Best epoch theo validation |", "| --- | ---: | ---: |"]
    for record in records:
        text.append(f"| [{record['log']}](logs/{record['log']}) | {record['epochs']} | {record['best_epoch']} |")
    text += ["", "## Hình và giới hạn diễn giải", "",
             "- [Loss train/validation](figures/C_loss_curves.png): mean theo các seed còn chạy ở từng epoch; early stopping làm số seed ở đoạn cuối thay đổi.",
             "- [Test MSE](figures/C_test_mse.png), [test RMSE](figures/C_test_rmse.png), [thời gian](figures/C_train_time.png), [RMSE và thời gian](figures/C_error_and_time.png).",
             "- Best epoch và min val loss đã đối chiếu với log; config/seed/độ sâu/số mẫu khớp baseline; bảng mean/std đã tính lại từ summary.",
             "- Metric train/val/test và thời gian lấy từ summary của lượt chạy đã lưu. Log không chứa dự đoán hoặc checkpoint từng lượt C, nên chưa thể tính lại độc lập các metric từ trọng số.",
             "- Test đã được đánh giá cho 12 lượt để báo cáo C. Không dùng bảng test để chốt cấu hình. Quyết định mô hình cuối ngày 03/11 thuộc bước phối hợp với nhóm.",
             "- Thời gian là đo vòng train trên cùng máy của lượt chạy trước, không gồm xuất file/vẽ hình; không đại diện cho mọi máy.", ""]
    table.to_csv(directory / "table_C_mean_std.csv", index=False)
    (directory / "table_C_mean_std.md").write_text("\n".join(text), encoding="utf-8")
    make_group_figures("C", "C", "Số tầng ẩn (C)")
    print(f"Audited {len(records)} logs; exported all train/val/test metrics, time and best epoch.")


if __name__ == "__main__":
    export_tables()
