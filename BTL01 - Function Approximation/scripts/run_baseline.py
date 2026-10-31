"""Chạy thử baseline end-to-end: đọc CSV -> train -> lưu checkpoint -> nạp lại -> đánh giá -> vẽ hình.

Phụ trách: Trúc (mốc 1, 31/10). Mục đích là kiểm tra cả pipeline nối với nhau chạy được trên repo,
không phải để chọn mô hình. Chỉ dùng train và val; tập test để dành cho lần đánh giá cuối (3/11).

Chạy:  python -m scripts.run_baseline            (seed 0)
       python -m scripts.run_baseline --seed 1
Ra:    results/baseline/logs/baseline_seed<k>.csv     loss theo epoch
       results/baseline/baseline_seed<k>.pt           checkpoint tốt nhất theo val loss
       results/baseline/figures/baseline_seed<k>_val_y_yhat.png
       results/baseline/figures/baseline_seed<k>_surface.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # chỉ lưu file, không mở cửa sổ
import numpy as np
import torch

from src.data import load_config, load_split, load_train
from src.evaluate import evaluate_model, plot_surface_comparison, plot_y_vs_yhat, predict
from src.paths import RESULTS_DIR
from src.train import load_checkpoint, train_model

OUT_DIR = RESULTS_DIR / "baseline"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    seed = parser.parse_args().seed
    torch.set_num_threads(1)  # giống runner A/B/C, để thời gian train so được

    cfg = load_config()
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    options = {k: train_cfg[k] for k in ("optimizer", "lr", "batch_size", "max_epochs", "patience")}

    X_train, y_train = load_train(train_cfg["n_train"])
    X_val, y_val = load_split("val")

    name = f"baseline_seed{seed}"
    ckpt_path = OUT_DIR / f"{name}.pt"
    result = train_model(
        X_train, y_train, X_val, y_val,
        hidden=train_cfg["hidden"], activation=train_cfg["activation"],
        config=options, seed=seed,
        log_path=OUT_DIR / "logs" / f"{name}.csv", checkpoint_path=ckpt_path,
    )

    # Nạp lại từ file .pt và so với mô hình trong bộ nhớ: phải ra đúng cùng dự đoán
    reloaded, ckpt = load_checkpoint(ckpt_path)
    max_diff = float(np.abs(predict(result.model, X_val) - predict(reloaded, X_val)).max())

    train_m = evaluate_model(reloaded, X_train, y_train)
    val_m = evaluate_model(reloaded, X_val, y_val)
    baseline_zero = float(np.mean(y_val**2))  # MSE nếu đoán ŷ = 0 cho mọi điểm

    fig_dir = OUT_DIR / "figures"
    plot_y_vs_yhat(y_val, predict(reloaded, X_val), fig_dir / f"{name}_val_y_yhat.png")
    plot_surface_comparison(reloaded, data_cfg["mu"], data_cfg["cov"],
                            data_cfg["x1_range"], data_cfg["x2_range"],
                            save_path=fig_dir / f"{name}_surface.png")

    n_params = sum(p.numel() for p in reloaded.parameters())
    print(f"Kiến trúc 2 -> {' -> '.join(map(str, ckpt['hidden']))} -> 1, {ckpt['activation']}, "
          f"{n_params} tham số; n_train = {len(X_train)}, seed = {seed}")
    print(f"Dừng ở epoch {len(result.history)}, best epoch {result.best_epoch}, "
          f"early stopping: {result.stopped_early}, thời gian train {result.elapsed_seconds:.1f} s")
    for split, m in (("train", train_m), ("val", val_m)):
        print(f"  {split:<5} MSE {m['mse']:.4e}  MAE {m['mae']:.4e}  RMSE {m['rmse']:.4e}")
    print(f"  val   MSE khi đoán ŷ = 0: {baseline_zero:.4e}")
    print(f"Checkpoint nạp lại lệch tối đa {max_diff:.1e} so với mô hình trong bộ nhớ")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
