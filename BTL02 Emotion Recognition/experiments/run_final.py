"""Mô hình cuối: train cấu hình tốt nhất với 3 seed, đánh giá PrivateTest MỘT lần, vẽ hình cho báo cáo.

Phụ trách: Huỳnh Ngọc Trúc

Cấu hình lấy từ configs/final.yaml (baseline + các thay đổi tốt nhất của E1–E6, phải chot: true).
    python -m experiments.run_final --thu           # chạy thử toàn bộ, chỉ validation, ghi vào results/_thu/
    python -m experiments.run_final                 # F-0, F-1, F-2 -> đánh giá test 1 lần -> hình
    python -m experiments.run_final --seeds 0 2 --chi-train   # chỉ train vài seed (chia 2 GPU trên Kaggle)
Seed đã có log thì bỏ qua, nên lệnh đầy đủ chạy sau --chi-train sẽ chỉ còn bước đánh giá.
Khi results/final/final_metrics.json đã có, script dừng ngay: PrivateTest đã được dùng.
"""
import argparse
import sys
import time

import yaml

from src.evaluate import evaluate_checkpoint, evaluate_final
from src.paths import CONFIG_DIR, RESULTS_DIR
from src.train import load_config, train
from src.visualize import visualize

SEEDS = (0, 1, 2)
FINAL_DIR = RESULTS_DIR / "final"


def load_final(path=CONFIG_DIR / "final.yaml"):
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    override = spec.get("override") or {}
    load_config(override=override)  # khóa sai tên báo lỗi ngay, trước khi train
    return bool(spec.get("chot")), override


def describe(override):
    return "; ".join(f"{k}={v}" for k, v in override.items()) or "giống baseline"


def train_seeds(override, seeds, out_dir, workers=None, csv_path=None, **kwargs):
    """Train F-<seed> cho từng seed; seed đã có log trong out_dir thì bỏ qua."""
    for seed in seeds:
        run_id = f"F-{seed}"
        if (out_dir / "logs" / f"{run_id}.csv").exists():
            print(f"[{run_id}] đã có log, bỏ qua.")
            continue
        cfg = load_config(override={**override, "seed": seed})
        if workers is not None:
            cfg["train"]["num_workers"] = workers
        if csv_path is not None:  # chỉ dùng khi kiểm code bằng dữ liệu giả
            cfg["data"]["csv"] = str(csv_path)
        train(run_id, cfg, change=f"Mô hình cuối: {describe(override)}", out_dir=out_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Mô hình cuối: 3 seed, đánh giá PrivateTest một lần")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--chi-train", action="store_true", help="chỉ train, chưa đánh giá")
    parser.add_argument("--thu", action="store_true",
                        help="chạy thử 1 epoch × 5 batch mỗi seed, chỉ đánh giá validation")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--steps", type=int, default=150, help="số bước vẽ filter (visualize)")
    args = parser.parse_args()

    chot, override = load_final()
    print(f"Cấu hình mô hình cuối: {describe(override)} (chot: {chot})")

    if args.thu:
        out = RESULTS_DIR / "_thu" / f"final_{time.strftime('%Y%m%d-%H%M%S')}"
        train_seeds(override, args.seeds, out, args.workers, max_epochs=1, max_batches=5)
        # Không đọc PrivateTest: chỉ đánh giá validation cho seed đầu và vẽ hình.
        ckpt = out / "checkpoints" / f"F-{args.seeds[0]}.pt"
        evaluate_checkpoint(ckpt, out / "eval")
        visualize(ckpt, out / "figures", n_filters=16, steps=10)
        return

    if (FINAL_DIR / "final_metrics.json").exists():
        sys.exit("results/final/final_metrics.json đã có: PrivateTest đã được đánh giá. Không chạy lại.")
    if not chot:
        sys.exit("configs/final.yaml chưa chốt (chot: false). Điền override theo kết quả E1–E6, "
                 "đổi chot: true rồi chạy lại. Muốn kiểm code thì dùng --thu.")

    train_seeds(override, args.seeds, FINAL_DIR, args.workers)
    if args.chi_train:
        return
    missing = [s for s in SEEDS if not (FINAL_DIR / "logs" / f"F-{s}.csv").exists()]
    if missing:
        sys.exit(f"Chưa đủ 3 seed (thiếu {missing}); train xong mới đánh giá.")
    evaluate_final(FINAL_DIR, [f"F-{s}" for s in SEEDS])
    visualize(FINAL_DIR / "best_model.pt", FINAL_DIR / "figures", steps=args.steps)


if __name__ == "__main__":
    main()
