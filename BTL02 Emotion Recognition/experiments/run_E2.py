"""E2 – BatchNorm: kiến trúc tốt nhất của E1, thêm BatchNorm sau mỗi Conv.

Phụ trách: Phạm Trần Minh Khang

    python -m experiments.run_E2                  # tự lấy kiến trúc tốt nhất từ results/E1/summary.csv
    python -m experiments.run_E2 --arch medium_01 # hoặc chỉ định
"Tốt nhất" = val_acc cao nhất tại epoch có val_loss nhỏ nhất (quy tắc chọn cấu hình của nhóm).
So E2-1 với run E1 cùng kiến trúc: hai run chỉ khác nhau ở BatchNorm.
"""
import argparse
import csv
import sys

from experiments.common import plot_group, print_summary, run_group
from src.paths import RESULTS_DIR

E1_ARCH = {"E1-1": "small", "E1-2": "medium", "E1-3": "medium_01", "E1-4": "large"}


def best_e1_arch():
    path = RESULTS_DIR / "E1" / "summary.csv"
    if not path.exists():
        sys.exit("Chưa có results/E1/summary.csv: chạy E1 trước, hoặc truyền --arch.")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["run_id"] in E1_ARCH]
    best = max(rows, key=lambda r: float(r["val_acc_at_best"]))
    print(f"Kiến trúc tốt nhất E1: {best['run_id']} ({E1_ARCH[best['run_id']]}), "
          f"val_acc {float(best['val_acc_at_best']):.4f}")
    return E1_ARCH[best["run_id"]], best["run_id"]


def main():
    parser = argparse.ArgumentParser(description="E2 – BatchNorm")
    parser.add_argument("--arch", choices=list(E1_ARCH.values()), help="mặc định: tốt nhất của E1")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--thu", action="store_true")
    args = parser.parse_args()

    if args.arch:
        arch, ref = args.arch, next(k for k, v in E1_ARCH.items() if v == args.arch)
    else:
        arch, ref = best_e1_arch()
    runs = {"E2-1": ({"model.arch": arch, "model.batchnorm": True},
                     f"{arch} + BatchNorm sau mỗi Conv (so với {ref})")}
    run_group(runs, workers=args.workers, thu=args.thu)
    if not args.thu:
        print_summary("E2")
        plot_group("E2", ["E2-1"])


if __name__ == "__main__":
    main()
