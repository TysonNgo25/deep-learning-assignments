"""Phần dùng chung cho experiments/run_E*.py: chọn run, chạy tiếp run còn thiếu, bảng tóm tắt, đồ thị nhóm.

Phụ trách: Huỳnh Ngọc Trúc

Mỗi run_E*.py chỉ khai báo RUNS = {mã run: (override, mô tả thay đổi)} rồi gọi main_group(...).
"""
import argparse
import csv
import time

from src.paths import RESULTS_DIR
from src.train import group_of, run


def parse_args(description, run_ids):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runs", nargs="+", choices=run_ids, help="chỉ chạy các run này (mặc định: tất cả)")
    parser.add_argument("--workers", type=int, help="ghi đè num_workers (máy thiếu RAM dùng 0)")
    parser.add_argument("--thu", action="store_true",
                        help="chạy thử mỗi run 1 epoch × 5 batch vào results/_thu/ để kiểm config")
    return parser.parse_args()


def run_group(runs, selected=None, workers=None, thu=False):
    """Chạy lần lượt các run. Run đã có log thì bỏ qua (không ghi đè), nên chạy lại lệnh
    sau khi Colab/Kaggle ngắt giữa chừng sẽ tiếp tục từ run còn thiếu."""
    summaries = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for run_id, (override, change) in runs.items():
        if selected and run_id not in selected:
            continue
        if thu:
            summaries.append(run(run_id, override, change, workers,
                                 out_dir=RESULTS_DIR / "_thu" / f"{run_id}_{stamp}",
                                 max_epochs=1, max_batches=5))
            continue
        log_path = RESULTS_DIR / group_of(run_id) / "logs" / f"{run_id}.csv"
        if log_path.exists():
            print(f"[{run_id}] đã có {log_path.name}, bỏ qua. Muốn chạy lại thì dùng mã {run_id}_r2.")
            continue
        summaries.append(run(run_id, override, change, workers))
    return summaries


def print_summary(group):
    path = RESULTS_DIR / group / "summary.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    print(f"\n{path}:")
    print(f"{'run':8s} {'best_ep':>7s} {'val_loss':>9s} {'val_acc':>8s} {'train_acc':>9s} {'time_s':>8s}  change")
    for r in rows:
        print(f"{r['run_id']:8s} {r['best_epoch']:>7s} {float(r['best_val_loss']):9.4f} "
              f"{float(r['val_acc_at_best']):8.4f} {float(r['train_acc_at_best']):9.4f} "
              f"{float(r['train_time_s']):8.0f}  {r['change']}")


def plot_group(group, run_ids, with_base=True):
    """Đồ thị val_loss và val_acc theo epoch của các run trong nhóm (kèm BASE để so) cho báo cáo mục 3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = {}
    if with_base:
        base = RESULTS_DIR / "BASE" / "logs" / "BASE.csv"
        if base.exists():
            logs["BASE"] = base
    for run_id in run_ids:
        path = RESULTS_DIR / group / "logs" / f"{run_id}.csv"
        if path.exists():
            logs[run_id] = path
    if not logs:
        return

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for name, path in logs.items():
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        epochs = [int(r["epoch"]) for r in rows]
        style = dict(color="black", linestyle="--") if name == "BASE" else {}
        ax_loss.plot(epochs, [float(r["val_loss"]) for r in rows], label=name, **style)
        ax_acc.plot(epochs, [float(r["val_acc"]) for r in rows], label=name, **style)
    ax_loss.set(title=f"{group}: val_loss theo epoch", xlabel="epoch", ylabel="val_loss")
    ax_acc.set(title=f"{group}: val_acc theo epoch", xlabel="epoch", ylabel="val_acc")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / group / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)
    print(f"Đã vẽ {out / 'curves.png'}")


def main_group(group, runs, description, with_base=True):
    args = parse_args(description, list(runs))
    run_group(runs, args.runs, args.workers, args.thu)
    if not args.thu:
        print_summary(group)
        plot_group(group, list(runs), with_base)
