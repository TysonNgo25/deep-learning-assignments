"""Tổng hợp mọi run BASE, E1–E6 thành một bảng và các hình dùng cho báo cáo mục 3.

Phụ trách: Huỳnh Ngọc Trúc

    python -m experiments.tong_hop
Chỉ đọc results/<nhóm>/summary.csv và logs/, không train gì. Ghi:
    results/tong_hop.csv                    mỗi run một dòng, thêm cột so với BASE, train − val, chạm trần
    results/figures/tong_hop_val_acc.png    val_acc của mọi run so với BASE
    results/BASE/figures/curves.png         loss, accuracy theo epoch của BASE (train và val)
"""
import csv

from src.data import EXPECTED_COUNTS
from src.evaluate import plot_curves
from src.paths import RESULTS_DIR
from src.train import load_config


def exact(acc, n):
    """Accuracy chính xác từ giá trị đã làm tròn 5 chữ số trong log.

    Log ghi 0.66425 cho 2384/3589 = 0.6642519…; lấy lại số ảnh đúng (số nguyên) rồi chia,
    để làm tròn 2 chữ số phần trăm không bị lệch ở ca "đúng nửa" (66.425 -> 66.43, không phải 66.42).
    """
    return round(float(acc) * n) / n

GROUPS = ["BASE", "E1", "E2", "E3", "E4", "E5", "E6"]
COLUMNS = ["run_id", "group", "change", "best_epoch", "epochs_ran", "hit_max_epochs", "best_val_loss",
           "val_acc_at_best", "train_acc_at_best", "train_minus_val", "delta_val_acc_vs_base",
           "train_time_s", "n_params"]


def collect():
    max_epochs = load_config()["train"]["max_epochs"]
    rows = []
    for g in GROUPS:
        path = RESULTS_DIR / g / "summary.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            rows.append({**r, "group": g})
    n_val, n_train = EXPECTED_COUNTS["val"], EXPECTED_COUNTS["train"]
    base_va = exact(next(r for r in rows if r["run_id"] == "BASE")["val_acc_at_best"], n_val)
    out = []
    for r in rows:
        va, tr = exact(r["val_acc_at_best"], n_val), exact(r["train_acc_at_best"], n_train)
        out.append({
            "run_id": r["run_id"], "group": r["group"], "change": r["change"],
            "best_epoch": int(r["best_epoch"]), "epochs_ran": int(r["epochs_ran"]),
            # epoch tốt nhất là epoch cuối và đã hết số epoch tối đa: có thể vẫn đang cải thiện
            "hit_max_epochs": int(r["epochs_ran"]) == max_epochs and int(r["best_epoch"]) == max_epochs,
            "best_val_loss": float(r["best_val_loss"]), "val_acc_at_best": round(va, 6),
            "train_acc_at_best": round(tr, 6), "train_minus_val": round(tr - va, 6),
            "delta_val_acc_vs_base": round(va - base_va, 6),
            # cộng thẳng epoch_time_s trong log (summary.csv đã làm tròn 1 chữ số)
            "train_time_s": round(sum(float(x["epoch_time_s"]) for x in csv.DictReader(
                open(RESULTS_DIR / r["group"] / "logs" / f"{r['run_id']}.csv", encoding="utf-8"))), 2),
            "n_params": int(r["n_params"]),
        })
    return out


def plot_val_acc(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = next(r for r in rows if r["run_id"] == "BASE")["val_acc_at_best"] * 100
    runs = [r for r in rows if r["run_id"] not in ("BASE", "E1-2")]  # E1-2 trùng BASE
    colors = {g: f"C{i}" for i, g in enumerate(GROUPS[1:])}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = range(len(runs))
    bars = ax.bar(x, [r["val_acc_at_best"] * 100 for r in runs],
                  color=[colors[r["group"]] for r in runs],
                  hatch=["//" if r["hit_max_epochs"] else "" for r in runs], edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.axhline(base, color="black", linestyle="--", linewidth=1, label=f"BASE {base:.2f}%")
    ax.set_xticks(list(x), [r["run_id"] for r in runs], rotation=45)
    ax.set_ylim(50, 70)
    ax.set_xlabel("Mã run")
    ax.set_ylabel("val_acc tại epoch val_loss nhỏ nhất (%)")
    ax.set_title("val_acc của các run so với BASE (cột gạch chéo: chạm trần 60 epoch, chưa hội tụ)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rows = collect()
    with open(RESULTS_DIR / "tong_hop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    plot_val_acc(rows, RESULTS_DIR / "figures" / "tong_hop_val_acc.png")
    (RESULTS_DIR / "BASE" / "figures").mkdir(parents=True, exist_ok=True)
    plot_curves({"BASE": RESULTS_DIR / "BASE" / "logs" / "BASE.csv"},
                RESULTS_DIR / "BASE" / "figures" / "curves.png")
    for r in rows:
        flag = " (chạm trần)" if r["hit_max_epochs"] else ""
        print(f"{r['run_id']:5s} {100 * r['val_acc_at_best']:6.2f}%  {100 * r['delta_val_acc_vs_base']:+6.2f}  "
              f"train−val {100 * r['train_minus_val']:+6.2f}{flag}")
    print(f"Đã ghi {RESULTS_DIR / 'tong_hop.csv'} và 2 hình")


if __name__ == "__main__":
    main()
