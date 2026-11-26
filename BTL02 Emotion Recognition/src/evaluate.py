"""Đánh giá mô hình: accuracy, precision/recall/F1 từng lớp, confusion matrix, ca đúng/ca sai, đường học.

Phụ trách: Huỳnh Ngọc Trúc (confusion matrix, ca đúng/ca sai: Tạ Quang Trung)

Hai cách dùng:
    python -m src.evaluate --ckpt results/BASE/checkpoints/BASE.pt
        -> chỉ đánh giá trên validation (PublicTest), dùng cho checkpoint bất kỳ, chạy lại thoải mái.
    evaluate_final(results/final, ["F-0", "F-1", "F-2"])   (gọi từ experiments/run_final.py)
        -> chọn mô hình cuối theo val_loss rồi đánh giá PrivateTest MỘT lần, ghi final_metrics.json.
Dòng lệnh không có cách nào đọc PrivateTest.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data import NUM_CLASSES, FERDataset, load_fer2013
from src.models import build_model
from src.paths import EMOTIONS, ROOT

SPLIT_NAMES = {"val": "Validation (PublicTest)", "test": "Test (PrivateTest)"}


# ---------------------------------------------------------------- dự đoán
def load_checkpoint(path, device):
    """Dựng lại mô hình từ checkpoint của train.py (config + trọng số + mean/std của train)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, images, labels, mean, std, device, batch_size=256):
    """Trả về (nhãn đoán, xác suất N×7). Không augmentation, chuẩn hóa bằng mean/std của train."""
    loader = DataLoader(FERDataset(images, labels, mean, std), batch_size=batch_size, shuffle=False)
    model.eval()
    probs = torch.cat([torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader]).numpy()
    return probs.argmax(axis=1), probs


def split_metrics(y_true, y_pred):
    """Accuracy, macro/weighted F1 và precision/recall/F1/support từng lớp.

    Macro F1 lấy trung bình đều 7 lớp nên Disgust (1,5% dữ liệu) nặng ngang Happy;
    accuracy thì gần như bỏ qua lớp nhỏ. Báo cả hai để thấy mô hình có bỏ lớp hiếm không.
    """
    rep = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                target_names=EMOTIONS, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(np.mean(y_true == y_pred)), 5),
        "macro_f1": round(rep["macro avg"]["f1-score"], 5),
        "weighted_f1": round(rep["weighted avg"]["f1-score"], 5),
        "per_class": {name: {"precision": round(rep[name]["precision"], 5),
                             "recall": round(rep[name]["recall"], 5),
                             "f1": round(rep[name]["f1-score"], 5),
                             "support": int(rep[name]["support"])} for name in EMOTIONS},
    }


# ---------------------------------------------------------------- ghi file
def write_report_csv(path, metrics_by_split):
    """classification_report.csv: mỗi dòng một (tập, lớp), thêm dòng macro avg và accuracy."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "precision", "recall", "f1", "support"])
        for split, m in metrics_by_split.items():
            for name, c in m["per_class"].items():
                w.writerow([split, name, c["precision"], c["recall"], c["f1"], c["support"]])
            total = sum(c["support"] for c in m["per_class"].values())
            w.writerow([split, "macro avg", "", "", m["macro_f1"], total])
            w.writerow([split, "accuracy", "", "", m["accuracy"], total])


def write_predictions(path, y_true, y_pred, probs):
    """Dự đoán từng ảnh (theo thứ tự trong tập) để phân tích lỗi và đối chiếu số liệu."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "true", "pred", "true_name", "pred_name", "prob_pred"]
                   + [f"p_{e}" for e in EMOTIONS])
        for i, (t, p, pr) in enumerate(zip(y_true, y_pred, probs)):
            w.writerow([i, int(t), int(p), EMOTIONS[t], EMOTIONS[p], round(float(pr[p]), 4)]
                       + [round(float(v), 4) for v in pr])


# ---------------------------------------------------------------- hình
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(results, path):
    """Confusion matrix các tập đặt cạnh nhau (như valid_vgg16.html của thầy).

    Màu theo tỷ lệ trên mỗi hàng (= recall của lớp thật) để lớp ít ảnh như Disgust vẫn đọc được;
    trong ô ghi cả số ảnh lẫn %. Nhãn theo thứ tự gốc FER2013 0–6, không theo chữ cái như code mẫu.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, len(results), figsize=(7.2 * len(results), 6.2), squeeze=False)
    for ax, (split, (y_true, y_pred)) in zip(axes[0], results.items()):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        rate = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm[i, j]}\n{100 * rate[i, j]:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="white" if rate[i, j] > 0.5 else "black")
        ax.set_xticks(range(NUM_CLASSES), EMOTIONS, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES), EMOTIONS)
        ax.set_xlabel("Lớp đoán")
        ax.set_ylabel("Lớp thật")
        acc = np.mean(y_true == y_pred)
        ax.set_title(f"{SPLIT_NAMES.get(split, split)}: accuracy {acc:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cases(images, y_true, y_pred, probs, correct, split, path, per_class=5, seed=0):
    """Mỗi hàng một lớp thật, per_class ảnh đoán đúng (chọn ngẫu nhiên theo seed)
    hoặc đoán sai (chọn những ca mô hình tự tin nhất mà vẫn sai, đáng phân tích nhất).

    Ảnh và nhãn lấy từ cùng một tập (code mẫu có hình "Validating Images" vẽ nhầm
    x_train[valid_idx] với nhãn y_valid).
    """
    plt = _plt()
    rng = np.random.default_rng(seed)
    conf = probs[np.arange(len(y_pred)), y_pred]
    fig, axes = plt.subplots(NUM_CLASSES, per_class,
                             figsize=(per_class * 1.5, NUM_CLASSES * 1.75), squeeze=False)
    for c, name in enumerate(EMOTIONS):
        idx = np.flatnonzero((y_true == c) & ((y_pred == y_true) == correct))
        if correct:
            idx = rng.permutation(idx)[:per_class]
        else:
            idx = idx[np.argsort(-conf[idx])][:per_class]
        for j, ax in enumerate(axes[c]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j < len(idx):
                k = idx[j]
                ax.imshow(images[k], cmap="gray", vmin=0, vmax=255)
                label = EMOTIONS[y_pred[k]] if correct else f"→ {EMOTIONS[y_pred[k]]}"
                ax.set_title(f"{label} {conf[k]:.2f}", fontsize=7.5,
                             color="green" if correct else "red")
        axes[c, 0].set_ylabel(name, fontsize=9)
    kind = "đoán đúng (ngẫu nhiên)" if correct else "đoán sai (tự tin nhất)"
    fig.suptitle(f"{SPLIT_NAMES.get(split, split)}: ca {kind}; hàng = lớp thật, "
                 f"tiêu đề = lớp đoán + xác suất", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(log_paths, path):
    """Loss và accuracy theo epoch (train nét đứt, val nét liền) cho từng run/seed."""
    plt = _plt()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for k, (name, log) in enumerate(log_paths.items()):
        rows = list(csv.DictReader(open(log, encoding="utf-8")))
        ep = [int(r["epoch"]) for r in rows]
        color = f"C{k}"
        for ax, key in ((ax_loss, "loss"), (ax_acc, "acc")):
            ax.plot(ep, [float(r[f"train_{key}"]) for r in rows], "--", color=color, alpha=0.7,
                    label=f"{name} train")
            ax.plot(ep, [float(r[f"val_{key}"]) for r in rows], "-", color=color, label=f"{name} val")
    ax_loss.set(title="Loss theo epoch", xlabel="epoch", ylabel="loss")
    ax_acc.set(title="Accuracy theo epoch", xlabel="epoch", ylabel="accuracy")
    from matplotlib.ticker import MaxNLocator
    for ax in (ax_loss, ax_acc):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- hai chế độ đánh giá
def _csv_path(ckpt, csv_path):
    return Path(csv_path) if csv_path else ROOT / ckpt["config"]["data"]["csv"]


def evaluate_checkpoint(ckpt_path, out_dir=None, csv_path=None):
    """Đánh giá một checkpoint trên validation; không đọc PrivateTest."""
    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir or ckpt_path.parent.parent / "eval" / ckpt_path.stem)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(ckpt_path, device)

    images, labels = load_fer2013(_csv_path(ckpt, csv_path), splits=("val",))["val"]
    y_pred, probs = predict(model, images, labels, ckpt["train_mean"], ckpt["train_std"], device)
    metrics = {"val": split_metrics(labels, y_pred)}

    write_report_csv(out_dir / "classification_report.csv", metrics)
    write_predictions(out_dir / "predictions_val.csv", labels, y_pred, probs)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"checkpoint": ckpt_path.name, "epoch": ckpt["epoch"], **metrics}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    plot_confusion({"val": (labels, y_pred)}, out_dir / "figures" / "confusion_matrix.png")
    plot_cases(images, labels, y_pred, probs, True, "val", out_dir / "figures" / "true_cases.png")
    plot_cases(images, labels, y_pred, probs, False, "val", out_dir / "figures" / "wrong_cases.png")
    _print_metrics(metrics)
    print(f"Đã ghi vào {out_dir}")
    return metrics


def evaluate_final(final_dir, run_ids, csv_path=None):
    """Mô hình cuối: tổng hợp val của các seed, chọn checkpoint, đánh giá PrivateTest MỘT lần.

    1. Đọc summary.csv của các run (F-0, F-1, F-2): val_acc trung bình ± độ lệch chuẩn mẫu.
    2. Chọn seed có best_val_loss nhỏ nhất (chỉ dựa trên validation), chép thành best_model.pt.
    3. Đánh giá mô hình đó trên validation và PrivateTest, ghi final_metrics.json NGAY sau khi có
       số test (đánh dấu test đã dùng), rồi mới vẽ hình.
    Gọi lần hai khi final_metrics.json đã có thì báo lỗi: đó là đánh giá test lần hai.
    """
    final_dir = Path(final_dir)
    metrics_path = final_dir / "final_metrics.json"
    if metrics_path.exists():
        raise RuntimeError(f"{metrics_path} đã có: PrivateTest đã được đánh giá. Không đánh giá lần hai.")

    rows = {r["run_id"]: r for r in csv.DictReader(open(final_dir / "summary.csv", encoding="utf-8"))}
    missing = [r for r in run_ids if r not in rows]
    if missing:
        raise ValueError(f"summary.csv thiếu run {missing}")
    val_acc = np.array([float(rows[r]["val_acc_at_best"]) for r in run_ids])
    val_loss = np.array([float(rows[r]["best_val_loss"]) for r in run_ids])
    ddof = 1 if len(run_ids) > 1 else 0  # độ lệch chuẩn mẫu (chia n-1)
    chosen = run_ids[int(np.argmin(val_loss))]

    shutil.copy(final_dir / "checkpoints" / f"{chosen}.pt", final_dir / "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(final_dir / "best_model.pt", device)
    data = load_fer2013(_csv_path(ckpt, csv_path), splits=("val", "test"))
    preds, metrics = {}, {}
    for split in ("val", "test"):
        images, labels = data[split]
        preds[split] = predict(model, images, labels, ckpt["train_mean"], ckpt["train_std"], device)
        metrics[split] = split_metrics(labels, preds[split][0])

    summary = {
        "runs": run_ids,
        "seeds": [int(rows[r]["seed"]) for r in run_ids],
        "val_acc_per_run": dict(zip(run_ids, val_acc.round(5).tolist())),
        "val_acc_mean": round(float(val_acc.mean()), 5),
        "val_acc_std": round(float(val_acc.std(ddof=ddof)), 5),
        "best_val_loss_mean": round(float(val_loss.mean()), 5),
        "best_val_loss_std": round(float(val_loss.std(ddof=ddof)), 5),
        "chosen_run": chosen,
        "chosen_by": "best_val_loss nhỏ nhất trên validation (PublicTest), chọn trước khi xem test",
        "chosen_epoch": ckpt["epoch"],
        "val": metrics["val"],
        "test": metrics["test"],
        "test_evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    figs = final_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    write_report_csv(final_dir / "classification_report.csv", metrics)
    for split in ("val", "test"):
        write_predictions(final_dir / f"predictions_{split}.csv", data[split][1], *preds[split])
    plot_confusion({s: (data[s][1], preds[s][0]) for s in ("val", "test")}, figs / "confusion_matrix.png")
    images, labels = data["test"]
    plot_cases(images, labels, *preds["test"], True, "test", figs / "true_cases.png")
    plot_cases(images, labels, *preds["test"], False, "test", figs / "wrong_cases.png")
    plot_curves({r: final_dir / "logs" / f"{r}.csv" for r in run_ids}, figs / "curves.png")

    print(f"val_acc {len(run_ids)} seed: {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f}; "
          f"mô hình cuối = {chosen} (epoch {ckpt['epoch']})")
    _print_metrics(metrics)
    print(f"Đã ghi vào {final_dir}")
    return summary


def duplicate_effect(final_dir, csv_path=None):
    """Accuracy của mô hình cuối trên ảnh val/test trùng pixel với một ảnh train, so với phần còn lại.

    Chỉ đọc lại dự đoán ĐÃ LƯU (predictions_<tập>.csv) do evaluate_final ghi; không chạy lại mô hình,
    nên không phải là đánh giá test lần hai. Ảnh test chỉ dùng để biết ảnh nào trùng train (băm MD5).
    """
    import hashlib

    final_dir = Path(final_dir)
    csv_path = Path(csv_path) if csv_path else ROOT / "data" / "fer2013.csv"
    data = load_fer2013(csv_path, splits=("train", "val", "test"))
    md5 = lambda img: hashlib.md5(img.tobytes()).hexdigest()
    train_keys = {md5(img) for img in data["train"][0]}
    out = {}
    for split in ("val", "test"):
        rows = list(csv.DictReader(open(final_dir / f"predictions_{split}.csv", encoding="utf-8")))
        images, labels = data[split]
        assert len(rows) == len(labels) and all(int(r["true"]) == y for r, y in zip(rows, labels))
        dup = np.array([md5(img) in train_keys for img in images])
        correct = np.array([r["true"] == r["pred"] for r in rows])
        out[split] = {
            "n_dup": int(dup.sum()), "acc_dup": round(float(correct[dup].mean()), 5),
            "n_rest": int((~dup).sum()), "acc_rest": round(float(correct[~dup].mean()), 5),
            "acc_all": round(float(correct.mean()), 5),
        }
    (final_dir / "duplicate_effect.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for split, m in out.items():
        print(f"{SPLIT_NAMES[split]}: trùng train {m['n_dup']} ảnh acc {m['acc_dup']:.4f} | "
              f"còn lại {m['n_rest']} ảnh acc {m['acc_rest']:.4f} | tất cả {m['acc_all']:.4f}")
    return out


def _print_metrics(metrics):
    for split, m in metrics.items():
        print(f"{SPLIT_NAMES.get(split, split)}: accuracy {m['accuracy']:.4f}, macro F1 {m['macro_f1']:.4f}")
        print("   " + "  ".join(f"{n} F1 {c['f1']:.2f}" for n, c in m["per_class"].items()))


def main():
    parser = argparse.ArgumentParser(description="Đánh giá một checkpoint trên validation (PublicTest)")
    parser.add_argument("--ckpt", type=Path, help="vd. results/BASE/checkpoints/BASE.pt")
    parser.add_argument("--out", type=Path, help="mặc định results/<nhóm>/eval/<mã run>/")
    parser.add_argument("--csv", type=Path, help="mặc định đường dẫn trong config của checkpoint")
    parser.add_argument("--trung-lap", type=Path, metavar="THƯ_MỤC_FINAL",
                        help="phân tích ảnh trùng train từ dự đoán đã lưu của mô hình cuối (vd. results/final)")
    args = parser.parse_args()
    if args.trung_lap:
        duplicate_effect(args.trung_lap, args.csv)
    elif args.ckpt:
        evaluate_checkpoint(args.ckpt, args.out, args.csv)
    else:
        parser.error("cần --ckpt hoặc --trung-lap")


if __name__ == "__main__":
    main()
