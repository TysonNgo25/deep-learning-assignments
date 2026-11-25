"""Vòng train một run: đọc config, train, checkpoint theo val_loss, early stopping, log CSV, summary.csv.

Phụ trách: Huỳnh Ngọc Trúc

Chạy (đứng ở thư mục BTL02):
    python -m src.train --run BASE --thu                 # chạy thử 2 epoch × 20 batch, ghi vào results/_thu/
    python -m src.train --run BASE                       # baseline đầy đủ -> results/BASE/
    python -m src.train --run E3-2 --set train.lr=0.0001 --change "Adam lr 1e-4"
Trên máy thiếu RAM thêm --workers 0.

Chỉ dùng train (Training) và validation (PublicTest). PrivateTest không được đọc ở đây.
"""
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
import yaml
from torch import nn

from src.data import get_dataloaders
from src.models import build_model, count_params
from src.paths import CONFIG_DIR, RESULTS_DIR

LOG_COLUMNS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"]
SUMMARY_COLUMNS = ["run_id", "change", "seed", "best_epoch", "best_val_loss", "val_acc_at_best",
                   "train_acc_at_best", "epochs_ran", "train_time_s", "n_params"]
# ReduceLROnPlateau giống code mẫu: giảm lr 10 lần khi val_loss 2 epoch liền không giảm.
PLATEAU = dict(factor=0.1, patience=2, min_lr=1e-7)


# ---------------------------------------------------------------- config, seed
def load_config(path=CONFIG_DIR / "baseline.yaml", override=None):
    """Đọc baseline rồi ghi đè các khóa dạng "nhóm.khóa" (vd. {"model.activation": "elu"}).

    Khóa không có trong baseline thì báo lỗi ngay, để gõ sai tên khóa không âm thầm
    thành một run giống hệt baseline.
    """
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in (override or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                raise KeyError(f"Không có nhóm '{p}' trong {path.name} (khóa '{key}')")
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Không có khóa '{key}' trong {path.name}")
        node[leaf] = value
    return cfg


def set_seed(seed):
    """Cố định mọi nguồn ngẫu nhiên: khởi tạo trọng số, xáo dữ liệu, augmentation, dropout.

    Phải gọi trước khi tạo DataLoader và mô hình.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN mặc định chọn thuật toán nhanh nhất theo phần cứng, có thể khác nhau giữa các lần chạy.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_of(run_id):
    """Thư mục kết quả của run: "E1-2" -> "E1", "E3-2_r2" -> "E3", "F-0" -> "final", "BASE" -> "BASE"."""
    head = run_id.split("_")[0]
    if head.startswith("F-"):
        return "final"
    return head.split("-")[0] if head.startswith("E") else head


# ---------------------------------------------------------------- train / đánh giá
def build_optimizer(train_cfg, params):
    name = train_cfg["optimizer"]
    if name == "adam":
        return torch.optim.Adam(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=train_cfg["lr"], momentum=train_cfg["momentum"],
                               weight_decay=train_cfg["weight_decay"])
    raise ValueError(f"optimizer '{name}' không hỗ trợ (adam | sgd)")


def build_scheduler(train_cfg, optimizer):
    name = train_cfg["scheduler"]
    if name in (None, "none"):
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **PLATEAU)
    raise ValueError(f"scheduler '{name}' không hỗ trợ (none | plateau)")


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """Một lượt qua loader. Có optimizer thì train, không có thì chỉ đánh giá.

    Trả về (loss trung bình theo ảnh, accuracy). train_acc tính trong lúc train nên vẫn có
    dropout và augmentation, giống cách Keras báo `acc`.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def train(run_id, cfg, change="", out_dir=None, max_epochs=None, max_batches=None):
    """Train một run và ghi kết quả. Trả về dòng summary (dict).

    Ghi vào out_dir (mặc định results/<nhóm>/):
      logs/<run>.csv          log theo epoch
      checkpoints/<run>.pt    mô hình có val_loss nhỏ nhất
      configs/<run>.json      config đầy đủ + thông tin máy, để biết run này chạy với gì
      summary.csv             thêm 1 dòng cho run
    """
    out_dir = out_dir or RESULTS_DIR / group_of(run_id)
    log_path = out_dir / "logs" / f"{run_id}.csv"
    if log_path.exists():
        raise FileExistsError(
            f"{log_path} đã có. Không ghi đè run đã chạy; chạy lại thì đặt mã {run_id}_r2."
        )
    for sub in ("logs", "checkpoints", "configs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, info = get_dataloaders(cfg, seed=seed)
    model = build_model(cfg["model"]).to(device)
    n_params = count_params(model)

    weight = info["class_weights"].to(device) if cfg["train"]["class_weighted_loss"] else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = build_optimizer(cfg["train"], model.parameters())
    scheduler = build_scheduler(cfg["train"], optimizer)
    max_epochs = max_epochs or cfg["train"]["max_epochs"]
    patience = cfg["train"]["early_stopping_patience"]

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    meta = {"run_id": run_id, "change": change, "device": device_name, "n_params": n_params,
            "train_mean": info["mean"], "train_std": info["std"], "config": cfg}
    (out_dir / "configs" / f"{run_id}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{run_id}] {change or 'baseline'} | {device_name} | {n_params:,} tham số | "
          f"tối đa {max_epochs} epoch, patience {patience}")

    history, best, bad_epochs = [], None, 0
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for epoch in range(1, max_epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            start = time.perf_counter()
            train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device,
                                              optimizer, max_batches)
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device,
                                          max_batches=max_batches)
            row = {"epoch": epoch, "train_loss": round(train_loss, 5),
                   "train_acc": round(train_acc, 5), "val_loss": round(val_loss, 5),
                   "val_acc": round(val_acc, 5), "lr": lr,
                   "epoch_time_s": round(time.perf_counter() - start, 2)}
            writer.writerow(row)
            f.flush()  # Colab ngắt giữa chừng thì log các epoch đã xong vẫn còn
            history.append(row)

            improved = best is None or val_loss < best["val_loss"]
            if improved:
                best, bad_epochs = row, 0
                torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch,
                            "val_loss": val_loss, "val_acc": val_acc,
                            "train_mean": info["mean"], "train_std": info["std"]},
                           out_dir / "checkpoints" / f"{run_id}.pt")
            else:
                bad_epochs += 1
            print(f"  epoch {epoch:2d} | train {train_loss:.4f} / {train_acc:.4f} | "
                  f"val {val_loss:.4f} / {val_acc:.4f} | lr {lr:.0e} | "
                  f"{row['epoch_time_s']:.1f}s{' *' if improved else ''}")

            if scheduler is not None:
                scheduler.step(val_loss)
            if bad_epochs >= patience:
                print(f"  Early stopping: val_loss {patience} epoch liền không giảm.")
                break

    summary = {
        "run_id": run_id, "change": change, "seed": seed,
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "val_acc_at_best": best["val_acc"], "train_acc_at_best": best["train_acc"],
        "epochs_ran": len(history),
        "train_time_s": round(sum(r["epoch_time_s"] for r in history), 1),
        "n_params": n_params,
    }
    append_summary(out_dir / "summary.csv", summary)
    print(f"[{run_id}] tốt nhất epoch {best['epoch']}: val_loss {best['val_loss']:.4f}, "
          f"val_acc {best['val_acc']:.4f} | {summary['train_time_s']:.0f}s | "
          f"ghi vào {out_dir}")
    return summary


def append_summary(path, row):
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run(run_id, override=None, change="", workers=None, **kwargs):
    """Điểm vào cho experiments/run_E*.py: một run = baseline + override."""
    cfg = load_config(override=override)
    if workers is not None:  # không phải biến thực nghiệm, chỉ để chạy được trên máy yếu
        cfg["train"]["num_workers"] = workers
    return train(run_id, cfg, change=change, **kwargs)


# ---------------------------------------------------------------- dòng lệnh
def parse_set(items):
    """["train.lr=0.0001", "model.batchnorm=true"] -> {"train.lr": 0.0001, "model.batchnorm": True}."""
    override = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--set cần dạng nhóm.khóa=giá_trị, nhận '{item}'")
        override[key.strip()] = yaml.safe_load(value)
    return override


def main():
    parser = argparse.ArgumentParser(description="Train một run CNN trên FER2013")
    parser.add_argument("--run", required=True, help="mã run: BASE, E1-1, …")
    parser.add_argument("--set", action="append", metavar="KHÓA=GIÁ_TRỊ",
                        help="ghi đè config, vd. --set model.arch=large (lặp lại được)")
    parser.add_argument("--change", default="", help="mô tả thay đổi so với baseline (ghi vào summary)")
    parser.add_argument("--workers", type=int, help="ghi đè num_workers (máy thiếu RAM dùng 0)")
    parser.add_argument("--thu", action="store_true",
                        help="chạy thử: ít epoch, ít batch, ghi vào results/_thu/")
    parser.add_argument("--epochs", type=int, default=2, help="số epoch khi --thu (mặc định 2)")
    parser.add_argument("--max-batches", type=int, default=20, help="số batch mỗi epoch khi --thu")
    args = parser.parse_args()

    kwargs = {}
    if args.thu:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kwargs = dict(out_dir=RESULTS_DIR / "_thu" / f"{args.run}_{stamp}",
                      max_epochs=args.epochs, max_batches=args.max_batches)
    run(args.run, override=parse_set(args.set), change=args.change, workers=args.workers, **kwargs)


if __name__ == "__main__":
    main()
