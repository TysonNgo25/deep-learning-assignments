"""Đọc FER2013, chia tập theo cột Usage, tạo Dataset/DataLoader có augmentation, và vẽ EDA.

Phụ trách: Huỳnh Ngọc Trúc

Chạy kiểm tra dữ liệu + EDA (đứng ở thư mục BTL02):
    python -m src.data
Kết quả ghi vào results/eda/.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.paths import CONFIG_DIR, EMOTIONS, FER_CSV, RESULTS_DIR, ROOT

IMAGE_SIZE = 48
NUM_CLASSES = len(EMOTIONS)

# Tên tập trong code -> giá trị cột Usage của fer2013.csv.
SPLITS = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
EXPECTED_COUNTS = {"train": 28709, "val": 3589, "test": 3589}


# ---------------------------------------------------------------- đọc dữ liệu
def load_fer2013(csv_path=FER_CSV, splits=("train", "val")):
    """Trả về {tên tập: (ảnh uint8 N×48×48, nhãn int64 N)}.

    Mặc định không đọc PrivateTest: tập test chỉ được lấy khi đánh giá mô hình cuối,
    nên phải ghi rõ splits=(..., "test") mới có.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không thấy {csv_path}. Tải fer2013.csv từ Kaggle và đặt vào thư mục data/."
        )
    # Đọc từng dòng và đổi ngay sang uint8 thay vì pd.read_csv: pandas giữ cả ~300 MB chuỗi
    # pixel trong RAM (máy Trúc báo out of memory), còn cách này chỉ tốn ~83 MB cho ảnh.
    wanted = {SPLITS[name]: name for name in splits}
    images = {name: [] for name in splits}
    labels = {name: [] for name in splits}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = wanted.get(row["Usage"])
            if name is None:  # dòng của tập không cần (vd. PrivateTest khi train) thì bỏ qua
                continue
            images[name].append(_parse_pixels(row["pixels"]))
            labels[name].append(int(row["emotion"]))
    return {name: (np.stack(images[name]), np.array(labels[name], dtype=np.int64))
            for name in splits}


def _parse_pixels(text):
    """Chuỗi 2304 số cách nhau bởi dấu cách -> ảnh 48×48 uint8."""
    pixels = np.array(text.split(), dtype=np.uint8)
    if pixels.size != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"Ảnh có {pixels.size} pixel, cần {IMAGE_SIZE * IMAGE_SIZE}.")
    return pixels.reshape(IMAGE_SIZE, IMAGE_SIZE)


def compute_mean_std(images, chunk=4096):
    """Mean/std của pixel (đã chia 255) trên tập train.

    Cộng dồn theo từng khối ảnh để không phải tạo một mảng float64 ~0,5 GB cùng lúc.
    """
    total, total_sq, n = 0.0, 0.0, 0
    for i in range(0, len(images), chunk):
        block = images[i:i + chunk].astype(np.float64) / 255.0
        total += block.sum()
        total_sq += np.square(block).sum()
        n += block.size
    mean = total / n
    std = np.sqrt(total_sq / n - mean ** 2)
    return float(mean), float(std)


def class_counts(labels):
    """Số ảnh của từng lớp 0..6 (lớp không có ảnh vẫn ra 0)."""
    return np.bincount(labels, minlength=NUM_CLASSES)


def class_weights(labels):
    """Trọng số lớp w_c = N / (K · n_c) cho CrossEntropy có trọng số (E6-3).

    Lớp càng ít ảnh (Disgust) thì trọng số càng lớn; lớp có đúng N/K ảnh được trọng số 1.
    """
    counts = class_counts(labels).astype(np.float64)
    weights = len(labels) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------- Dataset
def build_train_transform(aug, fill=0.0):
    """Augmentation cho tập train từ khối `augmentation` của config.

    Tương ứng với ImageDataGenerator của Keras trong code mẫu:
    horizontal_flip -> RandomHorizontalFlip, rotation_range -> degrees,
    width/height_shift_range -> translate, zoom_range -> scale.
    `fill` là giá trị điền vào phần viền trống sau khi xoay/dịch; truyền mean của train
    để viền thành 0 sau chuẩn hóa thay vì một dải đen bất thường mà mạng có thể học nhầm.
    Trả về None khi tắt hết (E4-1: không augmentation).
    """
    steps = []
    if aug.get("hflip", False):
        steps.append(T.RandomHorizontalFlip(p=0.5))
    rotation = aug.get("rotation_deg", 0) or 0
    shift = aug.get("shift", 0) or 0
    zoom = aug.get("zoom", 0) or 0
    if rotation or shift or zoom:
        steps.append(T.RandomAffine(
            degrees=rotation,
            translate=(shift, shift) if shift else None,
            scale=(1 - zoom, 1 + zoom) if zoom else None,
            fill=fill,
        ))
    return T.Compose(steps) if steps else None


class FERDataset(Dataset):
    """Ảnh FER2013 dạng tensor 1×48×48 đã chuẩn hóa theo mean/std của train.

    Giữ ảnh ở uint8 (≈66 MB cho tập train) và chỉ đổi sang float khi lấy từng ảnh.
    Augmentation chạy trên ảnh [0, 1] rồi mới chuẩn hóa. Khi num_workers = 0, phép
    biến đổi ngẫu nhiên dùng RNG chung của torch, nên set_seed() trong train.py phải
    gọi trước khi tạo DataLoader thì kết quả mới lặp lại được.
    """

    def __init__(self, images, labels, mean, std, transform=None):
        self.images = torch.from_numpy(images).unsqueeze(1)  # N×1×48×48, uint8
        self.labels = torch.from_numpy(labels)
        self.mean, self.std = mean, std
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx].float() / 255.0
        if self.transform is not None:
            x = self.transform(x)
        x = (x - self.mean) / self.std
        return x, self.labels[idx]


def get_dataloaders(cfg, include_test=False, seed=None):
    """Tạo DataLoader cho train/val (và test nếu include_test=True).

    cfg là dict đọc từ configs/baseline.yaml. Trả về (loaders, info), trong đó info có
    mean/std của train, số ảnh mỗi lớp và trọng số lớp để train.py dùng.
    Chỉ tập train được xáo trộn và augmentation; val/test giữ nguyên ảnh gốc.
    """
    splits = ("train", "val", "test") if include_test else ("train", "val")
    data = load_fer2013(_csv_from_cfg(cfg), splits=splits)
    x_train, y_train = data["train"]
    mean, std = compute_mean_std(x_train)

    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 0)
    seed = cfg.get("seed", 0) if seed is None else seed
    # Generator cố định để thứ tự xáo trộn và augmentation lặp lại được theo seed.
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    loaders = {
        "train": DataLoader(
            FERDataset(x_train, y_train, mean, std,
                       build_train_transform(cfg["augmentation"], fill=mean)),
            shuffle=True, generator=generator, **common,
        )
    }
    for name in splits[1:]:
        x, y = data[name]
        loaders[name] = DataLoader(FERDataset(x, y, mean, std), shuffle=False, **common)

    info = {
        "mean": mean,
        "std": std,
        "counts": {name: class_counts(data[name][1]).tolist() for name in splits},
        "class_weights": class_weights(y_train),
    }
    return loaders, info


def _csv_from_cfg(cfg):
    """Đường dẫn CSV trong config là tương đối so với thư mục BTL02."""
    return ROOT / cfg["data"]["csv"]


# ---------------------------------------------------------------- kiểm tra + EDA
def check_counts(data):
    """In số ảnh từng tập so với số chuẩn của FER2013; trả về True nếu khớp hết."""
    ok = True
    for name, (images, labels) in data.items():
        n, expected = len(labels), EXPECTED_COUNTS[name]
        n_classes = int((class_counts(labels) > 0).sum())
        good = n == expected and n_classes == NUM_CLASSES
        ok &= good
        status = "OK " if good else "SAI"
        print(f"[{status}] {name:5s} ({SPLITS[name]:11s}): {n:6d} ảnh (chuẩn {expected}), "
              f"{n_classes}/{NUM_CLASSES} lớp, shape {images.shape[1:]}")
    return ok


def run_eda(csv_path=FER_CSV, out_dir=RESULTS_DIR / "eda", seed=0):
    """Kiểm số ảnh và vẽ hình EDA cho mục 1 của báo cáo.

    PrivateTest chỉ dùng để mô tả dữ liệu (đếm ảnh mỗi lớp, đếm ảnh trùng với train),
    không vẽ ảnh test, không tính thống kê nào đưa vào mô hình.
    """
    import matplotlib
    matplotlib.use("Agg")  # không cần màn hình, chạy được trên Colab/Kaggle
    import matplotlib.pyplot as plt
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_fer2013(csv_path, splits=("train", "val", "test"))
    counts_ok = check_counts(data)

    x_train, y_train = data["train"]
    mean, std = compute_mean_std(x_train)
    counts = pd.DataFrame(
        {name: class_counts(y) for name, (_, y) in data.items()}, index=EMOTIONS
    )
    counts.index.name = "emotion"
    counts["train_pct"] = (100 * counts["train"] / counts["train"].sum()).round(2)
    counts.to_csv(out_dir / "class_counts.csv")
    print("\nSố ảnh mỗi lớp:\n" + counts.to_string())

    train_counts = counts["train"]
    stats = {
        "counts": {name: int(len(y)) for name, (_, y) in data.items()},
        "counts_match_fer2013": bool(counts_ok),
        "train_pixel_mean": round(mean, 4),
        "train_pixel_std": round(std, 4),
        "largest_class": train_counts.idxmax(),
        "smallest_class": train_counts.idxmin(),
        "imbalance_ratio": round(float(train_counts.max() / max(train_counts.min(), 1)), 2),
        "class_weights": [round(w, 4) for w in class_weights(y_train).tolist()],
        "duplicates": duplicate_stats(data),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"\nMean/std pixel tập train (sau khi chia 255): {mean:.4f} / {std:.4f}")
    print(f"Lớp nhiều nhất {stats['largest_class']}, ít nhất {stats['smallest_class']}, "
          f"chênh {stats['imbalance_ratio']} lần")
    print("Ảnh trùng: " + json.dumps(stats["duplicates"], ensure_ascii=False))

    _plot_class_distribution(plt, counts, out_dir / "class_distribution.png")
    rng = np.random.default_rng(seed)
    _plot_samples(plt, x_train, y_train, rng, out_dir / "samples_per_class.png")
    _plot_mean_faces(plt, x_train, y_train, out_dir / "mean_face_per_class.png")
    baseline = yaml.safe_load((CONFIG_DIR / "baseline.yaml").read_text(encoding="utf-8"))
    _plot_augmentation(plt, x_train, rng, baseline["augmentation"], mean, seed,
                       out_dir / "augmentation_examples.png")
    print(f"\nĐã ghi hình và bảng vào {out_dir}")
    return counts_ok


def duplicate_stats(data):
    """Đếm ảnh trùng pixel trong và giữa các tập (so bằng mã băm MD5 của 2304 byte ảnh).

    Ảnh của tập val/test trùng y hệt một ảnh train nghĩa là mô hình đã "thấy" ảnh đó lúc train,
    nên accuracy trên các ảnh này cao hơn thực tế. Ảnh trùng mà khác nhãn cho thấy nhãn nhiễu.
    Ảnh một màu (min = max) không chứa khuôn mặt.
    """
    import hashlib

    keys = {name: [hashlib.md5(img.tobytes()).hexdigest() for img in images]
            for name, (images, _) in data.items()}
    out = {}
    for name, ks in keys.items():
        out[f"{name}_unique"] = len(set(ks))
        out[f"{name}_duplicate_images"] = len(ks) - len(set(ks))
    all_keys = [k for ks in keys.values() for k in ks]
    out["all_unique"] = len(set(all_keys))
    train_set = set(keys.get("train", []))
    for name in ("val", "test"):
        if name in keys:
            out[f"{name}_images_also_in_train"] = sum(k in train_set for k in keys[name])
    if "val" in keys and "test" in keys:
        val_set = set(keys["val"])
        out["test_images_also_in_val"] = sum(k in val_set for k in keys["test"])
    labels_of = {}
    for name, (_, labels) in data.items():
        for k, y in zip(keys[name], labels):
            labels_of.setdefault(k, set()).add(int(y))
    out["pixel_groups_with_conflicting_labels"] = sum(len(v) > 1 for v in labels_of.values())
    out["constant_images"] = {name: int(sum(img.min() == img.max() for img in images))
                              for name, (images, _) in data.items()}
    return out


def _plot_class_distribution(plt, counts, path):
    names = ["train", "val", "test"]
    labels = {"train": "Training (train)", "val": "PublicTest (val)", "test": "PrivateTest (test)"}
    x = np.arange(NUM_CLASSES)
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(names):
        bars = ax.bar(x + (i - 1) * width, counts[name], width, label=labels[name])
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(x, EMOTIONS)
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố lớp của FER2013 theo từng tập")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_samples(plt, images, labels, rng, path, per_class=8):
    """Mỗi hàng một cảm xúc, chọn ngẫu nhiên (theo seed) per_class ảnh của tập train."""
    fig, axes = plt.subplots(NUM_CLASSES, per_class, figsize=(per_class * 1.1, NUM_CLASSES * 1.2))
    for c, name in enumerate(EMOTIONS):
        idx = np.flatnonzero(labels == c)
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for j, ax in enumerate(axes[c]):
            ax.axis("off")
            if j < len(chosen):
                ax.imshow(images[chosen[j]], cmap="gray", vmin=0, vmax=255)
        axes[c, 0].set_title(name, loc="left", fontsize=9)
    fig.suptitle("Ảnh mẫu tập train theo từng lớp")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mean_faces(plt, images, labels, path):
    """Ảnh trung bình mỗi lớp: cho thấy các lớp khác nhau chủ yếu ở vùng miệng, mắt."""
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(NUM_CLASSES * 1.5, 2))
    for c, (ax, name) in enumerate(zip(axes, EMOTIONS)):
        idx = labels == c
        if idx.any():
            ax.imshow(images[idx].mean(axis=0), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}\n(n={int(idx.sum())})", fontsize=8)
        ax.axis("off")
    fig.suptitle("Ảnh trung bình của từng lớp (tập train)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_augmentation(plt, images, rng, aug_cfg, fill, seed, path, n_aug=7):
    """Một ảnh gốc và n_aug phiên bản sau augmentation của baseline (giống lúc train)."""
    transform = build_train_transform(aug_cfg, fill=fill)
    img = torch.from_numpy(images[rng.integers(len(images))]).unsqueeze(0).float() / 255.0
    torch.manual_seed(seed)
    versions = [img] + [transform(img) if transform else img for _ in range(n_aug)]
    fig, axes = plt.subplots(1, len(versions), figsize=(len(versions) * 1.4, 1.8))
    for i, (ax, v) in enumerate(zip(axes, versions)):
        ax.imshow(v.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title("Gốc" if i == 0 else f"Aug {i}", fontsize=8)
        ax.axis("off")
    desc = (f"lật={aug_cfg.get('hflip')}, xoay ±{aug_cfg.get('rotation_deg')}°, "
            f"dịch {aug_cfg.get('shift')}, zoom {aug_cfg.get('zoom')}")
    fig.suptitle(f"Augmentation baseline ({desc})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra FER2013 và vẽ EDA")
    parser.add_argument("--csv", type=Path, default=FER_CSV, help="mặc định data/fer2013.csv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "eda", help="mặc định results/eda")
    args = parser.parse_args()
    ok = run_eda(args.csv, args.out)
    if not ok:
        print("\nCẢNH BÁO: số ảnh không khớp FER2013 chuẩn (28709/3589/3589). Kiểm tra lại file CSV.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
