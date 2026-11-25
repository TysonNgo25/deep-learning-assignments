"""Nhìn vào bên trong CNN: filter lớp conv cuối, feature map, Grad-CAM.

Phụ trách: Phạm Trần Minh Khang (filter conv cuối, feature map), Ngô Nguyễn Thành Nhân (Grad-CAM)

    python -m src.visualize                                   # mô hình cuối results/final/best_model.pt
    python -m src.visualize --ckpt results/BASE/checkpoints/BASE.pt --out results/BASE/figures
Chỉ dùng ảnh validation (PublicTest); không đọc PrivateTest.

Hình ghi vào --out (mặc định <thư mục checkpoint>/figures):
    last_conv_filters.png     ảnh tổng hợp làm từng filter của lớp conv cuối kích hoạt mạnh nhất
    last_conv_top_images.png  ảnh validation thật kích hoạt mạnh nhất từng filter
    last_conv_weights.png     trọng số 3×3 của filter (trung bình theo kênh vào)
    feature_maps.png          một ảnh đi qua các khối conv
    gradcam.png               Grad-CAM tại conv đầu / giữa / cuối, mỗi cảm xúc 1 ảnh đoán đúng
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data import NUM_CLASSES, load_fer2013
from src.evaluate import load_checkpoint, predict
from src.models import conv_layers
from src.paths import EMOTIONS, RESULTS_DIR, ROOT


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def to_input(images, mean, std):
    """Ảnh uint8 N×48×48 -> tensor N×1×48×48 chuẩn hóa như lúc train."""
    x = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    return (x - mean) / std


def activation_after(model, conv):
    """Lớp kích hoạt ngay sau một Conv2d trong model.features (bỏ qua BatchNorm nếu có).

    Lấy đầu ra sau kích hoạt vì đó là thứ lớp sau thực sự nhận được (feature map).
    """
    mods = list(model.features)
    i = next(k for k, m in enumerate(mods) if m is conv)
    return next(m for m in mods[i + 1:] if not isinstance(m, nn.BatchNorm2d))


def capture(model, module, x):
    """Chạy model trên x, trả về (logits, đầu ra của module)."""
    store = {}
    handle = module.register_forward_hook(lambda m, i, o: store.__setitem__("out", o))
    try:
        logits = model(x)
    finally:
        handle.remove()
    return logits, store["out"]


@torch.no_grad()
def mean_activations(model, module, x, batch_size=256):
    """Mức kích hoạt trung bình theo không gian của từng kênh, cho từng ảnh: N×C."""
    outs = [capture(model, module, x[i:i + batch_size])[1].mean(dim=(2, 3))
            for i in range(0, len(x), batch_size)]
    return torch.cat(outs).cpu().numpy()


# ---------------------------------------------------------------- filter lớp conv cuối
def filter_patterns(model, conv, filters, device, steps=150, lr=0.05, seed=0):
    """Activation maximization: tìm ảnh đầu vào làm mỗi filter phản ứng mạnh nhất.

    Bắt đầu từ nhiễu nhỏ, tăng dần ảnh theo gradient để làm lớn trung bình đầu ra của filter.
    Dùng đầu ra conv TRƯỚC kích hoạt: với ReLU, filter đang bằng 0 thì gradient sau ReLU cũng 0
    và ảnh không đổi. Ba cách chống nhiễu tần số cao (Yosinski và cộng sự, 2015): mỗi bước dịch
    ảnh ngẫu nhiên vài pixel, phạt độ lớn ảnh (L2), và cứ 4 bước làm mờ ảnh bằng Gaussian 3×3.
    """
    blur = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=device).div(16).view(1, 1, 3, 3)
    gen = torch.Generator().manual_seed(seed)
    img = (0.1 * torch.randn(len(filters), 1, 48, 48, generator=gen)).to(device).requires_grad_(True)
    opt = torch.optim.Adam([img], lr=lr)
    idx = torch.arange(len(filters), device=device)
    filt = torch.as_tensor(filters, device=device)
    model.eval()
    for step in range(steps):
        if step % 4 == 3:
            with torch.no_grad():
                img.copy_(F.conv2d(img, blur, padding=1))
        dx, dy = torch.randint(-2, 3, (2,), generator=gen).tolist()
        shifted = torch.roll(img, shifts=(dx, dy), dims=(2, 3))
        _, out = capture(model, conv, shifted)
        act = out[idx, filt].mean(dim=(1, 2))
        loss = -act.sum() + 1e-3 * img.pow(2).sum()
        # Chỉ cần gradient theo ảnh; trọng số mô hình giữ nguyên, không tính gradient cho chúng.
        img.grad, = torch.autograd.grad(loss, img)
        opt.step()
    return img.detach().squeeze(1).cpu().numpy()


def _grid(n, cols):
    return int(np.ceil(n / cols)), cols


def plot_filter_patterns(patterns, filters, path, title):
    plt = _plt()
    rows, cols = _grid(len(filters), 8)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.45), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, p, f in zip(axes.flat, patterns, filters):
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)  # mỗi ảnh tự co giãn về [0, 1] để xem mẫu hình
        ax.imshow(p, cmap="gray")
        ax.set_title(f"filter {f}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_filter_weights(conv, filters, path):
    """Trọng số 3×3 của từng filter, lấy trung bình trên các kênh vào (đỏ dương, xanh âm)."""
    plt = _plt()
    w = conv.weight.detach().cpu().numpy()[filters].mean(axis=1)  # n×3×3
    lim = np.abs(w).max()
    rows, cols = _grid(len(filters), 8)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, k, f in zip(axes.flat, w, filters):
        ax.imshow(k, cmap="bwr", vmin=-lim, vmax=lim)
        ax.set_title(f"{f}", fontsize=7)
    fig.suptitle(f"Trọng số lớp conv cuối ({conv.weight.shape[1]} kênh vào × 3×3, "
                 f"trung bình theo kênh vào)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_images(acts, images, filters, path, k=6):
    """Mỗi hàng một filter: k ảnh validation làm filter đó kích hoạt mạnh nhất."""
    plt = _plt()
    fig, axes = plt.subplots(len(filters), k, figsize=(k * 1.2, len(filters) * 1.3), squeeze=False)
    for r, f in enumerate(filters):
        top = np.argsort(-acts[:, f])[:k]
        for c, i in enumerate(top):
            ax = axes[r, c]
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(f"filter {f}", fontsize=8)
    fig.suptitle("Ảnh validation kích hoạt mạnh nhất từng filter lớp conv cuối", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- feature map
@torch.no_grad()
def plot_feature_maps(model, x1, image, title, path, per_layer=8):
    """Một ảnh qua mạng: mỗi hàng là đầu ra (sau kích hoạt) của conv cuối mỗi khối,
    hiện per_layer kênh hoạt động mạnh nhất. Càng sâu, ảnh càng nhỏ (48 -> 24 -> 12) và càng trừu tượng."""
    plt = _plt()
    convs = [m for _, m in conv_layers(model)]
    mods = list(model.features)
    # conv cuối của mỗi khối = conv đứng ngay trước một MaxPool
    block_ends = []
    for k, m in enumerate(mods):
        if isinstance(m, nn.MaxPool2d):
            block_ends.append([c for c in convs if mods.index(c) < k][-1])
    rows = len(block_ends)
    fig, axes = plt.subplots(rows, per_layer + 1, figsize=((per_layer + 1) * 1.3, rows * 1.45),
                             squeeze=False)
    for r, conv in enumerate(block_ends):
        _, out = capture(model, activation_after(model, conv), x1)
        out = out[0].cpu().numpy()
        top = np.argsort(-out.mean(axis=(1, 2)))[:per_layer]
        axes[r, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title("ảnh vào" if r == 0 else "", fontsize=7)
        axes[r, 0].set_ylabel(f"khối {r + 1}\n{out.shape[1]}×{out.shape[2]}", fontsize=8)
        for c, ch in enumerate(top, start=1):
            axes[r, c].imshow(out[ch], cmap="viridis")
            axes[r, c].set_title(f"kênh {ch}", fontsize=7)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Grad-CAM
def grad_cam(model, x1, target, module):
    """Grad-CAM (Selvaraju và cộng sự, 2017) cho một ảnh x1 (1×1×48×48) và lớp `target`.

    A = feature map tại `module`; trọng số mỗi kênh α_k = trung bình không gian của ∂y_target/∂A_k;
    bản đồ = ReLU(Σ_k α_k·A_k), phóng về 48×48 rồi chuẩn hóa về [0, 1].
    Vùng sáng là vùng làm điểm của lớp đó tăng.
    """
    model.eval()
    logits, act = capture(model, module, x1)
    grad, = torch.autograd.grad(logits[0, target], act)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x1.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def plot_gradcam(model, x, images, labels, y_pred, probs, path):
    """Hàng đầu: ảnh gốc; 3 hàng sau: Grad-CAM tại conv đầu, giữa, cuối.
    Mỗi cột một cảm xúc: ảnh validation đoán đúng với xác suất cao nhất của lớp đó."""
    plt = _plt()
    convs = conv_layers(model)
    picks = {"conv đầu": convs[0], "conv giữa": convs[len(convs) // 2], "conv cuối": convs[-1]}
    device = next(model.parameters()).device
    fig, axes = plt.subplots(len(picks) + 1, NUM_CLASSES, figsize=(NUM_CLASSES * 1.6, 7.2),
                             squeeze=False)
    for c, name in enumerate(EMOTIONS):
        ok = np.flatnonzero((labels == c) & (y_pred == c))
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
        if len(ok) == 0:  # mô hình không đoán đúng ảnh nào của lớp này
            axes[0, c].set_title(f"{name}\n(không có ca đúng)", fontsize=8)
            continue
        i = ok[np.argmax(probs[ok, c])]
        x1 = x[i:i + 1].to(device)
        axes[0, c].imshow(images[i], cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(f"{name}\np={probs[i, c]:.2f}", fontsize=8)
        for r, (label, (conv_name, conv)) in enumerate(picks.items(), start=1):
            cam = grad_cam(model, x1, c, activation_after(model, conv))
            axes[r, c].imshow(images[i], cmap="gray", vmin=0, vmax=255)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if c == 0:
                axes[r, 0].set_ylabel(f"{label}\n({conv_name})", fontsize=8)
    fig.suptitle("Grad-CAM trên ảnh validation đoán đúng: vùng đỏ làm tăng điểm của lớp đúng",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- chạy tất cả
def visualize(ckpt_path, out_dir=None, csv_path=None, n_filters=64, steps=150):
    ckpt_path = Path(ckpt_path)
    # results/final/best_model.pt -> results/final/figures; results/E1/checkpoints/E1-3.pt -> results/E1/figures
    base = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    out_dir = Path(out_dir or base / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(ckpt_path, device)
    mean, std = ckpt["train_mean"], ckpt["train_std"]
    csv_path = Path(csv_path) if csv_path else ROOT / ckpt["config"]["data"]["csv"]
    images, labels = load_fer2013(csv_path, splits=("val",))["val"]
    x = to_input(images, mean, std)
    y_pred, probs = predict(model, images, labels, mean, std, device)
    print(f"{ckpt_path.name}: epoch {ckpt['epoch']}, val_acc {np.mean(y_pred == labels):.4f}")

    # Filter lớp conv cuối: chọn n_filters filter hoạt động mạnh nhất trên validation
    # (filter "chết", luôn bằng 0 sau ReLU, không có gì để xem).
    last = conv_layers(model)[-1][1]
    acts = mean_activations(model, activation_after(model, last), x.to(device))
    filters = np.argsort(-acts.mean(axis=0))[:n_filters].tolist()
    n_dead = int((acts.max(axis=0) == 0).sum())
    print(f"Lớp conv cuối: {last.out_channels} filter, {n_dead} filter không kích hoạt trên ảnh val nào")

    patterns = filter_patterns(model, last, filters, device, steps=steps)
    plot_filter_patterns(patterns, filters, out_dir / "last_conv_filters.png",
                         f"Lớp conv cuối ({last.out_channels} filter): ảnh làm từng filter kích hoạt "
                         f"mạnh nhất, {len(filters)} filter hoạt động mạnh nhất trên validation")
    plot_top_images(acts, images, filters[:10], out_dir / "last_conv_top_images.png")
    plot_filter_weights(last, filters, out_dir / "last_conv_weights.png")

    # Feature map của ảnh Happy đoán đúng tự tin nhất (dễ thấy miệng cười qua các khối).
    happy = EMOTIONS.index("Happy")
    ok = np.flatnonzero((labels == happy) & (y_pred == happy))
    i = ok[np.argmax(probs[ok, happy])] if len(ok) else 0
    plot_feature_maps(model, x[i:i + 1].to(device), images[i],
                      f"Feature map của một ảnh validation ({EMOTIONS[labels[i]]}) qua các khối conv",
                      out_dir / "feature_maps.png")

    plot_gradcam(model, x, images, labels, y_pred, probs, out_dir / "gradcam.png")
    print(f"Đã ghi 5 hình vào {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Filter conv cuối, feature map, Grad-CAM (chỉ dùng validation)")
    parser.add_argument("--ckpt", type=Path, default=RESULTS_DIR / "final" / "best_model.pt")
    parser.add_argument("--out", type=Path, help="mặc định <thư mục checkpoint>/figures")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--n-filters", type=int, default=64)
    parser.add_argument("--steps", type=int, default=150, help="số bước tối ưu ảnh cho mỗi filter")
    args = parser.parse_args()
    visualize(args.ckpt, args.out, args.csv, args.n_filters, args.steps)


if __name__ == "__main__":
    main()
