"""Vẽ scatter 3D, bề mặt hàm thật, histogram x1, x2, y từ data/*.csv; lưu hình cho mục 2 báo cáo.

Phụ trách: Trúc (hạn 30/10).

Chạy:  python -m scripts.plot_data      (sau python -m src.data)
Ra:    report/figures/fig_data_scatter3d.png
       report/figures/fig_data_true_surface.png
       report/figures/fig_data_histograms.png
"""
import matplotlib

matplotlib.use("Agg")  # chỉ lưu file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.data import SPLITS, gaussian_2d, load_config
from src.paths import DATA_DIR, REPORT_DIR

FIG_DIR = REPORT_DIR / "figures"

# Mỗi tập một màu (thứ tự cố định) và một hình marker, để phân biệt được cả khi in đen trắng
STYLE = {
    "train_pool": {"color": "#2a78d6", "marker": "o", "name": "Train pool"},
    "val": {"color": "#eb6834", "marker": "^", "name": "Validation"},
    "test": {"color": "#1baf7a", "marker": "s", "name": "Test"},
}
# Thang một màu (xanh dương, nhạt -> đậm) cho độ lớn f(x)
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelcolor": INK,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def load_splits():
    return {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in SPLITS}


def label(name, df):
    return f"{STYLE[name]['name']} (n = {len(df)})"


def plot_scatter3d(splits, path):
    """Scatter 3D (x1, x2, y) của cả 3 tập trên cùng một trục."""
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(projection="3d")
    sizes = {"train_pool": 6, "val": 12, "test": 26}  # tập nhỏ vẽ to hơn, vẽ sau để nằm trên
    for name, df in splits.items():
        s = STYLE[name]
        ax.scatter(df["x1"], df["x2"], df["y"], c=s["color"], marker=s["marker"], s=sizes[name],
                   alpha=0.75, edgecolors="white", linewidths=0.3, depthshade=False, label=label(name, df))
    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$y = f(x_1, x_2)$", xticks=[-2, -1, 0, 1, 2])
    ax.set_title("Dữ liệu lấy mẫu đều trên miền $[-2, 2] \\times [-4, 4]$")
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_true_surface(cfg, path, n_grid=201):
    """Bề mặt hàm thật f(x) trên lưới đều: bên trái 3D, bên phải đường đồng mức nhìn từ trên xuống."""
    x1 = np.linspace(*cfg["x1_range"], n_grid)
    x2 = np.linspace(*cfg["x2_range"], n_grid)
    g1, g2 = np.meshgrid(x1, x2)
    z = gaussian_2d(np.column_stack([g1.ravel(), g2.ravel()]), cfg["mu"], cfg["cov"]).reshape(g1.shape)
    peak = z.max()

    fig = plt.figure(figsize=(10, 5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.05)
    ax3d = fig.add_subplot(grid[0], projection="3d")
    ax3d.plot_surface(g1, g2, z, cmap=BLUES, rstride=3, cstride=3, linewidth=0, antialiased=True)
    ax3d.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$f(x_1, x_2)$", xticks=[-2, -1, 0, 1, 2])
    ax3d.set_title("(a) Bề mặt hàm thật")
    ax3d.view_init(elev=28, azim=-58)

    ax = fig.add_subplot(grid[1])
    filled = ax.contourf(g1, g2, z, levels=20, cmap=BLUES)
    lines = ax.contour(g1, g2, z, levels=[0.01, 0.05, 0.1, 0.2], colors=INK, linewidths=0.7)
    ax.clabel(lines, fmt="%.2f", fontsize=8)
    ax.plot(*cfg["mu"], marker="+", color="white", markersize=10, markeredgewidth=1.5)
    ax.set_aspect("equal")  # giữ đúng tỷ lệ để thấy elip nghiêng do hiệp phương sai âm
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=f"(b) Đường đồng mức, đỉnh f(μ) = {peak:.4f} tại +")
    fig.colorbar(filled, ax=ax, label="$f(x_1, x_2)$")
    fig.savefig(path)
    plt.close(fig)


def plot_histograms(splits, cfg, path, near_zero=0.01):
    """Phân bố x1, x2, y của từng tập. Trục tung là % số mẫu của tập đó, để so được các tập khác cỡ."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    peak = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cfg["cov"])))
    n_bins = 10
    specs = [
        ("x1", "$x_1$", np.linspace(*cfg["x1_range"], n_bins + 1)),
        ("x2", "$x_2$", np.linspace(*cfg["x2_range"], n_bins + 1)),
        ("y", "$y$", np.linspace(0, peak, 14)),
    ]
    for ax, (col, xlabel, bins) in zip(axes, specs):
        for name, df in splits.items():
            ax.hist(df[col], bins=bins, weights=np.full(len(df), 100 / len(df)), histtype="step",
                    linewidth=1.8, color=STYLE[name]["color"], label=label(name, df))
        if col != "y":
            # Phân bố đều chia 10 khoảng bằng nhau thì mỗi khoảng có 10% số mẫu
            ax.axhline(100 / n_bins, color=MUTED, linestyle="--", linewidth=1, label="Đều lý thuyết")
            ax.set_ylim(0, 2.2 * 100 / n_bins)
        else:
            share = (splits["train_pool"]["y"] < near_zero).mean()
            ax.text(0.97, 0.95, f"{share:.1%} mẫu train có y < {near_zero}\nđỉnh f(μ) = {peak:.4f}",
                    transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
        ax.set(xlabel=xlabel, ylabel="Tỷ lệ mẫu (%)", title=f"Phân bố {xlabel}")
        ax.grid(axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    cfg = load_config()["data"]
    splits = load_splits()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "fig_data_scatter3d.png": lambda p: plot_scatter3d(splits, p),
        "fig_data_true_surface.png": lambda p: plot_true_surface(cfg, p),
        "fig_data_histograms.png": lambda p: plot_histograms(splits, cfg, p),
    }
    for filename, draw in outputs.items():
        draw(FIG_DIR / filename)
        print(f"  -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
