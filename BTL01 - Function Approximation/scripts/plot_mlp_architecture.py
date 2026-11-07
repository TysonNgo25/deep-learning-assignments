"""Render the baseline MLP architecture diagram used in the report."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.paths import REPORT_DIR


def main() -> None:
    destination = REPORT_DIR / "figures" / "mlp_baseline_architecture.png"
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(11, 3.2))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 3.2)
    axis.axis("off")

    layers = [
        (0.35, 2.0, 1.7, 0.8, "2 đầu vào\n$x_1, x_2$", "#DCEEFF"),
        (3.0, 2.0, 1.9, 0.8, "16 nơ-ron\n$tanh$", "#E5F4E3"),
        (6.0, 2.0, 1.9, 0.8, "16 nơ-ron\n$tanh$", "#E5F4E3"),
        (9.0, 2.0, 1.65, 0.8, "1 đầu ra\nTuyến tính $\\hat y$", "#FFE9CC"),
    ]

    for x, y, width, height, label, color in layers:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4,
            edgecolor="#38536B",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=12,
        )

    for left, right in zip(layers, layers[1:]):
        start = (left[0] + left[2] + 0.12, left[1] + left[3] / 2)
        end = (right[0] - 0.12, right[1] + right[3] / 2)
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#38536B",
            )
        )

    axis.text(5.5, 0.95, "MLP baseline: 2 → 16 → 16 → 1", ha="center", fontsize=13)
    axis.text(5.5, 0.48, "337 tham số được học", ha="center", fontsize=11, color="#555555")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved architecture diagram: {destination}")


if __name__ == "__main__":
    main()
