"""Vẽ Hình 4.1 (so sánh Keras ↔ PyTorch) từ results/so_sanh.csv.

Chạy ở thư mục BTL03:  python report/ve_hinh_tong_hop.py
Ghi: report/hinh/so_sanh_keras_pytorch.png. Không gõ số tay: mọi số đọc từ so_sanh.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "hinh" / "so_sanh_keras_pytorch.png"
KERAS, PYTORCH, MUTED, GRID = "#257de0", "#f47230", "#666666", "#e3e3e3"

TEN = {  # tên hiển thị, theo thứ tự mục 3 của báo cáo
    "MNIST_MLP": "3.2 MNIST MLP",
    "1_MNIST_CNN": "3.3 MNIST CNN",
    "2_Human_Gender_CNN": "3.4 Gender CNN",
    "3_Emotion_Reg_CNN": "3.5 Emotion CNN",
    "tu_2_Human_Gender_CNN": "3.7 Gender + callback",
    "tu_3_Emotion_Reg_CNN": "3.8 Emotion + augmentation",
    "1_segnet": "3.9 SegNet (pixel acc)",
    "2_obj_detection": "3.10 Phát hiện xe",
}

df = pd.read_csv(ROOT / "results" / "so_sanh.csv")
df = df[df["notebook"].isin(TEN)].set_index("notebook").loc[list(TEN)[::-1]]
k, p = df["keras"] * 100, df["pytorch"] * 100
d = p - k
y = range(len(df))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True)

for yi, a, b in zip(y, k, p):
    ax1.plot([a, b], [yi, yi], color="#bbbbbb", lw=2, zorder=1)
# Keras: vòng tròn rỗng lớn; PyTorch: chấm đặc nhỏ, nên khi hai số gần trùng vẫn thấy cả hai
ax1.scatter(k, y, s=140, facecolor="white", edgecolor=KERAS, linewidth=2.2, label="Keras (gốc)", zorder=3)
ax1.scatter(p, y, s=45, color=PYTORCH, label="PyTorch", zorder=4)
ax1.set_yticks(list(y))
ax1.set_yticklabels([TEN[n] for n in df.index])
ax1.set_xlabel("Accuracy trên tập test/validation của gốc (%)")
ax1.set_xlim(55, 102)
ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)

ax2.barh(list(y), d, color=MUTED, height=0.55)
ax2.axvline(0, color="#333333", lw=1)
for yi, v in zip(y, d):
    ax2.text(v + (0.15 if v >= 0 else -0.15), yi, f"{v:+.2f}", va="center",
             ha="left" if v >= 0 else "right", fontsize=9, color="#222222")
ax2.set_xlabel("PyTorch − Keras (điểm %)")
ax2.set_xlim(-2.6, 4.9)

for ax in (ax1, ax2):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("Đã ghi", OUT)
print(pd.DataFrame({"keras": k.round(2), "pytorch": p.round(2), "chenh": d.round(2)}))
