# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 基于 README 中实测数据生成展示图表

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_BASE = "#4C72B0"
C_OK = "#2E7D32"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "DejaVu Sans"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------
# 图 1：融合收益（latency 2.1x + 显存 -33.29%）
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

labels = ["PyTorch\n(分开算)", "Triton\n(融合)"]
lat = [2.10, 1.00]
bars = axes[0].bar(labels, lat, color=[C_BASE, C_OK], width=0.5)
bars[1].set_edgecolor("black")
bars[1].set_linewidth(1.8)
axes[0].text(0, 2.10 + 0.05, "2.10x", ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[0].text(1, 0.5, "1.00x", ha="center", va="center", fontweight="bold", fontsize=11,
             color="white")
axes[0].set_ylabel("Normalized latency")
axes[0].set_title("Latency (N=4096, M=512)", fontweight="bold", fontsize=10.5)
axes[0].set_ylim(0, 2.6)
axes[0].annotate("2.1x faster", xy=(1, 1.02), xytext=(0.2, 2.35),
                 arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.8),
                 fontsize=10, color=C_OK, fontweight="bold")

mem = [100.0, 66.71]
bars2 = axes[1].bar(labels, mem, color=[C_BASE, C_OK], width=0.5)
bars2[1].set_edgecolor("black")
bars2[1].set_linewidth(1.8)
axes[1].text(0, 100.0 + 2, "100%", ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[1].text(1, 33, "66.71%", ha="center", va="center", fontweight="bold", fontsize=11,
             color="white")
axes[1].set_ylabel("Peak memory (%)")
axes[1].set_title("Peak Memory: -33.29%", fontweight="bold", fontsize=10.5)
axes[1].set_ylim(0, 125)
axes[1].annotate("-33.29%", xy=(1, 67), xytext=(0.35, 112),
                 arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.8),
                 fontsize=10, color=C_OK, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_fusion_gain.png"), bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------
# 图 0：架构图
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.4)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.13",
                                facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold")


def arrow(x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color))


# 上半：未融合
ax.text(1.3, 3.95, "Unfused (PyTorch): HBM round-trip", fontsize=9.5, fontweight="bold",
        color="#B71C1C")
box(0.3, 3.05, 1.6, 0.75, "LayerNorm", "#E3F2FD", C_BASE, fs=9)
box(2.3, 3.05, 1.6, 0.75, "GELU", "#FFF3E0", "#DD8452", fs=9)
box(4.3, 3.05, 1.7, 0.75, "Write/Read HBM", "#FFEBEE", "#C62828", fs=8)
arrow(1.9, 3.42, 2.3, 3.42)
arrow(3.9, 3.42, 4.3, 3.42)
ax.annotate("", xy=(4.3, 3.2), xytext=(1.9, 3.2),
            arrowprops=dict(arrowstyle="<->", color="#C62828", lw=1.6, ls="--"))
ax.text(3.1, 3.0, "HBM round-trip (wasted)", ha="center", fontsize=8, color="#C62828",
        style="italic")

# 下半：融合
ax.text(1.3, 2.5, "Fused (Triton): stay in registers", fontsize=9.5, fontweight="bold",
        color=C_OK)
ax.add_patch(FancyBboxPatch((0.3, 0.7), 5.7, 1.5,
                            boxstyle="round,pad=0.07,rounding_size=0.15",
                            facecolor="#E8F5E9", edgecolor=C_OK, linewidth=1.8, alpha=0.6))
ax.text(0.5, 1.95, "single Triton kernel", fontsize=8.5, color=C_OK, fontweight="bold")
box(0.55, 1.0, 1.5, 0.7, "LayerNorm\n(tl.mean/var)", "#FFFFFF", C_OK, fs=8)
box(2.3, 1.0, 1.5, 0.7, "GELU (tl.erf)\nexact formula", "#FFFFFF", C_OK, fs=8)
box(4.05, 1.0, 1.7, 0.7, "output\n(no HBM write)", "#FFFFFF", C_OK, fs=8)
arrow(2.05, 1.35, 2.3, 1.35, color=C_OK)
arrow(3.8, 1.35, 4.05, 1.35, color=C_OK)
ax.text(3.1, 0.82, "intermediate value stays in register", ha="center", fontsize=8,
        color=C_OK, style="italic")

# 迭代链
box(6.4, 2.6, 1.45, 0.8, "01 vector add\n@triton.jit", "#F5F5F5", "#9E9E9E", fs=8)
box(6.4, 1.55, 1.45, 0.8, "02 matmul\ntl.dot tiling", "#F5F5F5", "#9E9E9E", fs=8)
box(6.4, 0.5, 1.45, 0.8, "03 softmax\nonline", "#F5F5F5", "#9E9E9E", fs=8)
arrow(7.125, 2.6, 7.125, 2.35, color="#9E9E9E")
arrow(7.125, 1.55, 7.125, 1.3, color="#9E9E9E")
box(8.2, 1.55, 1.6, 0.8, "04 fused LN+GELU\n(this repo)", "#FFF8E1", "#F9A825", fs=8.5)
arrow(7.85, 1.95, 8.2, 1.95, color="#F9A825")
ax.text(9.0, 1.35, "2.1x / -33.29% mem", ha="center", fontsize=8, color="#F9A825",
        fontweight="bold")
ax.text(7.85, 3.5, "Iteration chain", fontsize=9.5, fontweight="bold", color="#616161")

ax.text(5.5, 4.2, "Fused LayerNorm + GELU in Triton", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#263238")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig0_architecture.png"), bbox_inches="tight", dpi=170)
plt.close(fig)

print("已生成:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}  ({os.path.getsize(os.path.join(FIG_DIR, f))} bytes)")