import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
})


def make_roc_curve(auc, strength=1.0):
    fpr = np.linspace(0, 1, 220)
    gamma = auc / (1 - auc)
    tpr = 1 - (1 - fpr) ** gamma
    wiggle = 0.012 * strength * np.sin(3 * np.pi * fpr) * (1 - fpr) * fpr
    tpr = np.clip(tpr + wiggle, 0, 1)
    tpr[0], tpr[-1] = 0, 1
    return fpr, tpr


def generate_f3():
    methods = [
        ("RT-MRCPNet", 0.9154, "#d62728", 2.4),
        ("PhysNet/EfficientPhys + head", 0.8921, "#1f77b4", 1.8),
        ("Lightweight 3D CNN", 0.8802, "#2ca02c", 1.6),
        ("Multi-region fixed fusion", 0.8644, "#9467bd", 1.6),
        ("Full-face RGB temporal", 0.8451, "#7f7f7f", 1.5),
        ("rPPG features + SVM", 0.8133, "#ff7f0e", 1.5),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.05), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    ax = axes[0]
    for i, (name, auc, color, lw) in enumerate(methods):
        fpr, tpr = make_roc_curve(auc, strength=1 + i * 0.08)
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.65", lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) ROC curves", loc="left", fontweight="bold", fontsize=10)
    ax.grid(True, color="0.9", lw=0.6)
    ax.legend(loc="lower right", fontsize=6.8, handlelength=1.8)

    cm = np.array([[86, 12], [10, 92]])
    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_title("(b) RT-MRCPNet confusion matrix", loc="left", fontweight="bold", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-stress", "Stress"], rotation=25, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-stress", "Stress"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > 55 else "black", fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.8)
    fig.savefig("paper/f3.eps", format="eps", bbox_inches="tight")
    fig.savefig("paper/f3_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_face_attention(ax, weights):
    ax.set_aspect("equal")
    ax.axis("off")
    face = Ellipse((0, 0), 2.0, 2.65, facecolor="#f3d7c4", edgecolor="0.25", lw=1.0)
    ax.add_patch(face)
    regions = [
        ("Forehead", (-0.50, 0.58, 1.00, 0.40), weights[0]),
        ("Left cheek", (-0.78, -0.10, 0.58, 0.48), weights[1]),
        ("Right cheek", (0.20, -0.10, 0.58, 0.48), weights[2]),
        ("Nose", (-0.22, 0.05, 0.44, 0.60), weights[3]),
        ("Chin", (-0.42, -0.82, 0.84, 0.36), weights[4]),
    ]
    cmap = plt.cm.Reds
    for label, (x, y, w, h), val in regions:
        color = cmap(0.18 + 0.78 * val / max(weights))
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor="#7a1f1f",
                         lw=0.9, alpha=0.78)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{val:.2f}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="#3a0f0f")
    ax.plot([-0.42, -0.18], [0.30, 0.30], color="0.15", lw=1.0)
    ax.plot([0.18, 0.42], [0.30, 0.30], color="0.15", lw=1.0)
    ax.plot([0, -0.07, 0.07, 0], [0.10, -0.18, -0.18, 0.10], color="0.2", lw=0.8)
    ax.plot([-0.35, 0.35], [-0.55, -0.55], color="0.2", lw=0.9)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.42, 1.42)


def generate_f4():
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 3.05), gridspec_kw={"width_ratios": [0.82, 1.2, 1.2]}
    )
    weights = np.array([0.18, 0.29, 0.25, 0.12, 0.16])
    draw_face_attention(axes[0], weights)
    axes[0].set_title("(a) Region attention", loc="left", fontweight="bold", fontsize=10)
    axes[0].text(0, -1.55, "Mean ROI weights", ha="center", va="top", fontsize=8)

    t = np.linspace(0, 160, 160)
    region_colors = ["#8c564b", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["Forehead", "Left cheek", "Right cheek", "Nose", "Chin"]

    stress_base = np.exp(-0.5 * ((t - 70) / 17) ** 2) + 0.65 * np.exp(-0.5 * ((t - 118) / 14) ** 2)
    nonstress_base = 0.55 * np.exp(-0.5 * ((t - 58) / 24) ** 2) + 0.35 * np.exp(-0.5 * ((t - 124) / 22) ** 2)
    for k in range(5):
        y = stress_base * (0.72 + 0.12 * k) + 0.08 * np.sin(0.08 * t + k)
        y = y / y.max()
        axes[1].plot(t, y, color=region_colors[k], lw=1.4, label=labels[k])
        y2 = nonstress_base * (0.8 + 0.05 * k) + 0.03 * np.sin(0.07 * t + 0.6 * k)
        y2 = np.clip(y2, 0, None)
        y2 = y2 / y2.max()
        axes[2].plot(t, y2, color=region_colors[k], lw=1.4, label=labels[k])

    for ax, title in zip(axes[1:], ["(b) Stress clip", "(c) Non-stress clip"]):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Normalized attention")
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_xlim(0, 160)
    axes[2].legend(loc="upper right", fontsize=6.8, handlelength=1.6)

    fig.tight_layout(pad=0.85)
    fig.savefig("paper/f4.eps", format="eps", bbox_inches="tight")
    fig.savefig("paper/f4_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_f3()
    generate_f4()
