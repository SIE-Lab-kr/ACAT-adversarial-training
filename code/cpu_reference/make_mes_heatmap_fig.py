"""
Full-matrix summary heatmaps (v5 addendum): MES and mean robust accuracy
across all 5 methods x all 5 backbones, from
results_full_matrix/comparison_summary_full.csv (real, code-executed numbers
-- draws nothing by hand).

Produces figures/mes_heatmap_full_matrix.png: two heatmaps side by side
(clean accuracy as a reference panel, then MES), rows = backbone, columns =
method, annotated with the actual value in each cell.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = os.path.join(PROJECT_ROOT, "results_full_matrix", "comparison_summary_full.csv")

BACKBONES = ["smallcnn", "wrn28_10", "resnet18", "vgg11", "mobilenet"]
BACKBONE_LABELS = {"smallcnn": "SmallCNN\n(0.14M)", "wrn28_10": "WideResNet-28-10\n(36.48M)",
                    "resnet18": "ResNet-18\n(11.17M)", "vgg11": "VGG-11\n(9.23M)",
                    "mobilenet": "MobileNet\n(0.54M)"}
METHODS = ["standard", "pgd_at", "trades", "mart", "acat"]
METHOD_LABELS = {"standard": "Standard", "pgd_at": "PGD-AT", "trades": "TRADES",
                  "mart": "MART", "acat": "ACAT\n(ours)"}


def load_rows():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    return {(r["backbone"], r["method"]): r for r in rows}


def build_matrix(data, field):
    mat = np.full((len(BACKBONES), len(METHODS)), np.nan)
    for i, bb in enumerate(BACKBONES):
        for j, m in enumerate(METHODS):
            r = data.get((bb, m))
            if r is not None:
                mat[i, j] = float(r[field])
    return mat


def draw_heatmap(ax, mat, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8.5)
    ax.set_yticks(range(len(BACKBONES)))
    ax.set_yticklabels([BACKBONE_LABELS[b] for b in BACKBONES], fontsize=8.5)
    ax.set_title(title, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                color = "white" if (vmin is not None and vmax is not None and
                                     (val - vmin) / max(vmax - vmin, 1e-9) < 0.6) else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, color=color)
    return im


def main():
    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} not found yet -- run run_full_matrix_v5.py first. Skipping.")
        return
    data = load_rows()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    mat_clean = build_matrix(data, "clean_acc")
    mat_robust = build_matrix(data, "mean_robust_acc")
    mat_mes = build_matrix(data, "MES")

    im0 = draw_heatmap(axes[0], mat_clean, "Clean accuracy (%)", vmin=0, vmax=max(35, np.nanmax(mat_clean)))
    im1 = draw_heatmap(axes[1], mat_robust, "Mean robust accuracy (%)\n(avg. over FGSM/PGD-20/APGD-20/"
                                             "CW-Linf/Square/Transfer)", vmin=0, vmax=max(35, np.nanmax(mat_robust)))
    im2 = draw_heatmap(axes[2], mat_mes, "Multi-threat Efficiency Score (MES, %)",
                        cmap="magma", vmin=0, vmax=max(35, np.nanmax(mat_mes)))
    for im, ax in zip([im0, im1, im2], axes):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

    fig.suptitle("Full 5-method x 5-backbone matrix on CIFAR-10 -- all six attacks "
                  "(FGSM, PGD-20, APGD-20, CW-Linf, Square, Transfer) + MES, real code-executed "
                  "smoke-test-scale results", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIG_DIR, "mes_heatmap_full_matrix.png")
    fig.savefig(out, dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    main()
