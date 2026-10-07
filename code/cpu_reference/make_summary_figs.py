"""
Builds summary bar-chart figures from the paper's REAL, already-reported
numbers (no new experiments here -- just visualizing data already in the
manuscript's tables):

  1. figures/attack_success_rate_cifar10.png   -- ASR = 1 - RobustAcc (Eq. 9)
     for TRADES/MART/SPAT/ACAT on CIFAR-10 (from Table III's real numbers)
     and for Standard/PGD-AT/TRADES/MART/ACAT on the CIFAR-10 compact-CNN
     run (from Table VI's real numbers), grouped by attack.
  2. figures/mes_comparison_all_datasets.png   -- MES (Eq. 12) side-by-side
     bar chart across every dataset x method combination reported in the
     paper (Tables III, IV, and VI).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Attack Success Rate on CIFAR-10 (main WideResNet-28-10 protocol, Table III)
# ---------------------------------------------------------------------------
methods_main = ["TRADES", "MART", "SPAT", "ACAT (ours)"]
colors_main = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]
attacks_main = ["FGSM", "PGD-100", "C&W", "AA"]
# accuracy (%) from Table III (CIFAR-10)
acc_main = {
    "TRADES":     [60.06, 55.12, 53.35, 52.60],
    "MART":       [61.59, 56.37, 52.75, 51.40],
    "SPAT":       [63.55, 60.57, 52.47, 50.64],
    "ACAT (ours)":[63.66, 59.04, 54.01, 52.43],
}
asr_main = {m: [100 - a for a in accs] for m, accs in acc_main.items()}

fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(attacks_main))
width = 0.2
for i, m in enumerate(methods_main):
    ax.bar(x + (i - 1.5) * width, asr_main[m], width, label=m, color=colors_main[i])
ax.set_xticks(x)
ax.set_xticklabels(attacks_main)
ax.set_ylabel("Attack Success Rate (%)")
ax.set_title("Attack Success Rate on CIFAR-10 (WideResNet-28-10 protocol)\nASR = 100% $-$ robust accuracy (Eq. 9), lower is better")
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out1 = os.path.join(FIG_DIR, "attack_success_rate_cifar10.png")
fig.savefig(out1, dpi=200)
print("Saved", out1)

# ---------------------------------------------------------------------------
# 2. MES comparison across all datasets/methods reported in the paper
# ---------------------------------------------------------------------------
# (dataset, method) -> MES value (%), all taken verbatim from the paper's tables
mes_data = [
    ("CIFAR-10\n(WRN-28-10)", "TRADES", 96.29, "#4C72B0"),
    ("CIFAR-10\n(WRN-28-10)", "MART", 96.40, "#55A868"),
    ("CIFAR-10\n(WRN-28-10)", "SPAT", 97.81, "#8172B2"),
    ("CIFAR-10\n(WRN-28-10)", "ACAT", 99.43, "#C44E52"),
    ("CIFAR-100\n(WRN-28-10)", "TRADES", 92.90, "#4C72B0"),
    ("CIFAR-100\n(WRN-28-10)", "MART", 90.11, "#55A868"),
    ("CIFAR-100\n(WRN-28-10)", "SPAT", 95.69, "#8172B2"),
    ("CIFAR-100\n(WRN-28-10)", "ACAT", 97.07, "#C44E52"),
    ("CIFAR-10\n(compact CNN)", "PGD-AT", 87.43, "#DD8452"),
    ("CIFAR-10\n(compact CNN)", "TRADES", 75.51, "#4C72B0"),
    ("CIFAR-10\n(compact CNN)", "MART", 82.74, "#55A868"),
    ("CIFAR-10\n(compact CNN)", "ACAT", 87.78, "#C44E52"),
    ("Fashion-MNIST\n(compact CNN)", "PGD-AT", 96.49, "#DD8452"),
    ("Fashion-MNIST\n(compact CNN)", "TRADES", 98.21, "#4C72B0"),
    ("Fashion-MNIST\n(compact CNN)", "MART", 97.58, "#55A868"),
    ("Fashion-MNIST\n(compact CNN)", "ACAT", 96.63, "#C44E52"),
    ("Tiny-ImageNet\n(compact CNN)", "PGD-AT", 81.66, "#DD8452"),
    ("Tiny-ImageNet\n(compact CNN)", "TRADES", 75.02, "#4C72B0"),
    ("Tiny-ImageNet\n(compact CNN)", "MART", 91.72, "#55A868"),
    ("Tiny-ImageNet\n(compact CNN)", "ACAT", 94.17, "#C44E52"),
]

datasets_order = ["CIFAR-10\n(WRN-28-10)", "CIFAR-100\n(WRN-28-10)", "CIFAR-10\n(compact CNN)",
                   "Fashion-MNIST\n(compact CNN)", "Tiny-ImageNet\n(compact CNN)"]
fig, axes = plt.subplots(1, len(datasets_order), figsize=(3.1 * len(datasets_order), 4.0), sharey=True)
for ax, ds in zip(axes, datasets_order):
    rows = [r for r in mes_data if r[0] == ds]
    methods = [r[1] for r in rows]
    vals = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    bars = ax.bar(methods, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_title(ds, fontsize=9.5)
    ax.set_ylim(70, 101)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("MES (%, Eq. 12)")
fig.suptitle("Multi-threat Efficiency Score (MES) across every dataset and method reported in this paper", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out2 = os.path.join(FIG_DIR, "mes_comparison_all_datasets.png")
fig.savefig(out2, dpi=200)
print("Saved", out2)
