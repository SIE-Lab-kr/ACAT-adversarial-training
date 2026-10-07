"""
Grouped bar chart comparing all 5 training objectives (Standard, PGD-AT,
TRADES, MART, ACAT) across the 3 new backbones (ResNet-18, VGG-11,
MobileNet) added in this round, all trained/evaluated under the identical
small-subset CIFAR-10 protocol in run_multi_model_comparison.py -- so, unlike
mixing in SmallCNN/WideResNet-28-10 numbers from other (larger-scale)
protocols, every bar here is directly comparable to every other bar.

Reads results_multi_model/comparison_summary.csv (real, code-executed
numbers) -- draws nothing by hand.
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

with open(os.path.join(PROJECT_ROOT, "results_multi_model", "comparison_summary.csv")) as f:
    rows = list(csv.DictReader(f))

backbones = ["resnet18", "vgg11", "mobilenet"]
backbone_labels = {"resnet18": "ResNet-18\n(11.2M params)", "vgg11": "VGG-11\n(9.2M params)",
                    "mobilenet": "MobileNet\n(0.5M params)"}
methods = ["standard", "pgd_at", "trades", "mart", "acat"]
method_labels = {"standard": "Standard", "pgd_at": "PGD-AT", "trades": "TRADES",
                  "mart": "MART", "acat": "ACAT (ours)"}
colors = {"standard": "#8c8c8c", "pgd_at": "#DD8452", "trades": "#4C72B0",
          "mart": "#55A868", "acat": "#C44E52"}

data = {(r["backbone"], r["method"]): r for r in rows}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(backbones))
width = 0.16

for ax, metric, title in zip(axes, ["clean_acc", "pgd20_acc"],
                              ["Clean accuracy (%)", "PGD-20 robust accuracy (%)"]):
    for i, method in enumerate(methods):
        vals = [float(data[(bb, method)][metric]) for bb in backbones]
        bars = ax.bar(x + (i - 2) * width, vals, width, label=method_labels[method], color=colors[method])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([backbone_labels[bb] for bb in backbones], fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0, max(35, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.3)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("All 5 training objectives x 3 new backbones on CIFAR-10 (identical small-subset "
             "smoke-test protocol -- real, code-executed, directly comparable)", fontsize=11)
fig.tight_layout(rect=[0, 0.1, 1, 0.92])
out = os.path.join(FIG_DIR, "multi_model_comparison_cifar10.png")
fig.savefig(out, dpi=200)
print("Saved", out)
