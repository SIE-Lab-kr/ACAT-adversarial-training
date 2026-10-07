"""
Regenerates figs/training_dynamics_cifar10.png from the v2 (longer-epoch,
multi-seed) CIFAR-10 runs: two panels (train loss, in-loop PGD-10 robust
accuracy) vs. epoch, one line per method, mean across seeds with a shaded
+/- 1 std band when more than one seed is available.
"""
import os
import json
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METHOD_LABEL = {
    "standard": "Standard", "pgd_at": "PGD-AT", "trades": "TRADES",
    "mart": "MART", "acat": "ACAT (ours)",
}
METHOD_ORDER = ["standard", "pgd_at", "trades", "mart", "acat"]
COLORS = {"standard": "#999999", "pgd_at": "#1f77b4", "trades": "#ff7f0e",
          "mart": "#2ca02c", "acat": "#d62728"}
DATASET = "cifar10"


def discover_seeds(method):
    seeds = []
    for p in glob.glob(os.path.join(RESULTS_DIR, DATASET, method, "seed*")):
        m = re.search(r"seed(\d+)$", p)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def load_histories(method):
    hists = []
    for seed in discover_seeds(method):
        p = os.path.join(RESULTS_DIR, DATASET, method, f"seed{seed}", "history.json")
        if os.path.exists(p):
            with open(p) as f:
                hists.append(json.load(f))
    return hists


def stacked(hists, key):
    arrs = [np.array(h[key]) for h in hists]
    min_len = min(len(a) for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    stacked_arr = np.stack(arrs, axis=0)
    return stacked_arr.mean(0), stacked_arr.std(0), min_len


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for method in METHOD_ORDER:
    hists = load_histories(method)
    if not hists:
        continue
    loss_mean, loss_std, n = stacked(hists, "train_loss")
    acc_mean, acc_std, _ = stacked(hists, "pgd10_robust_acc")
    epochs = np.arange(1, n + 1)
    color = COLORS[method]
    label = METHOD_LABEL[method] + (f" (n={len(hists)} seeds)" if len(hists) > 1 else "")

    axes[0].plot(epochs, loss_mean, color=color, label=label, linewidth=1.8)
    if len(hists) > 1:
        axes[0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.15)

    axes[1].plot(epochs, acc_mean, color=color, label=label, linewidth=1.8)
    if len(hists) > 1:
        axes[1].fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, color=color, alpha=0.15)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training loss")
axes[0].set_title("(a) Training loss")
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("PGD-10 robust accuracy")
axes[1].set_title("(b) In-loop PGD-10 robust accuracy")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="best")

fig.tight_layout()
out_path = os.path.join(FIG_DIR, "training_dynamics_cifar10.png")
fig.savefig(out_path, dpi=200)
print("Saved", out_path)
