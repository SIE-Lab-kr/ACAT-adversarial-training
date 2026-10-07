"""
Computes real margin-based hardness h_i (Eq. 1) and boundary proximity b_i
(Eq. 2) for held-out test images, using the actual ACAT (seed 42) model
checkpoints trained in this project for CIFAR-10, Fashion-MNIST, and the
Tiny-ImageNet subset (results/<dataset>/acat/seed42/model.pt) -- i.e. every
number plotted here comes from a real forward pass through a real,
code-executed trained model, not a synthetic or illustrative distribution.
CIFAR-100 is not included because no checkpoint was trained for it in this
CPU-only sandbox (Table II); its results in the paper come from a separate,
GPU-trained codebase this project does not have.

Produces:
  figures/hardness_distribution_all_datasets.png
  figures/boundary_distribution_all_datasets.png
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import get_dataset, DATASET_INFO
from src.models import build_model, NormalizedModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DATASETS = ["cifar10", "fashionmnist", "tinyimagenet"]
LABELS = {"cifar10": "CIFAR-10", "fashionmnist": "Fashion-MNIST", "tinyimagenet": "Tiny-ImageNet"}
N_SAMPLES = 200
SEED = 42


@torch.no_grad()
def compute_h_b(model, x, y):
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    true_prob = probs.gather(1, y.unsqueeze(1)).squeeze(1)
    other_prob = probs.clone()
    other_prob.scatter_(1, y.unsqueeze(1), -1.0)
    max_other_prob = other_prob.max(1)[0]
    h_i = (true_prob - max_other_prob).cpu().numpy()

    eps = 1e-8
    max_abs_h = np.abs(h_i).max() + eps
    b_i = 1.0 - (np.abs(h_i) / max_abs_h)
    return h_i, b_i


def load_model(dataset):
    info = DATASET_INFO[dataset]
    ckpt_path = os.path.join(RESULTS_DIR, dataset, "acat", f"seed{SEED}", "model.pt")
    net = build_model(info, width=32, backbone="smallcnn")
    model = NormalizedModel(net, info["mean"], info["std"])
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    device = torch.device("cpu")
    all_h, all_b, correct_frac = {}, {}, {}

    for dataset in DATASETS:
        model = load_model(dataset).to(device)
        _, test_ds, _ = get_dataset(dataset)
        rng = np.random.RandomState(0)
        n = min(N_SAMPLES, len(test_ds))
        idxs = rng.choice(len(test_ds), size=n, replace=False)

        xs, ys = [], []
        for i in idxs:
            x, y, _ = test_ds[i]
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys).to(device)

        h_i, b_i = compute_h_b(model, x, y)
        all_h[dataset] = h_i
        all_b[dataset] = b_i
        correct_frac[dataset] = float((h_i > 0).mean())
        print(f"{dataset}: n={n} mean_h={h_i.mean():.3f} mean_b={b_i.mean():.3f} "
              f"frac_correct(h>0)={correct_frac[dataset]:.3f}")

    # ---- Hardness distribution figure ----
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 3.4), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        h_i = all_h[dataset]
        ax.hist(h_i, bins=25, color="#4C72B0", edgecolor="white", alpha=0.9)
        ax.axvline(0, color="#C44E52", linestyle="--", linewidth=1.2)
        ax.set_title(LABELS[dataset])
        ax.set_xlabel(r"Margin-based hardness $h_i$")
    axes[0].set_ylabel("Number of test samples")
    fig.suptitle("Distribution of margin-based hardness $h_i$ on real held-out test samples\n"
                  "(computed from the trained ACAT model, Eq. 1)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out1 = os.path.join(FIG_DIR, "hardness_distribution_all_datasets.png")
    fig.savefig(out1, dpi=200)
    print("Saved", out1)

    # ---- Boundary proximity distribution figure ----
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 3.4), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        b_i = all_b[dataset]
        ax.hist(b_i, bins=25, color="#DD8452", edgecolor="white", alpha=0.9, range=(0, 1))
        ax.set_title(LABELS[dataset])
        ax.set_xlabel(r"Boundary proximity $b_i$")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Number of test samples")
    fig.suptitle("Distribution of boundary proximity $b_i$ on real held-out test samples\n"
                  "(computed from the trained ACAT model, Eq. 2; $b_i\\to1$ = near the decision boundary)",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out2 = os.path.join(FIG_DIR, "boundary_distribution_all_datasets.png")
    fig.savefig(out2, dpi=200)
    print("Saved", out2)

    # ---- Save the raw numbers too, for transparency ----
    import json
    summary = {
        dataset: {
            "n": len(all_h[dataset]),
            "h_mean": float(all_h[dataset].mean()), "h_std": float(all_h[dataset].std()),
            "b_mean": float(all_b[dataset].mean()), "b_std": float(all_b[dataset].std()),
            "frac_correct": correct_frac[dataset],
        } for dataset in DATASETS
    }
    with open(os.path.join(RESULTS_DIR, "hardness_boundary_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved", os.path.join(RESULTS_DIR, "hardness_boundary_summary.json"))


if __name__ == "__main__":
    main()
