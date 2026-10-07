"""
Latent-space (t-SNE) visualizations, computed from real forward passes
through this project's actually-trained model checkpoints -- not synthetic
or illustrative data.

Produces:
  1. figures/latent_space_before_after_all_datasets.png
     2 rows (before training / after ACAT training) x 3 columns (CIFAR-10,
     Fashion-MNIST, Tiny-ImageNet subset), t-SNE of the penultimate-layer
     embedding (model.forward_features), colored by true class label. "Before
     training" uses a freshly-initialized (untrained, random-weight) copy of
     the exact same architecture as a baseline for what an unstructured
     embedding space looks like; "after training" uses the real trained ACAT
     (seed 42) checkpoint for that dataset.
  2. figures/latent_space_across_backbones_cifar10.png
     After-ACAT-training-only, one panel per backbone (SmallCNN,
     WideResNet-28-10, ResNet-18, VGG-11, MobileNet) on CIFAR-10, to show how
     backbone choice affects the shape of the learned embedding space at this
     project's compute scale.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import get_dataset, DATASET_INFO
from src.models import build_model, NormalizedModel
from src.utils import set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SAMPLES = 300
SEED = 42


def load_samples(dataset, n=N_SAMPLES, seed=0):
    _, test_ds, class_names = get_dataset(dataset)
    rng = np.random.RandomState(seed)
    n = min(n, len(test_ds))
    idxs = rng.choice(len(test_ds), size=n, replace=False)
    xs, ys = [], []
    for i in idxs:
        x, y, _ = test_ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys), class_names


def build_normalized(dataset, backbone="smallcnn", width=32):
    info = DATASET_INFO[dataset]
    net = build_model(info, width=width, backbone=backbone)
    return NormalizedModel(net, info["mean"], info["std"])


def load_trained(dataset, backbone, ckpt_path, width=32):
    model = build_normalized(dataset, backbone=backbone, width=width)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def embed(model, x, batch_size=64):
    feats = []
    for i in range(0, x.size(0), batch_size):
        feats.append(model.forward_features(x[i:i + batch_size]))
    return torch.cat(feats, 0).numpy()


def tsne2d(feats, seed=SEED):
    n = feats.shape[0]
    perplexity = max(5, min(30, n // 4))
    return TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                init="pca", learning_rate="auto").fit_transform(feats)


def scatter_panel(ax, xy, labels, class_names, title):
    cmap = plt.get_cmap("tab20" if len(class_names) > 10 else "tab10")
    for c in range(len(class_names)):
        mask = labels == c
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, color=cmap(c % 20), alpha=0.75,
                   label=class_names[c])
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Figure 1: before vs. after ACAT training, 3 datasets (SmallCNN backbone,
# the one backbone with an existing trained ACAT checkpoint on all 3
# real-executed datasets).
# ---------------------------------------------------------------------------
def make_before_after_figure():
    datasets = ["cifar10", "fashionmnist", "tinyimagenet"]
    labels_map = {"cifar10": "CIFAR-10", "fashionmnist": "Fashion-MNIST", "tinyimagenet": "Tiny-ImageNet"}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for col, dataset in enumerate(datasets):
        x, y, class_names = load_samples(dataset, seed=0)

        set_seed(0)
        untrained = build_normalized(dataset, backbone="smallcnn", width=32)
        untrained.eval()
        feats_before = embed(untrained, x)
        xy_before = tsne2d(feats_before)

        ckpt = os.path.join(RESULTS_DIR, dataset, "acat", "seed42", "model.pt")
        trained = load_trained(dataset, "smallcnn", ckpt, width=32)
        feats_after = embed(trained, x)
        xy_after = tsne2d(feats_after)

        scatter_panel(axes[0, col], xy_before, y.numpy(), class_names,
                      f"{labels_map[dataset]} -- before training")
        scatter_panel(axes[1, col], xy_after, y.numpy(), class_names,
                      f"{labels_map[dataset]} -- after ACAT training")

        handles, labels_ = axes[1, col].get_legend_handles_labels()
        ncol = 5 if len(class_names) > 10 else 5
        axes[1, col].legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                            ncol=ncol, fontsize=6.5, frameon=False, markerscale=1.3)

    axes[0, 0].set_ylabel("t-SNE dim 2\n(random-init weights)", fontsize=9)
    axes[1, 0].set_ylabel("t-SNE dim 2\n(trained ACAT weights)", fontsize=9)
    fig.suptitle("Latent-space distribution (t-SNE of penultimate-layer features) before vs. after "
                  "ACAT training, on real held-out test samples", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, "latent_space_before_after_all_datasets.png")
    fig.savefig(out, dpi=200)
    print("Saved", out)


# ---------------------------------------------------------------------------
# Figure 2: after-training-only, across all 5 backbones, CIFAR-10.
# ---------------------------------------------------------------------------
def make_across_backbones_figure():
    x, y, class_names = load_samples("cifar10", seed=0)

    backbone_ckpts = [
        ("SmallCNN", "smallcnn", os.path.join(RESULTS_DIR, "cifar10", "acat", "seed42", "model.pt"), 32),
        ("WideResNet-28-10", "wrn28_10",
         os.path.join(PROJECT_ROOT, "results_wrn28_10", "acat", "model.pt"), 32),
        ("ResNet-18", "resnet18",
         os.path.join(PROJECT_ROOT, "results_multi_model", "resnet18", "acat", "model.pt"), 32),
        ("VGG-11", "vgg11",
         os.path.join(PROJECT_ROOT, "results_multi_model", "vgg11", "acat", "model.pt"), 32),
        ("MobileNet", "mobilenet",
         os.path.join(PROJECT_ROOT, "results_multi_model", "mobilenet", "acat", "model.pt"), 32),
    ]
    available = [(name, bb, path, w) for name, bb, path, w in backbone_ckpts if os.path.exists(path)]
    if not available:
        print("No backbone checkpoints found yet -- skipping across-backbones figure.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 4.3))
    if len(available) == 1:
        axes = [axes]
    for ax, (name, backbone, ckpt, width) in zip(axes, available):
        model = load_trained("cifar10", backbone, ckpt, width=width)
        feats = embed(model, x)
        xy = tsne2d(feats)
        scatter_panel(ax, xy, y.numpy(), class_names, name)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=10, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Latent-space distribution (t-SNE) on CIFAR-10 after ACAT training, "
                  "by backbone architecture", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out = os.path.join(FIG_DIR, "latent_space_across_backbones_cifar10.png")
    fig.savefig(out, dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    make_before_after_figure()
    make_across_backbones_figure()
