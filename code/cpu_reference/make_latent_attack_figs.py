"""
Latent-space (t-SNE) visualizations of CLEAN vs. ADVERSARIALLY-ATTACKED
embeddings, for Standard training vs. ACAT (v5 addendum).

Unlike make_latent_space_figs.py (v4, which compares random-init vs. trained
weights, and compares backbones only on clean inputs), this script asks the
question that actually matters for a robustness claim: for a given trained
model, how much does an adversarial attack move a test image's embedding
relative to where its class cluster sits on clean inputs? A defense that is
doing its job should show adversarially-attacked points still landing close
to their clean-input class cluster; a defense that is not should show attacked
points scattered away from it (often into other classes' clusters).

All embeddings come from real forward passes (model.forward_features) through
this project's real, freshly-trained results_full_matrix/ checkpoints (v5),
on real CIFAR-10 held-out images, with real PGD-20 / FGSM / CW-Linf
adversarial examples generated on the fly against each model. Nothing here is
synthetic, interpolated, or hand-drawn.

Produces:
  1. figures/latent_space_clean_vs_attack_all_backbones.png
     5 rows (SmallCNN, WideResNet-28-10, ResNet-18, VGG-11, MobileNet) x
     4 columns (Standard-clean, Standard-under-PGD-20, ACAT-clean,
     ACAT-under-PGD-20), all on the same 80 held-out CIFAR-10 images.
  2. figures/latent_space_clean_vs_attacks_smallcnn.png
     2 rows (Standard, ACAT) x 4 columns (clean, FGSM, PGD-20, CW-Linf),
     SmallCNN backbone (cheapest, so this one affords more attack types and
     more samples: 200 held-out CIFAR-10 images), to show whether ACAT's
     embedding stability (or lack of it) holds across attack types, not just
     across backbones.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.attacks import fgsm_attack, pgd_attack, cw_linf_attack
from make_latent_space_figs import load_samples, load_trained, embed, tsne2d, scatter_panel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_full_matrix")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EVAL_EPS = 8 / 255
EVAL_ALPHA = 2 / 255


def ckpt_path(backbone, method):
    return os.path.join(RESULTS_DIR, backbone, method, "model.pt")


def gen_adv(model, x, y, kind, steps=20):
    if kind == "fgsm":
        return fgsm_attack(model, x, y, EVAL_EPS)
    if kind == "pgd20":
        return pgd_attack(model, x, y, EVAL_EPS, EVAL_ALPHA, steps)
    if kind == "cw_linf":
        return cw_linf_attack(model, x, y, EVAL_EPS, EVAL_ALPHA, steps=15)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Figure 1: clean vs. PGD-20-attacked, Standard vs. ACAT, across all 5
# backbones.
# ---------------------------------------------------------------------------
def make_across_backbones_attack_figure(n_vis=80):
    x, y, class_names = load_samples("cifar10", n=n_vis, seed=7)

    backbones = [
        ("SmallCNN", "smallcnn", 32),
        ("WideResNet-28-10", "wrn28_10", 32),
        ("ResNet-18", "resnet18", 32),
        ("VGG-11", "vgg11", 32),
        ("MobileNet", "mobilenet", 32),
    ]
    available = [(n, bb, w) for n, bb, w in backbones
                 if os.path.exists(ckpt_path(bb, "standard")) and os.path.exists(ckpt_path(bb, "acat"))]
    if not available:
        print("No results_full_matrix checkpoints found yet -- skipping across-backbones attack figure.")
        return

    fig, axes = plt.subplots(len(available), 4, figsize=(15.5, 3.7 * len(available)))
    if len(available) == 1:
        axes = axes[None, :]
    col_titles = ["Standard -- clean", "Standard -- under PGD-20", "ACAT -- clean", "ACAT -- under PGD-20"]

    for row, (name, backbone, width) in enumerate(available):
        model_std = load_trained("cifar10", backbone, ckpt_path(backbone, "standard"), width=width)
        model_acat = load_trained("cifar10", backbone, ckpt_path(backbone, "acat"), width=width)

        x_adv_std = gen_adv(model_std, x, y, "pgd20")
        x_adv_acat = gen_adv(model_acat, x, y, "pgd20")

        panels = [
            (embed(model_std, x), model_std),
            (embed(model_std, x_adv_std), model_std),
            (embed(model_acat, x), model_acat),
            (embed(model_acat, x_adv_acat), model_acat),
        ]
        for col, (feats, _) in enumerate(panels):
            xy = tsne2d(feats)
            title = col_titles[col] if row == 0 else ""
            scatter_panel(axes[row, col], xy, y.numpy(), class_names, title)
        axes[row, 0].set_ylabel(name, fontsize=10)

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=10, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Latent-space distribution: clean vs. PGD-20-attacked inputs, Standard vs. ACAT, "
                  "across all 5 backbones (real forward passes, real PGD-20 attacks)", fontsize=12)
    fig.tight_layout(rect=[0.01, 0.05, 1, 0.96])
    out = os.path.join(FIG_DIR, "latent_space_clean_vs_attack_all_backbones.png")
    fig.savefig(out, dpi=200)
    print("Saved", out)


# ---------------------------------------------------------------------------
# Figure 2: clean vs. {FGSM, PGD-20, CW-Linf}, Standard vs. ACAT, SmallCNN.
# ---------------------------------------------------------------------------
def make_across_attacks_figure(n_vis=200):
    x, y, class_names = load_samples("cifar10", n=n_vis, seed=7)
    backbone, width = "smallcnn", 32

    if not (os.path.exists(ckpt_path(backbone, "standard")) and os.path.exists(ckpt_path(backbone, "acat"))):
        print("No results_full_matrix SmallCNN checkpoints found yet -- skipping across-attacks figure.")
        return

    model_std = load_trained("cifar10", backbone, ckpt_path(backbone, "standard"), width=width)
    model_acat = load_trained("cifar10", backbone, ckpt_path(backbone, "acat"), width=width)

    attack_kinds = ["clean", "fgsm", "pgd20", "cw_linf"]
    attack_titles = ["Clean", "Under FGSM", "Under PGD-20", "Under CW-Linf"]

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
    for row, (label, model) in enumerate([("Standard", model_std), ("ACAT (ours)", model_acat)]):
        for col, kind in enumerate(attack_kinds):
            x_in = x if kind == "clean" else gen_adv(model, x, y, kind)
            feats = embed(model, x_in)
            xy = tsne2d(feats)
            title = attack_titles[col] if row == 0 else ""
            scatter_panel(axes[row, col], xy, y.numpy(), class_names, title)
        axes[row, 0].set_ylabel(label, fontsize=11)

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=10, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Latent-space distribution on CIFAR-10 (SmallCNN backbone): Standard vs. ACAT, "
                  "clean vs. three real attacks (FGSM, PGD-20, CW-Linf)", fontsize=12)
    fig.tight_layout(rect=[0.01, 0.04, 1, 0.95])
    out = os.path.join(FIG_DIR, "latent_space_clean_vs_attacks_smallcnn.png")
    fig.savefig(out, dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    make_across_backbones_attack_figure()
    make_across_attacks_figure()
