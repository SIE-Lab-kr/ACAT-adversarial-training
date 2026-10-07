"""
2D decision-boundary visualization from a real trained model (v6).

The classic adversarial-ML figure: pick a real test image x0, define a 2D
plane through input space spanned by (1) the real adversarial direction
found by PGD-20 against this exact model, and (2) a random direction
orthogonalized against it; classify every point on a grid over that plane
with the real model (a real forward pass per grid point, nothing
interpolated); color each grid cell by the model's predicted class; and mark
x0 (the clean point) and x_adv (the real adversarial example) on the plot.
Where x0's color region ends and a different-colored region begins IS this
model's decision boundary in that plane -- not a stylized illustration of one.

By default this uses the best real trained ACAT checkpoint in this
codebase (results/cifar10/acat/seed42/model.pt, SmallCNN, 20 epochs) --
pass --ckpt/--dataset/--backbone to point it at any other checkpoint,
including the results_gpu_matrix/ full-scale checkpoints once you have run
run_full_gpu_matrix.py on your own GPU.

Usage:
    python make_decision_boundary_fig.py \\
        --ckpt results/cifar10/acat/seed42/model.pt --dataset cifar10 --backbone smallcnn
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import get_dataset, DATASET_INFO
from src.models import build_model, NormalizedModel
from src.attacks import pgd_attack
from src.utils import set_seed

EVAL_EPS = 8 / 255
EVAL_ALPHA = 2 / 255


def load_model(dataset, backbone, ckpt_path, width=32):
    info = DATASET_INFO[dataset]
    net = build_model(info, width=width, backbone=backbone)
    model = NormalizedModel(net, info["mean"], info["std"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def pick_point(dataset, seed, index=None):
    _, test_ds, class_names = get_dataset(dataset)
    if index is None:
        rng = np.random.RandomState(seed)
        index = int(rng.randint(0, len(test_ds)))
    x, y, _ = test_ds[index]
    return x, y, class_names, index


@torch.no_grad()
def classify_grid(model, x0, adv_dir, rand_dir, coefs, batch_size=256):
    n = len(coefs)
    preds = np.zeros((n, n), dtype=np.int64)
    confs = np.zeros((n, n), dtype=np.float32)
    points, coords = [], []

    def flush():
        nonlocal points, coords
        if not points:
            return
        batch = torch.stack(points)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)
        conf = probs.max(1).values
        for (i, j), p, c in zip(coords, pred.numpy(), conf.numpy()):
            preds[i, j] = p
            confs[i, j] = c
        points, coords = [], []

    for i, a in enumerate(coefs):
        for j, b in enumerate(coefs):
            xp = torch.clamp(x0 + a * adv_dir + b * rand_dir, 0, 1)
            points.append(xp)
            coords.append((i, j))
            if len(points) >= batch_size:
                flush()
    flush()
    return preds, confs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/cifar10/acat/seed42/model.pt")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--backbone", default="smallcnn")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--point_index", type=int, default=None, help="test-set index of x0; omit for a random real image")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resolution", type=int, default=60)
    p.add_argument("--coef_range", type=float, default=2.5,
                    help="grid extent in units where 1.0 = the real PGD-20 adversarial example's distance from x0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt} -- train a model first (see EXECUTION_LOG_V6.md). Skipping.")
        return

    set_seed(args.seed)
    model = load_model(args.dataset, args.backbone, args.ckpt, width=args.width)
    x0, y0, class_names, index = pick_point(args.dataset, args.seed, args.point_index)
    x0b = x0.unsqueeze(0)
    y0b = torch.tensor([y0])

    with torch.no_grad():
        pred0 = model(x0b).argmax(1).item()

    x_adv = pgd_attack(model, x0b, y0b, EVAL_EPS, EVAL_ALPHA, steps=20)[0]
    with torch.no_grad():
        pred_adv = model(x_adv.unsqueeze(0)).argmax(1).item()

    adv_dir = x_adv - x0  # unnormalized: coefficient 1.0 lands exactly on x_adv
    adv_norm = adv_dir.norm().item()
    if adv_norm < 1e-8:
        print("PGD-20 did not move this point (norm ~0) -- picking a different point is recommended. Continuing anyway.")
        adv_norm = 1.0

    rng = np.random.RandomState(args.seed + 1)
    rand_dir = torch.tensor(rng.normal(size=x0.shape), dtype=torch.float32)
    # Gram-Schmidt orthogonalize rand_dir against adv_dir, then rescale to the
    # same norm as adv_dir so the plot's two axes are on a comparable scale.
    adv_unit = adv_dir / adv_norm
    rand_dir = rand_dir - (rand_dir * adv_unit).sum() * adv_unit
    rand_dir = rand_dir / rand_dir.norm() * adv_norm

    coefs = np.linspace(-args.coef_range, args.coef_range, args.resolution)
    preds, confs = classify_grid(model, x0, adv_dir, rand_dir, coefs)

    present_classes = sorted(set(preds.flatten().tolist()) | {y0, pred_adv})
    cmap = plt.get_cmap("tab20" if len(class_names) > 10 else "tab10")
    color_of = {c: cmap(c % 20) for c in range(len(class_names))}
    grid_rgb = np.zeros((args.resolution, args.resolution, 4))
    for i in range(args.resolution):
        for j in range(args.resolution):
            grid_rgb[i, j] = color_of[preds[i, j]]

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    ax.imshow(grid_rgb.transpose(1, 0, 2), origin="lower",
              extent=[-args.coef_range, args.coef_range, -args.coef_range, args.coef_range], aspect="equal")
    ax.contour(coefs, coefs, preds.T, levels=np.arange(len(class_names) + 1) - 0.5,
               colors="black", linewidths=0.6, alpha=0.6)

    ax.scatter([0], [0], s=180, marker="*", edgecolor="black", facecolor="white", linewidth=1.3,
               zorder=5, label=f"x0 (clean, true={class_names[y0]}, pred={class_names[pred0]})")
    ax.scatter([1], [0], s=110, marker="X", edgecolor="black",
               facecolor="yellow" if pred_adv != y0 else "white", linewidth=1.3, zorder=5,
               label=f"x_adv (real PGD-20, pred={class_names[pred_adv]})")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color_of[c],
                           markersize=10, label=class_names[c]) for c in present_classes]
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=min(6, len(handles)), fontsize=7.5, frameon=False)
    ax.set_xlabel("coefficient along real adversarial direction (x_adv - x0)")
    ax.set_ylabel("coefficient along a random orthogonal direction")
    ax.set_title(f"Decision boundary of the real trained model in a 2D slice of input space\n"
                 f"({args.dataset}, {args.backbone}, test image #{index}) -- every cell is a real forward pass",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = args.out or os.path.join("figures", f"decision_boundary_{args.dataset}_{args.backbone}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print("Saved", out)
    print(f"x0 true={class_names[y0]} pred={class_names[pred0]} | x_adv pred={class_names[pred_adv]} "
          f"| ||x_adv - x0||_2={adv_norm:.4f}")


if __name__ == "__main__":
    main()
