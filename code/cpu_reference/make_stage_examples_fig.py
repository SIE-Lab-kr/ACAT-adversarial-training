"""
Input -> output examples at the early / middle / final training stage (v6).

For one real trained run (a (dataset, backbone, method) cell under a
results_gpu_matrix-style directory, produced by run_full_gpu_matrix.py or
run_experiment.py with --save_stage_checkpoints), this loads the three
stage checkpoints (src/train.py: "early"/"middle"/"final", real weights
saved during that run's own training loop -- not separately retrained) and,
for a small fixed set of real held-out test images, shows:
  - the clean image with the model's predicted label at that stage
    (green title = correct, red = incorrect), and
  - the same image after a real PGD-20 attack against that stage's own
    weights, with its predicted label.

This directly answers "what does my model actually output, at each point in
training, for a real input" -- every prediction and every attacked image
comes from a real forward/backward pass through the real checkpoint for that
stage, nothing is illustrative.

Usage:
    python make_stage_examples_fig.py --run_dir results_gpu_matrix_quick/cifar10/smallcnn/acat \\
        --dataset cifar10 --backbone smallcnn --n_examples 6
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import get_dataset, DATASET_INFO
from src.models import build_model, NormalizedModel
from src.attacks import pgd_attack

EVAL_EPS = 8 / 255
EVAL_ALPHA = 2 / 255


def load_model(dataset, backbone, ckpt_path, width=32):
    info = DATASET_INFO[dataset]
    net = build_model(info, width=width, backbone=backbone)
    model = NormalizedModel(net, info["mean"], info["std"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def pick_examples(dataset, n=6, seed=11):
    _, test_ds, class_names = get_dataset(dataset)
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(test_ds), size=min(n, len(test_ds)), replace=False)
    xs, ys = [], []
    for i in idxs:
        x, y, _ = test_ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys), class_names


def to_img(x):
    arr = x.detach().cpu().numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    return np.clip(arr.transpose(1, 2, 0), 0, 1)


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="e.g. results_gpu_matrix_quick/cifar10/smallcnn/acat")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", default="smallcnn")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n_examples", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    manifest_path = os.path.join(args.run_dir, "checkpoints", "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"No stage-checkpoint manifest at {manifest_path} -- run training with "
              f"--save_stage_checkpoints first. Skipping.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)

    stages = [s for s in ["early", "middle", "final"] if s in manifest]
    x, y, class_names = pick_examples(args.dataset, n=args.n_examples)

    fig, axes = plt.subplots(2 * len(stages), args.n_examples,
                              figsize=(1.9 * args.n_examples, 2.0 * 2 * len(stages)))
    if axes.ndim == 1:
        axes = axes[None, :]

    for si, stage in enumerate(stages):
        ckpt_path = manifest[stage]["path"]
        model = load_model(args.dataset, args.backbone, ckpt_path, width=args.width)
        pred_clean = predict(model, x)
        x_adv = pgd_attack(model, x, y, EVAL_EPS, EVAL_ALPHA, steps=20)
        pred_adv = predict(model, x_adv)

        row_clean, row_adv = 2 * si, 2 * si + 1
        for col in range(args.n_examples):
            ax = axes[row_clean, col]
            ax.imshow(to_img(x[col]))
            correct = pred_clean[col].item() == y[col].item()
            ax.set_title(f"{class_names[pred_clean[col]]}", fontsize=8,
                         color="green" if correct else "red")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{stage}\nclean", fontsize=9)

            ax = axes[row_adv, col]
            ax.imshow(to_img(x_adv[col]))
            correct = pred_adv[col].item() == y[col].item()
            ax.set_title(f"{class_names[pred_adv[col]]}", fontsize=8,
                         color="green" if correct else "red")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{stage}\nunder PGD-20", fontsize=9)

    fig.suptitle(f"Real input -> real model output across training stages "
                 f"({args.dataset}, {args.backbone}, epoch = {[manifest[s]['epoch'] for s in stages]}) -- "
                 f"green title = correct prediction, red = incorrect. True labels: "
                 + ", ".join(class_names[c] for c in y.tolist()), fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = args.out or os.path.join("figures", f"stage_examples_{args.dataset}_{args.backbone}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    main()
