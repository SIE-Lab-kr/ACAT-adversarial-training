"""
Full-scale single-run CLI (v6: GPU-ready). Use this on a GPU machine to
reproduce/extend the paper's experimental matrix at full protocol scale
(configs/<dataset>.yaml settings) -- this CPU-only sandboxed session can only
validate that every code path here runs correctly at a small reduced scale,
never execute the real thing at paper scale (see EXECUTION_LOG_V6.md).

Device selection is automatic (src.train.resolve_device): this script runs on
GPU if one is visible to PyTorch, CPU otherwise -- no separate "GPU mode"
flag needed. All 5 backbones implemented in src/models.py (smallcnn,
wrn28_10, resnet18, vgg11, mobilenet) are available via --backbone; use
configs/<dataset>.yaml to see the backbone/epoch/lr/eps/schedule this
codebase's paper protocol specifies for a given dataset.

Example (full CIFAR-10 protocol, WideResNet-28-10, ACAT, on a GPU):
    python run_experiment.py --dataset cifar10 --method acat --backbone wrn28_10 \
        --epochs 100 --batch_size 128 --lr 0.1 --lr_schedule multistep \
        --out_dir results_gpu/cifar10/acat_wrn28_10 --save_stage_checkpoints
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train import run_training, resolve_device
from src.evaluate import evaluate_full
from src.data import DATASET_INFO
from src.methods import METHODS
from src.models import BACKBONES
import torch
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_INFO.keys()))
    p.add_argument("--method", required=True, choices=list(METHODS.keys()))
    p.add_argument("--backbone", default="smallcnn", choices=BACKBONES,
                    help="see configs/<dataset>.yaml for the backbone the paper protocol uses per dataset")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr_schedule", default="cosine", choices=["cosine", "multistep"])
    p.add_argument("--train_eps", type=float, default=8 / 255)
    p.add_argument("--train_alpha", type=float, default=2 / 255)
    p.add_argument("--train_steps", type=int, default=10, help="PGD steps used inside adversarial training")
    p.add_argument("--train_subset", type=int, default=None, help="omit for the full training set")
    p.add_argument("--test_subset", type=int, default=None, help="omit for the full test set")
    p.add_argument("--width", type=int, default=32, help="channel-width multiplier for the chosen backbone")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--eval_max_examples", type=int, default=1000)
    p.add_argument("--eval_pgd_steps", type=int, default=20)
    p.add_argument("--eval_cw_steps", type=int, default=30)
    p.add_argument("--eval_square_queries", type=int, default=500)
    p.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu; omit to auto-detect")
    p.add_argument("--save_stage_checkpoints", action="store_true",
                    help="also save early/middle/final-epoch checkpoints for make_stage_examples_fig.py")
    p.add_argument("--surrogate_dir", default=None,
                    help="path to a previously trained 'standard' model dir, used for the "
                         "black-box transfer attack surrogate; defaults to this run's own model "
                         "(a weaker, same-model transfer check) if not given")
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    model, history, class_names, test_ds = run_training(
        args.dataset, args.method, args.epochs, args.batch_size, args.lr, args.out_dir,
        train_subset=args.train_subset, test_subset=args.test_subset, width=args.width,
        train_eps=args.train_eps, train_alpha=args.train_alpha, train_steps=args.train_steps,
        backbone=args.backbone, seed=args.seed, device=device, lr_schedule=args.lr_schedule,
        save_stage_checkpoints=args.save_stage_checkpoints)

    surrogate = model
    if args.surrogate_dir is not None:
        from src.models import build_model, NormalizedModel
        info = DATASET_INFO[args.dataset]
        backbone_net = build_model(info, width=args.width, backbone=args.backbone)
        surrogate = NormalizedModel(backbone_net, info["mean"], info["std"])
        surrogate.load_state_dict(torch.load(os.path.join(args.surrogate_dir, "model.pt"), map_location=device))
        surrogate.to(device)
        surrogate.eval()

    model.to(device)
    surrogate.to(device)
    res = evaluate_full(model, test_ds, device, surrogate_model=surrogate,
                         max_examples=args.eval_max_examples, pgd_steps=args.eval_pgd_steps,
                         square_queries=args.eval_square_queries, cw_steps=args.eval_cw_steps)
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
