"""
Full evaluation harness: white-box attacks (FGSM, PGD-20, APGD-20, CW-Linf),
black-box attacks (Square Attack - query-only, and a transfer attack from an
independently-trained surrogate), and the Multi-threat Efficiency Score (MES)
summary metric used in the paper.

MES is defined here as the harmonic mean of clean accuracy and the average
robust accuracy across every attack in the evaluation matrix -- a single
number that penalizes a defense for buying robustness at too steep a clean-
accuracy cost, or vice versa:

    MES = 2 * clean_acc * mean_robust_acc / (clean_acc + mean_robust_acc + eps)
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attacks import fgsm_attack, pgd_attack, apgd_attack, cw_linf_attack, square_attack, transfer_attack

EVAL_EPS = 8 / 255
EVAL_ALPHA = 2 / 255


@torch.no_grad()
def _acc(model, x, y):
    pred = model(x).argmax(1)
    return (pred == y).float().sum().item()


def evaluate_full(model, test_ds, device, surrogate_model=None, batch_size=32,
                   max_examples=200, pgd_steps=20, square_queries=150, cw_steps=30, seed=0):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    if surrogate_model is not None:
        surrogate_model.eval()

    results = {"n_examples": 0, "correct": {
        "clean": 0, "fgsm": 0, "pgd20": 0, "apgd20": 0, "cw_linf": 0,
        "square_bbox": 0,
    }}
    if surrogate_model is not None:
        results["correct"]["transfer_bbox"] = 0

    seen = 0
    for x, y, idx in loader:
        if seen >= max_examples:
            break
        take = min(x.size(0), max_examples - seen)
        x, y = x[:take].to(device), y[:take].to(device)

        results["correct"]["clean"] += _acc(model, x, y)
        x_fgsm = fgsm_attack(model, x, y, EVAL_EPS)
        results["correct"]["fgsm"] += _acc(model, x_fgsm, y)
        x_pgd = pgd_attack(model, x, y, EVAL_EPS, EVAL_ALPHA, pgd_steps)
        results["correct"]["pgd20"] += _acc(model, x_pgd, y)
        x_apgd = apgd_attack(model, x, y, EVAL_EPS, steps=pgd_steps)
        results["correct"]["apgd20"] += _acc(model, x_apgd, y)
        x_cw = cw_linf_attack(model, x, y, EVAL_EPS, EVAL_ALPHA, steps=cw_steps)
        results["correct"]["cw_linf"] += _acc(model, x_cw, y)
        x_sq = square_attack(model, x, y, EVAL_EPS, n_queries=square_queries)
        results["correct"]["square_bbox"] += _acc(model, x_sq, y)
        if surrogate_model is not None:
            x_tr = transfer_attack(surrogate_model, model, x, y, EVAL_EPS, EVAL_ALPHA, steps=pgd_steps)
            results["correct"]["transfer_bbox"] += _acc(model, x_tr, y)

        seen += take

    results["n_examples"] = seen
    accs = {k: v / max(seen, 1) for k, v in results["correct"].items()}
    results["accuracy"] = accs

    attack_keys = [k for k in accs if k != "clean"]
    mean_robust = float(np.mean([accs[k] for k in attack_keys])) if attack_keys else 0.0
    clean = accs["clean"]
    eps = 1e-8
    mes = 2 * clean * mean_robust / (clean + mean_robust + eps)
    results["mean_robust_acc"] = mean_robust
    results["MES"] = mes
    return results
