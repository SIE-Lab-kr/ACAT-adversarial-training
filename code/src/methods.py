"""
Training objectives: Standard (clean, no defense), PGD-AT (Madry et al. 2018),
TRADES (Zhang et al. 2019), MART (Wang et al. 2019), and ACAT (proposed).

Every method exposes the same call signature so train.py can dispatch on a
string name:

    loss, logs = method_fn(model, x, y, idx, ema_tracker, step_state, cfg)

`step_state` carries the adaptive-curriculum progress (current epoch /
total epochs) that ACAT's lambda_t schedule needs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attacks import pgd_attack


def _pgd_train_attack(model, x, y, cfg):
    return pgd_attack(model, x, y, cfg["train_eps"], cfg["train_alpha"], cfg["train_steps"])


# ----------------------------------------------------------------------------
# Standard (clean) training -- no adversarial term. Baseline for clean acc.
# ----------------------------------------------------------------------------
def standard_loss(model, x, y, idx, ema_tracker, step_state, cfg):
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    return loss, {}


# ----------------------------------------------------------------------------
# PGD-AT (Madry et al., 2018): plain min-max adversarial training.
# ----------------------------------------------------------------------------
def pgd_at_loss(model, x, y, idx, ema_tracker, step_state, cfg):
    model.eval()
    x_adv = _pgd_train_attack(model, x, y, cfg)
    model.train()
    logits_adv = model(x_adv)
    loss = F.cross_entropy(logits_adv, y)
    return loss, {}


# ----------------------------------------------------------------------------
# TRADES (Zhang et al., 2019): clean CE + beta * KL(adv || clean).
# ----------------------------------------------------------------------------
def trades_loss(model, x, y, idx, ema_tracker, step_state, cfg):
    beta = cfg.get("trades_beta", 6.0)
    model.eval()
    with torch.no_grad():
        logits_clean_detached = model(x)
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, 0, 1)
    for _ in range(cfg["train_steps"]):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                       F.softmax(logits_clean_detached, dim=1), reduction="batchmean")
        grad = torch.autograd.grad(kl, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + cfg["train_alpha"] * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + cfg["train_eps"]), x - cfg["train_eps"])
            x_adv = torch.clamp(x_adv, 0, 1).detach()
    model.train()
    logits_clean = model(x)
    logits_adv = model(x_adv)
    loss_ce = F.cross_entropy(logits_clean, y)
    loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
                         F.softmax(logits_clean, dim=1), reduction="batchmean")
    loss = loss_ce + beta * loss_kl
    return loss, {}


# ----------------------------------------------------------------------------
# MART (Wang et al., 2019): BCE-margin term + explicit misclassification weight.
# ----------------------------------------------------------------------------
def mart_loss(model, x, y, idx, ema_tracker, step_state, cfg):
    beta = cfg.get("mart_beta", 5.0)
    model.eval()
    x_adv = _pgd_train_attack(model, x, y, cfg)
    model.train()
    logits_adv = model(x_adv)
    logits_clean = model(x)

    probs_adv = F.softmax(logits_adv, dim=1)
    top2 = probs_adv.topk(2, dim=1)
    true_prob = probs_adv.gather(1, y.unsqueeze(1)).squeeze(1)
    other_prob = torch.where(top2.indices[:, 0] == y, top2.values[:, 1], top2.values[:, 0])
    bce = -torch.log(true_prob + 1e-12) - torch.log(1 - other_prob + 1e-12)

    probs_clean = F.softmax(logits_clean, dim=1)
    true_prob_clean = probs_clean.gather(1, y.unsqueeze(1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
    kl_elem = F.kl_div(F.log_softmax(logits_adv, dim=1), probs_clean, reduction="none").sum(1)
    weight = 1.0 - true_prob_clean

    loss = bce.mean() + beta * (weight * kl_elem).mean()
    return loss, {}


# ----------------------------------------------------------------------------
# ACAT (proposed): margin-based hardness + boundary proximity, EMA-smoothed,
# blended by an adaptive curriculum schedule lambda_t, applied as a per-sample
# weight on the (clean + adversarial) loss.
# ----------------------------------------------------------------------------
def acat_lambda_schedule(epoch, total_epochs, schedule="linear"):
    """lambda_t in [0,1]: 0 => fully hardness-focused, 1 => fully boundary-focused."""
    t = min(max(epoch / max(total_epochs - 1, 1), 0.0), 1.0)
    if schedule == "linear":
        return t
    if schedule == "cosine":
        import math
        return 0.5 * (1 - math.cos(math.pi * t))
    return t


def acat_loss(model, x, y, idx, ema_tracker, step_state, cfg):
    model.eval()
    with torch.no_grad():
        logits_clean = model(x)
        probs = F.softmax(logits_clean, dim=1)
        true_prob = probs.gather(1, y.unsqueeze(1)).squeeze(1)
        other_prob = probs.clone()
        other_prob.scatter_(1, y.unsqueeze(1), -1.0)
        max_other_prob = other_prob.max(1)[0]
        h_i = true_prob - max_other_prob  # margin-based hardness, in [-1, 1]

    x_adv = _pgd_train_attack(model, x, y, cfg)
    model.train()

    eps = 1e-8
    max_abs_h = h_i.abs().max().clamp_min(eps)
    b_i = 1.0 - (h_i.abs() / (max_abs_h + eps))  # boundary proximity, in [0, 1]

    # EMA-smooth the *boundary proximity* signal (the temporal memory term
    # in the pipeline: d_i^(t) = alpha d_i^(t-1) + (1-alpha) d_tilde_i^(t)).
    ema_tracker.update(idx, b_i)
    b_i_smoothed = ema_tracker.get(idx).to(x.device)

    # normalize both signals to comparable [0,1] ranges for blending
    h_norm = (h_i - h_i.min()) / (h_i.max() - h_i.min() + eps)
    b_norm = (b_i_smoothed - b_i_smoothed.min()) / (b_i_smoothed.max() - b_i_smoothed.min() + eps)

    lam = acat_lambda_schedule(step_state["epoch"], step_state["total_epochs"],
                                 cfg.get("acat_schedule", "linear"))
    w_i = (1 - lam) * h_norm + lam * b_norm
    w_i = w_i / (w_i.mean() + eps)  # keep the effective loss scale stable

    logits_adv = model(x_adv)
    logits_clean_train = model(x)
    per_sample_ce_adv = F.cross_entropy(logits_adv, y, reduction="none")
    per_sample_ce_clean = F.cross_entropy(logits_clean_train, y, reduction="none")
    per_sample = cfg.get("acat_clean_weight", 0.5) * per_sample_ce_clean + \
                 (1 - cfg.get("acat_clean_weight", 0.5)) * per_sample_ce_adv

    loss = (w_i.detach() * per_sample).mean()
    logs = {
        "h_i_mean": h_i.mean().item(),
        "b_i_mean": b_i_smoothed.mean().item(),
        "lambda_t": lam,
        "w_i_mean": w_i.mean().item(),
        "w_i_std": w_i.std().item(),
    }
    return loss, logs


METHODS = {
    "standard": standard_loss,
    "pgd_at": pgd_at_loss,
    "trades": trades_loss,
    "mart": mart_loss,
    "acat": acat_loss,
}
