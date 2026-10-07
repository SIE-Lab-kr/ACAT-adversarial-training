"""
Attack suite: white-box gradient attacks, a white-box optimization attack,
and two black-box attacks (query-based Square Attack, and transfer-based).

All attacks assume the model accepts raw images in [0, 1] range (see
models.NormalizedModel) and operate under an L_infinity budget `eps`
(standard convention, eps typically expressed as a fraction of 255).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp(x):
    return torch.clamp(x, 0.0, 1.0)


# ----------------------------------------------------------------------------
# White-box attacks
# ----------------------------------------------------------------------------
def fgsm_attack(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + eps * grad.sign()
    return _clamp(x_adv).detach()


def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    x_orig = x.clone().detach()
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = _clamp(x_adv)
    else:
        x_adv = x_orig.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = _clamp(x_adv)
    return x_adv.detach()


def apgd_attack(model, x, y, eps, steps=20):
    """
    A simplified Auto-PGD (momentum + step-size halving on loss plateaus),
    in the spirit of Croce & Hein (2020). Included as a "new" white-box
    attack beyond plain PGD, per the reviewer request for stronger evaluation.
    """
    x_orig = x.clone().detach()
    alpha = eps
    momentum = 0.75
    x_adv = _clamp(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
    x_best = x_adv.clone()
    with torch.no_grad():
        best_loss = F.cross_entropy(model(x_adv), y, reduction="none")
    velocity = torch.zeros_like(x_orig)
    plateau = 0
    prev_loss_mean = best_loss.mean().item()
    for t in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            velocity = momentum * velocity + (1 - momentum) * grad.sign()
            step = x_adv + alpha * velocity.sign()
            step = torch.max(torch.min(step, x_orig + eps), x_orig - eps)
            x_adv = _clamp(step)
            cur_loss = F.cross_entropy(model(x_adv), y, reduction="none")
            improved = cur_loss > best_loss
            x_best[improved] = x_adv[improved]
            best_loss = torch.where(improved, cur_loss, best_loss)
            if cur_loss.mean().item() <= prev_loss_mean * 1.001:
                plateau += 1
            else:
                plateau = 0
            prev_loss_mean = cur_loss.mean().item()
            if plateau >= 4:
                alpha = max(alpha * 0.7, eps / 20)
                plateau = 0
    return x_best.detach()


def cw_linf_attack(model, x, y, eps, alpha, steps=30, kappa=0.0):
    """
    Carlini-Wagner-style margin loss, projected to an L_infinity ball (a
    practical CW/PGD hybrid commonly used when a pure-L2 CW attack is too
    slow for CPU-only evaluation). Maximizes (max_other_logit - true_logit).
    """
    x_orig = x.clone().detach()
    x_adv = _clamp(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
    num_classes_cache = None
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        if num_classes_cache is None:
            num_classes_cache = logits.shape[1]
        onehot = F.one_hot(y, num_classes_cache).float()
        true_logit = (logits * onehot).sum(1)
        other_logit = (logits - onehot * 1e9).max(1)[0]
        loss = torch.clamp(true_logit - other_logit + kappa, min=-50).sum()
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()  # minimize (true - other)
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = _clamp(x_adv)
    return x_adv.detach()


# ----------------------------------------------------------------------------
# Black-box attacks
# ----------------------------------------------------------------------------
@torch.no_grad()
def square_attack(model, x, y, eps, n_queries=200, p_init=0.2):
    """
    Query-based, gradient-free black-box attack (Andriushchenko et al. 2020,
    simplified single-channel-block version). Only ever queries model(x);
    never uses gradients, so it is a genuine black-box attack even against a
    model whose gradients are masked/obfuscated.
    """
    device = x.device
    x_orig = x.clone()
    B, C, H, W = x.shape
    x_adv = _clamp(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
    logits = model(x_adv)
    loss_best = F.cross_entropy(logits, y, reduction="none")

    for q in range(n_queries):
        p = max(p_init * (1 - q / n_queries), 0.05)
        side = max(int(round((p * H * W) ** 0.5)), 1)
        r = torch.randint(0, max(H - side, 1) + 1, (1,)).item()
        c = torch.randint(0, max(W - side, 1) + 1, (1,)).item()
        delta = (torch.randint(0, 2, (B, C, 1, 1), device=device).float() * 2 - 1) * eps
        x_try = x_adv.clone()
        x_try[:, :, r:r + side, c:c + side] = torch.clamp(
            x_orig[:, :, r:r + side, c:c + side] + delta, 0.0, 1.0)
        x_try = torch.max(torch.min(x_try, x_orig + eps), x_orig - eps)
        x_try = _clamp(x_try)
        loss_try = F.cross_entropy(model(x_try), y, reduction="none")
        improved = loss_try > loss_best
        x_adv[improved] = x_try[improved]
        loss_best = torch.where(improved, loss_try, loss_best)
    return x_adv


def transfer_attack(surrogate_model, target_model_unused, x, y, eps, alpha, steps=10):
    """
    Transfer-based black-box attack: craft a PGD adversarial example against
    an independently-trained SURROGATE model (never touching the target
    model's gradients), then evaluate that fixed perturbation against the
    target. `target_model_unused` is accepted only for call-signature
    symmetry with other attacks and is never queried here.
    """
    return pgd_attack(surrogate_model, x, y, eps, alpha, steps)


ATTACKS = {
    "fgsm": lambda model, x, y, cfg: fgsm_attack(model, x, y, cfg["eps"]),
    "pgd": lambda model, x, y, cfg: pgd_attack(model, x, y, cfg["eps"], cfg["alpha"], cfg["steps"]),
    "apgd": lambda model, x, y, cfg: apgd_attack(model, x, y, cfg["eps"], cfg["steps"]),
    "cw_linf": lambda model, x, y, cfg: cw_linf_attack(model, x, y, cfg["eps"], cfg["alpha"], cfg["steps"]),
    "square": lambda model, x, y, cfg: square_attack(model, x, y, cfg["eps"], cfg.get("n_queries", 200)),
}
