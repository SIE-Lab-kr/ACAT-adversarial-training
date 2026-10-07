"""Full evaluation of one trained cell (fp32 / TF32, no autocast).

python gpu/evaluate.py --dataset cifar10 --backbone resnet18 --method acat
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import GPUData, make_model, pgd, EPS, ALPHA

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="cifar10")
p.add_argument("--backbone", default="resnet18")
p.add_argument("--method", default="acat")
p.add_argument("--out", default="results_gpu")
p.add_argument("--surrogate", default=None, help="backbone whose *standard* model is the transfer source")
p.add_argument("--aa_n", type=int, default=1000)
p.add_argument("--square_n", type=int, default=1000)
p.add_argument("--square_queries", type=int, default=1000)
p.add_argument("--curve_n", type=int, default=1000)
p.add_argument("--bs", type=int, default=500)
a = p.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(0)

cell = os.path.join(a.out, a.dataset, a.backbone, a.method)
res_path = os.path.join(cell, "eval.json")
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
data = GPUData(a.dataset)


def load(backbone, method):
    m = make_model(data.info, backbone)
    m.load_state_dict(torch.load(os.path.join(a.out, a.dataset, backbone, method, "model_final.pt"), map_location="cuda"))
    return m.eval()


model = load(a.backbone, a.method)


def save():
    with open(res_path, "w") as f:
        json.dump(res, f, indent=1)


def run(name, fn):
    if name in res:
        return
    t0 = time.time()
    res[name] = fn()
    res.setdefault("eval_time_sec", {})[name] = time.time() - t0
    print(f"{name}: {res[name] if not isinstance(res[name], dict) else {k: v for k, v in res[name].items() if not isinstance(v, list)}}  ({time.time()-t0:.0f}s)", flush=True)
    save()


def full_pass(attack=None, n=None, keep_preds=False):
    correct, preds = 0, []
    total = 0
    for x, y in data.test_batches(a.bs, n):
        xa = x if attack is None else attack(x, y)
        with torch.no_grad():
            pr = model(xa).argmax(1)
        correct += (pr == y).sum().item(); total += len(y)
        if keep_preds:
            preds.append(pr.cpu())
    acc = correct / total
    return (acc, torch.cat(preds).tolist()) if keep_preds else acc


def fgsm(x, y):
    x = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(F.cross_entropy(model(x), y), x)
    return (x + EPS * g.sign()).clamp(0, 1).detach()


# -------- white-box on full test set --------
def _clean():
    acc, pr = full_pass(keep_preds=True)
    return {"acc": acc, "preds": pr}


def _pgd20():
    acc, pr = full_pass(lambda x, y: pgd(model, x, y, steps=20, amp=False), keep_preds=True)
    return {"acc": acc, "preds": pr}


run("clean", _clean)
run("fgsm", lambda: full_pass(fgsm))
run("pgd20", _pgd20)
run("cw_linf", lambda: full_pass(lambda x, y: pgd(model, x, y, steps=30, amp=False, loss="cw")))

# -------- AutoAttack (standard) on the first aa_n test images --------
def _aa():
    from autoattack import AutoAttack
    x = torch.cat([xb for xb, _ in data.test_batches(a.bs, a.aa_n)]).contiguous()
    y = data.yte[:a.aa_n]
    adversary = AutoAttack(model, norm="Linf", eps=EPS, version="standard", verbose=False)
    xa = adversary.run_standard_evaluation(x, y, bs=a.bs)
    with torch.no_grad():
        return (model(xa).argmax(1) == y).float().mean().item()


run("autoattack", _aa)

# -------- black-box --------
def _square():
    from autoattack.square import SquareAttack
    sq = SquareAttack(model, p_init=0.8, n_queries=a.square_queries, eps=EPS, norm="Linf", n_restarts=1, seed=0, verbose=False)
    c = t = 0
    for x, y in data.test_batches(a.bs, a.square_n):
        xa = sq.perturb(x.contiguous(), y)
        with torch.no_grad():
            c += (model(xa).argmax(1) == y).sum().item(); t += len(y)
    return c / t


run("square", _square)

if a.surrogate:
    sur = load(a.surrogate, "standard")

    def _transfer():
        return full_pass(lambda x, y: pgd(sur, x, y, steps=20, amp=False))
    run("transfer", _transfer)
    res["transfer_source"] = f"{a.surrogate}/standard"

# -------- curves on the first curve_n images --------
def _eps_curve():
    out = {}
    for e in [0, 1, 2, 4, 6, 8, 10, 12, 16]:
        eps = e / 255
        out[str(e)] = full_pass(None if e == 0 else (lambda x, y, eps=eps: pgd(model, x, y, eps=eps, alpha=max(eps / 4, 0.5 / 255), steps=20, amp=False)), n=a.curve_n)
    return out


def _steps_curve():
    out = {}
    for s in [1, 2, 5, 10, 20, 50, 100]:
        out[str(s)] = full_pass(lambda x, y, s=s: pgd(model, x, y, steps=s, amp=False), n=a.curve_n)
    return out


run("eps_curve", _eps_curve)
run("steps_curve", _steps_curve)
res["done"] = True
save()
print("eval finished", cell)
