"""
GPU pipeline for the full-scale backbone study (CIFAR-10 first; other datasets pluggable).

- Whole training set lives on the GPU as uint8; random-crop(4) + h-flip done on GPU.
- Training uses bf16 autocast; evaluation uses fp32 (TF32 matmuls) so attacks are not weakened.
- The five training objectives mirror src/methods.py (the reference implementation used for
  the paper's supplementary study); ACAT's EMA memory is a GPU tensor instead of a CPU tensor.
"""
import math
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import build_model, NormalizedModel  # noqa: E402
from src.data import DATASET_INFO  # noqa: E402

DEV = "cuda"
EPS, ALPHA, TRAIN_STEPS = 8 / 255, 2 / 255, 10
TORCH_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "torch")


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def _load_cifar_pickles(folder, files, label_key):
    xs, ys = [], []
    for f in files:
        with open(os.path.join(folder, f), "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        xs.append(d["data"].reshape(-1, 3, 32, 32))
        ys.append(np.array(d[label_key]))
    return np.concatenate(xs), np.concatenate(ys)


def load_dataset(name):
    """Returns uint8 NCHW numpy arrays (x_train, y_train, x_test, y_test)."""
    if name == "cifar10":
        folder = os.path.join(TORCH_CACHE, "cifar-10-batches-py")
        xtr, ytr = _load_cifar_pickles(folder, [f"data_batch_{i}" for i in range(1, 6)], "labels")
        xte, yte = _load_cifar_pickles(folder, ["test_batch"], "labels")
        return xtr, ytr, xte, yte
    if name == "cifar100":
        folder = os.path.join(TORCH_CACHE, "cifar-100-python")
        xtr, ytr = _load_cifar_pickles(folder, ["train"], "fine_labels")
        xte, yte = _load_cifar_pickles(folder, ["test"], "fine_labels")
        return xtr, ytr, xte, yte
    if name == "fashionmnist":
        import torchvision
        tr = torchvision.datasets.FashionMNIST(TORCH_CACHE, train=True, download=False)
        te = torchvision.datasets.FashionMNIST(TORCH_CACHE, train=False, download=False)
        return (tr.data.numpy()[:, None], tr.targets.numpy(), te.data.numpy()[:, None], te.targets.numpy())
    if name == "tinyimagenet":
        return _load_tiny15()
    raise ValueError(f"dataset {name} not available locally")


def _load_tiny15(n_classes=15):
    from PIL import Image
    root = os.path.join(TORCH_CACHE, "tiny-imagenet-200")
    cache = os.path.join(TORCH_CACHE, f"tiny{n_classes}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    wnids = [l.strip() for l in open(os.path.join(root, "wnids.txt")) if l.strip()][:n_classes]
    xtr, ytr = [], []
    for c, w in enumerate(wnids):
        folder = os.path.join(root, "train", w, "images")
        for f in sorted(os.listdir(folder)):
            xtr.append(np.array(Image.open(os.path.join(folder, f)).convert("RGB"))); ytr.append(c)
    xte, yte = [], []
    for line in open(os.path.join(root, "val", "val_annotations.txt")):
        f, w = line.split("	")[:2]
        if w in wnids:
            xte.append(np.array(Image.open(os.path.join(root, "val", "images", f)).convert("RGB"))); yte.append(wnids.index(w))
    xtr = np.stack(xtr).transpose(0, 3, 1, 2); xte = np.stack(xte).transpose(0, 3, 1, 2)
    ytr, yte = np.array(ytr), np.array(yte)
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte, wnids=np.array(wnids))
    return xtr, ytr, xte, yte


class GPUData:
    def __init__(self, name, device=DEV):
        xtr, ytr, xte, yte = load_dataset(name)
        self.info = DATASET_INFO[name]
        self.xtr = torch.from_numpy(xtr).to(device)
        self.ytr = torch.from_numpy(ytr).long().to(device)
        self.xte = torch.from_numpy(xte).to(device)
        self.yte = torch.from_numpy(yte).long().to(device)
        self.n_train, self.n_test = len(self.xtr), len(self.xte)
        self.pad = 4 if self.info["img_size"] >= 32 else 2

    def train_batches(self, batch_size, gen):
        perm = torch.randperm(self.n_train, device=self.xtr.device, generator=gen)
        for s in range(0, self.n_train - batch_size + 1, batch_size):  # drop last
            idx = perm[s:s + batch_size]
            x = self.xtr[idx].float() / 255.0
            yield self._augment(x, gen), self.ytr[idx], idx

    def _augment(self, x, gen):
        n, c, h, w = x.shape
        p = self.pad
        xp = F.pad(x, (p, p, p, p), mode="reflect")
        i = torch.randint(0, 2 * p + 1, (n,), device=x.device, generator=gen)
        j = torch.randint(0, 2 * p + 1, (n,), device=x.device, generator=gen)
        rows = (i[:, None] + torch.arange(h, device=x.device)[None])  # n,h
        cols = (j[:, None] + torch.arange(w, device=x.device)[None])  # n,w
        x = xp[torch.arange(n, device=x.device)[:, None, None, None],
               torch.arange(c, device=x.device)[None, :, None, None],
               rows[:, None, :, None], cols[:, None, None, :]]
        flip = torch.rand(n, device=x.device, generator=gen) < 0.5
        x[flip] = x[flip].flip(3)
        return x.contiguous(memory_format=torch.channels_last)

    def test_batches(self, batch_size, n=None):
        n = self.n_test if n is None else min(n, self.n_test)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            yield (self.xte[s:e].float() / 255.0).contiguous(memory_format=torch.channels_last), self.yte[s:e]


def make_model(info, backbone, device=DEV):
    m = NormalizedModel(build_model(info, backbone=backbone), info["mean"], info["std"])
    return m.to(device).to(memory_format=torch.channels_last)


# --------------------------------------------------------------------------------------------
# Attacks used inside training (bf16 autocast for speed)
# --------------------------------------------------------------------------------------------
def _fwd(model, x, amp):
    if amp:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x).float()
    return model(x)


def pgd(model, x, y, eps=EPS, alpha=ALPHA, steps=TRAIN_STEPS, amp=True, loss="ce", random_start=True):
    x0 = x.detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1) if random_start else x0.clone()
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = _fwd(model, xa, amp)
        if loss == "ce":
            l = F.cross_entropy(logits, y)
        else:  # C&W margin loss (maximize max-other minus true)
            onehot = F.one_hot(y, logits.size(1)).bool()
            true = logits[onehot]
            other = logits.masked_fill(onehot, -1e4).max(1)[0]
            l = (other - true).clamp(max=50).sum()
        g, = torch.autograd.grad(l, xa)
        xa = torch.min(torch.max(xa.detach() + alpha * g.sign(), x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# --------------------------------------------------------------------------------------------
# Training objectives (mirroring src/methods.py)
# --------------------------------------------------------------------------------------------
class EMAMemory:
    """Per-sample EMA of boundary proximity, kept on the GPU; first visit seeds the value."""
    def __init__(self, n, alpha, device=DEV):
        self.alpha = alpha
        self.v = torch.zeros(n, device=device)
        self.seen = torch.zeros(n, dtype=torch.bool, device=device)

    def update(self, idx, val):
        prev = self.v[idx]
        new = torch.where(self.seen[idx], self.alpha * prev + (1 - self.alpha) * val, val)
        self.v[idx] = new
        self.seen[idx] = True
        return new


def _kl(p_target, logits_input):
    """Per-sample KL(p_target || softmax(logits_input)), numerically safe (as in the MART/TRADES code)."""
    q = F.softmax(logits_input, 1)
    return (p_target * (torch.log(p_target + 1e-12) - torch.log(q + 1e-12))).sum(1)


def loss_standard(model, x, y, idx, st):
    return F.cross_entropy(_fwd(model, x, True), y), {}


def loss_pgd_at(model, x, y, idx, st):
    model.eval(); xa = pgd(model, x, y); model.train()
    return F.cross_entropy(_fwd(model, xa, True), y), {}


def loss_trades(model, x, y, idx, st, beta=6.0):
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(_fwd(model, x, True), 1)
    xa = (x + 0.001 * torch.randn_like(x)).clamp(0, 1).detach()
    for _ in range(TRAIN_STEPS):
        xa.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(_fwd(model, xa, True), 1), p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, xa)
        xa = torch.min(torch.max(xa.detach() + ALPHA * g.sign(), x - EPS), x + EPS).clamp(0, 1)
    model.train()
    lc, la = _fwd(model, x, True), _fwd(model, xa, True)
    loss = F.cross_entropy(lc, y) + beta * _kl(F.softmax(lc, 1), la).mean()
    return loss, {}


def loss_mart(model, x, y, idx, st, beta=5.0):
    model.eval(); xa = pgd(model, x, y); model.train()
    la, lc = _fwd(model, xa, True), _fwd(model, x, True)
    pa = F.softmax(la, 1)
    top2 = pa.topk(2, 1)
    true_p = pa.gather(1, y[:, None]).squeeze(1)
    other_p = torch.where(top2.indices[:, 0] == y, top2.values[:, 1], top2.values[:, 0])
    bce = -torch.log(true_p + 1e-12) - torch.log(1 - other_p + 1e-12)
    pc = F.softmax(lc, 1)
    tpc = pc.gather(1, y[:, None]).squeeze(1).clamp(1e-6, 1 - 1e-6)
    kl = _kl(pc, la)
    return bce.mean() + beta * ((1 - tpc) * kl).mean(), {}


def loss_acat(model, x, y, idx, st, eps_num=1e-8):
    """ACAT (Eqs. 1-8 of the paper). st holds epoch, total_epochs, ema, and a stats collector."""
    model.eval()
    with torch.no_grad():
        p = F.softmax(_fwd(model, x, True), 1)
        true_p = p.gather(1, y[:, None]).squeeze(1)
        other = p.scatter(1, y[:, None], -1.0).max(1)[0]
        h = true_p - other                                             # Eq. 1
    xa = pgd(model, x, y)
    model.train()
    beta_adv = st.get("beta_adv", 0.5)
    b = 1.0 - h.abs() / (h.abs().max().clamp_min(eps_num) + eps_num)   # Eq. 3
    b_s = st["ema"].update(idx, b)                                       # Eq. 4
    hn = (h - h.min()) / (h.max() - h.min() + eps_num)                   # Eq. 2
    if st.get("h_dir", "easy") == "hard":                                # original-submission form: (1 - h^n)
        hn = 1.0 - hn
    bn = (b_s - b_s.min()) / (b_s.max() - b_s.min() + eps_num)           # Eq. 5
    lam = min(max(st["epoch"] / max(st["total_epochs"] - 1, 1), 0.0), 1.0)  # Eq. 6
    if st.get("schedule", "linear") == "cosine":
        lam = 0.5 * (1 - math.cos(math.pi * lam))
    lam = {"sched": lam, "zero": 0.0, "one": 1.0}[st.get("lam_mode", "sched")]  # ablations
    w = (1 - lam) * hn + lam * bn                                        # Eq. 7
    w = (w / (w.mean() + eps_num)).detach()
    la, lc = _fwd(model, xa, True), _fwd(model, x, True)
    per = (1 - beta_adv) * F.cross_entropy(lc, y, reduction="none") + beta_adv * F.cross_entropy(la, y, reduction="none")
    loss = (w * per).mean()                                              # Eq. 8
    if st.get("collect") is not None:
        st["collect"].append(torch.stack([idx.float(), h, b, b_s, w], 1))
    return loss, {"lambda": lam, "w_std": w.std().item() if st.get("log_w") else None}


METHODS = {"standard": loss_standard, "pgd_at": loss_pgd_at, "trades": loss_trades,
           "mart": loss_mart, "acat": loss_acat}


# --------------------------------------------------------------------------------------------
# Quick in-training monitor (fp32)
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def accuracy(model, data, n=1000, bs=500):
    model.eval(); c = 0; t = 0
    for x, y in data.test_batches(bs, n):
        c += (model(x).argmax(1) == y).sum().item(); t += len(y)
    return c / t


def robust_accuracy(model, data, n=1000, bs=500, steps=10, loss="ce", eps=EPS, alpha=ALPHA):
    model.eval(); c = 0; t = 0
    for x, y in data.test_batches(bs, n):
        with torch.enable_grad():
            xa = pgd(model, x, y, eps=eps, alpha=alpha, steps=steps, amp=False, loss=loss)
        with torch.no_grad():
            c += (model(xa).argmax(1) == y).sum().item(); t += len(y)
    return c / t
