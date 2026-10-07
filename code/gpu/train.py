"""Train one (dataset, backbone, method) cell on the GPU.

python gpu/train.py --dataset cifar10 --backbone resnet18 --method acat --epochs 30
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from common import GPUData, make_model, METHODS, EMAMemory, accuracy, robust_accuracy

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="cifar10")
p.add_argument("--backbone", default="resnet18")
p.add_argument("--method", default="acat")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--ema_alpha", type=float, default=0.95)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="results_gpu")
p.add_argument("--monitor_n", type=int, default=1000)
p.add_argument("--tag", default=None, help="output folder name (default: method)")
p.add_argument("--lam_mode", default="sched", choices=["sched", "zero", "one"])
p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
p.add_argument("--beta_adv", type=float, default=0.5)
p.add_argument("--h_dir", default="easy", choices=["easy", "hard"], help="hard = original submission's (1 - h^n) term")
a = p.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(a.seed); np.random.seed(a.seed)
gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed)

out = os.path.join(a.out, a.dataset, a.backbone, a.tag or a.method)
os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, "DONE")):
    print("already done:", out); raise SystemExit

data = GPUData(a.dataset)
model = make_model(data.info, a.backbone)
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd, nesterov=False)
iters_per_epoch = data.n_train // a.batch_size
total_iters = iters_per_epoch * a.epochs
warm_iters = iters_per_epoch * a.warmup


def lr_at(it):
    if it < warm_iters:
        return a.lr * (it + 1) / warm_iters
    t = (it - warm_iters) / max(total_iters - warm_iters, 1)
    return 0.5 * a.lr * (1 + math.cos(math.pi * t))


stage_epochs = sorted({max(0, round(a.epochs * 0.1) - 1), a.epochs // 2 - 1, a.epochs - 1})
st = {"total_epochs": a.epochs, "lam_mode": a.lam_mode, "schedule": a.schedule, "beta_adv": a.beta_adv, "h_dir": a.h_dir}
if a.method == "acat":
    st["ema"] = EMAMemory(data.n_train, a.ema_alpha)
loss_fn = METHODS[a.method]
hist = {"args": vars(a), "n_params": sum(p.numel() for p in model.parameters()), "epochs": []}
it = 0
t_train = 0.0
start_ep = 0
ck_path = os.path.join(out, "ckpt_last.pt")
if os.path.exists(ck_path):  # resume after a crash (e.g. GPU driver reset)
    ck = torch.load(ck_path, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    it, t_train, start_ep, hist = ck["it"], ck["t_train"], ck["epoch"] + 1, ck["hist"]
    gen.set_state(ck["gen"].cpu()); torch.set_rng_state(ck["cpu_rng"].cpu()); torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
    if a.method == "acat":
        st["ema"].v.copy_(ck["ema_v"]); st["ema"].seen.copy_(ck["ema_seen"])
    print(f"resumed from epoch {start_ep}", flush=True)
for ep in range(start_ep, a.epochs):
    st["epoch"] = ep
    st["collect"] = [] if a.method == "acat" else None
    model.train()
    torch.cuda.synchronize(); t0 = time.time()
    tot, n = 0.0, 0
    for x, y, idx in data.train_batches(a.batch_size, gen):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        loss, _ = loss_fn(model, x, y, idx, st)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.detach(); n += 1; it += 1
    torch.cuda.synchronize(); ep_time = time.time() - t0; t_train += ep_time
    rec = {"epoch": ep, "train_loss": (tot / n).item(), "epoch_time": ep_time,
           "clean_acc": accuracy(model, data, a.monitor_n),
           "pgd10_acc": robust_accuracy(model, data, a.monitor_n, steps=10)}
    if a.method == "acat":
        c = torch.cat(st["collect"]).float()
        _, h, b, bs_, w = c.T
        lam = min(max(ep / max(a.epochs - 1, 1), 0.0), 1.0)
        if a.schedule == "cosine":
            lam = 0.5 * (1 - math.cos(math.pi * lam))
        lam = {"sched": lam, "zero": 0.0, "one": 1.0}[a.lam_mode]
        rec.update({"lambda": lam,
                    "h_mean": h.mean().item(), "h_std": h.std().item(),
                    "b_mean": b.mean().item(), "b_smooth_mean": bs_.mean().item(),
                    "w_std": w.std().item(), "w_p10": w.quantile(0.1).item(), "w_p90": w.quantile(0.9).item(),
                    "corr_w_h": torch.corrcoef(torch.stack([w, h]))[0, 1].item(),
                    "corr_w_b": torch.corrcoef(torch.stack([w, bs_]))[0, 1].item()})
        if ep in stage_epochs:
            np.savez_compressed(os.path.join(out, f"acat_signals_ep{ep}.npz"), data=c.cpu().numpy())
    hist["epochs"].append(rec)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    if ep in stage_epochs:
        torch.save(model.state_dict(), os.path.join(out, f"model_ep{ep}.pt"))
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    ck = {"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "t_train": t_train, "epoch": ep,
          "hist": hist, "gen": gen.get_state(), "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
    if a.method == "acat":
        ck["ema_v"], ck["ema_seen"] = st["ema"].v, st["ema"].seen
    torch.save(ck, ck_path + ".tmp"); os.replace(ck_path + ".tmp", ck_path)

hist["train_time_sec"] = t_train
hist["stage_epochs"] = stage_epochs
torch.save(model.state_dict(), os.path.join(out, "model_final.pt"))
with open(os.path.join(out, "history.json"), "w") as f:
    json.dump(hist, f, indent=1)
open(os.path.join(out, "DONE"), "w").write("ok")
if os.path.exists(ck_path):
    os.remove(ck_path)
print("finished", out, f"train_time={t_train:.0f}s")
