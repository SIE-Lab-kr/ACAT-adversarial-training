"""
Training loop shared by all (dataset, method) combinations. Logs per-epoch
clean accuracy, PGD-10 robust accuracy (quick in-loop check), and loss, then
saves a JSON metrics file + a matplotlib curve figure per run.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_dataset, DATASET_INFO
from src.models import build_model, NormalizedModel
from src.methods import METHODS
from src.attacks import pgd_attack
from src.utils import set_seed, AverageMeter, EMATracker, save_json


def evaluate_quick(model, loader, device, eps, alpha, steps, max_batches=None):
    model.eval()
    correct_clean, correct_adv, total = 0, 0, 0
    for bi, (x, y, idx) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred_clean = model(x).argmax(1)
        x_adv = pgd_attack(model, x, y, eps, alpha, steps)
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(1)
        correct_clean += (pred_clean == y).sum().item()
        correct_adv += (pred_adv == y).sum().item()
        total += y.size(0)
    model.train()
    return correct_clean / max(total, 1), correct_adv / max(total, 1)


def resolve_device(device=None):
    """Auto-detect CUDA if available and not overridden. This is the single
    place device selection happens, so every training/eval script in this
    codebase automatically uses a GPU when one is present, with no separate
    'GPU mode' to remember to turn on."""
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(dataset_name, method_name, epochs, batch_size, lr, out_dir,
                  train_subset=None, test_subset=None, seed=42, width=32,
                  train_eps=8/255, train_alpha=2/255, train_steps=7,
                  backbone="smallcnn", eval_pgd_steps=10, eval_max_batches=6,
                  quiet=False, device=None, lr_schedule="cosine",
                  multistep_milestones=None, multistep_gamma=0.1,
                  save_stage_checkpoints=False, stage_fracs=(0.0, 0.5, 1.0)):
    """
    lr_schedule: "cosine" (CosineAnnealingLR over `epochs`) or "multistep"
    (MultiStepLR; defaults to milestones at 75%/90% of `epochs`, gamma=0.1,
    matching the "multistep [75, 90] gamma=0.1" schedule used in
    configs/cifar10.yaml etc. at their paper-scale epoch counts).

    save_stage_checkpoints: when True, additionally save the model's state at
    the "early", "middle", and "final" epochs given by `stage_fracs` (as
    fractions of total epochs, e.g. (0.0, 0.5, 1.0) = right after the first
    epoch, halfway through, and the last epoch) to
    out_dir/checkpoints/<stage>_epoch<N>.pt, plus a
    out_dir/checkpoints/manifest.json recording which epoch each stage
    actually landed on -- used by make_stage_examples_fig.py to show how a
    model's real predictions on real images change over the course of
    training.
    """
    set_seed(seed)
    device = resolve_device(device)
    info = DATASET_INFO[dataset_name]

    train_ds, test_ds, class_names = get_dataset(dataset_name)
    if train_subset is not None and train_subset < len(train_ds):
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(train_ds), size=train_subset, replace=False)
        train_ds = torch.utils.data.Subset(train_ds, keep)
    if test_subset is not None and test_subset < len(test_ds):
        rng = np.random.RandomState(seed + 1)
        keep = rng.choice(len(test_ds), size=test_subset, replace=False)
        test_ds = torch.utils.data.Subset(test_ds, keep)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    net = build_model(info, width=width, backbone=backbone)
    model = NormalizedModel(net, info["mean"], info["std"]).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if lr_schedule == "multistep":
        milestones = multistep_milestones or [max(1, int(0.75 * epochs)), max(1, int(0.9 * epochs))]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=multistep_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ema_tracker = EMATracker(n_samples=len(train_ds.dataset) if hasattr(train_ds, "dataset") else len(train_ds),
                              alpha=0.9)

    # Resolve which real epoch index counts as "early"/"middle"/"final" for
    # stage-checkpoint saving, e.g. (0.0, 0.5, 1.0) over 40 epochs -> [0, 19, 39].
    # "final" always pins to the true last epoch (epochs - 1); "early" and
    # "middle" are resolved from stage_fracs and nudged to stay distinct from
    # each other and from "final", so at very low epoch counts (e.g. epochs=2)
    # every requested stage name still gets a real, distinct checkpoint rather
    # than "final" silently disappearing because "middle" already claimed its
    # epoch.
    stage_epochs = {}
    if save_stage_checkpoints:
        stage_names = ["early", "middle", "final"][:len(stage_fracs)]
        last_epoch = epochs - 1
        used = set()
        resolved = []
        for name, f in zip(stage_names, stage_fracs):
            if name == "final":
                e = last_epoch
            else:
                e = min(last_epoch, int(round(f * last_epoch)))
                while e in used and e < last_epoch:
                    e += 1
                while e in used and e > 0:
                    e -= 1
            used.add(e)
            resolved.append(e)
        for name, e in zip(stage_names, resolved):
            stage_epochs[e] = name
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    method_fn = METHODS[method_name]
    cfg = dict(train_eps=train_eps, train_alpha=train_alpha, train_steps=train_steps)

    history = {"epoch": [], "train_loss": [], "clean_acc": [], "pgd10_robust_acc": [],
               "epoch_time_sec": [], "extra": []}

    t_start = time.time()
    for epoch in range(epochs):
        t_epoch0 = time.time()
        loss_meter = AverageMeter()
        step_state = {"epoch": epoch, "total_epochs": epochs}
        extra_accum = {}
        for x, y, idx in train_loader:
            x, y, idx = x.to(device), y.to(device), idx.to(device)
            optimizer.zero_grad()
            loss, logs = method_fn(model, x, y, idx, ema_tracker, step_state, cfg)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), x.size(0))
            for k, v in logs.items():
                extra_accum.setdefault(k, []).append(v)
        scheduler.step()

        clean_acc, robust_acc = evaluate_quick(model, test_loader, device,
                                                eps=8/255, alpha=2/255, steps=eval_pgd_steps,
                                                max_batches=eval_max_batches)
        epoch_time = time.time() - t_epoch0
        history["epoch"].append(epoch)
        history["train_loss"].append(loss_meter.avg)
        history["clean_acc"].append(clean_acc)
        history["pgd10_robust_acc"].append(robust_acc)
        history["epoch_time_sec"].append(epoch_time)
        history["extra"].append({k: float(np.mean(v)) for k, v in extra_accum.items()})

        if not quiet:
            print(f"[{dataset_name}/{method_name}] epoch {epoch+1}/{epochs} "
                  f"loss={loss_meter.avg:.4f} clean_acc={clean_acc:.3f} "
                  f"pgd10_acc={robust_acc:.3f} time={epoch_time:.1f}s", flush=True)

        if epoch in stage_epochs:
            stage = stage_epochs[epoch]
            ckpt_path = os.path.join(out_dir, "checkpoints", f"{stage}_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            history.setdefault("stage_checkpoints", {})[stage] = {
                "epoch": epoch, "path": ckpt_path,
                "clean_acc": clean_acc, "pgd10_robust_acc": robust_acc,
            }

    total_time = time.time() - t_start
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    history["total_time_sec"] = total_time
    history["n_train"] = len(train_ds)
    history["n_test"] = len(test_ds)
    history["backbone"] = backbone
    history["device"] = str(device)
    history["n_params"] = sum(p.numel() for p in model.parameters())
    history["class_names"] = class_names
    save_json(history, os.path.join(out_dir, "history.json"))
    if save_stage_checkpoints and "stage_checkpoints" in history:
        save_json(history["stage_checkpoints"], os.path.join(out_dir, "checkpoints", "manifest.json"))
    return model, history, class_names, test_ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_INFO.keys()))
    p.add_argument("--method", required=True, choices=list(METHODS.keys()))
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--train_subset", type=int, default=None)
    p.add_argument("--test_subset", type=int, default=None)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    run_training(args.dataset, args.method, args.epochs, args.batch_size, args.lr,
                 args.out_dir, train_subset=args.train_subset, test_subset=args.test_subset)
