"""Master resumable queue for the GPU study (run once; re-running skips finished work).

Order: CIFAR-10 fast backbones -> ResNet-18 ACAT ablations -> other datasets (ResNet-18)
       -> CIFAR-10 WideResNet-28-10.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))
OUT, EPOCHS = "results_gpu", 30
METHODS = ["standard", "pgd_at", "trades", "mart", "acat"]
os.makedirs("logs_gpu", exist_ok=True)
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "torch")
AVAILABLE = {
    "cifar10": lambda: True,
    "cifar100": lambda: os.path.exists(os.path.join(CACHE, "cifar-100-python", "test")),
    "fashionmnist": lambda: os.path.exists(os.path.join(CACHE, "FashionMNIST", "raw", "t10k-labels-idx1-ubyte")),
    "tinyimagenet": lambda: os.path.exists(os.path.join(CACHE, "tiny-imagenet-200", "val", "val_annotations.txt")),
}

ABLATIONS = {  # folder -> extra train args (ResNet-18, CIFAR-10)
    "acat_adv_only": ["--beta_adv", "1.0"],
    "acat_hardness_only": ["--lam_mode", "zero"],
    "acat_boundary_only": ["--lam_mode", "one"],
    "acat_no_ema": ["--ema_alpha", "0.0"],
    "acat_cosine": ["--schedule", "cosine"],
    "acat_orig_hardfirst": ["--h_dir", "hard"],
    "acat_orig_hardfirst_adv_only": ["--h_dir", "hard", "--beta_adv", "1.0"],
}


def sh(args, log):
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n==== {time.ctime()} :: {' '.join(args)}\n"); f.flush()
        r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%m-%d %H:%M')}] {'ok' if r.returncode == 0 else 'FAILED'} {' '.join(args[1:9])}  {(time.time()-t0)/60:.1f} min", flush=True)


def cell(ds, bb, folder):
    return os.path.join(OUT, ds, bb, folder)


def train(ds, bb, folder, method, extra=()):
    for attempt in range(3):  # retries resume from ckpt_last.pt (e.g. after a GPU driver reset)
        if os.path.exists(os.path.join(cell(ds, bb, folder), "DONE")):
            return
        sh(["gpu/train.py", "--dataset", ds, "--backbone", bb, "--method", method, "--tag", folder,
            "--epochs", str(EPOCHS), "--out", OUT, *extra], f"logs_gpu/train_{ds}_{bb}_{folder}.log")


def evaluate(ds, bb, folder, surrogate):
    ej = os.path.join(cell(ds, bb, folder), "eval.json")
    if os.path.exists(ej) and json.load(open(ej)).get("done"):
        return
    if not os.path.exists(os.path.join(cell(ds, bb, folder), "DONE")):
        return
    args = ["gpu/evaluate.py", "--dataset", ds, "--backbone", bb, "--method", folder, "--out", OUT]
    if surrogate and os.path.exists(os.path.join(cell(ds, surrogate, "standard"), "DONE")):
        args += ["--surrogate", surrogate]
    sh(args, f"logs_gpu/eval_{ds}_{bb}_{folder}.log")


def full_backbone(ds, bb, surrogate):
    for m in METHODS:
        train(ds, bb, m, m)
    for m in METHODS:
        evaluate(ds, bb, m, surrogate)


# 1) CIFAR-10, VGG-11 and ResNet-18 (each is the other's transfer source)
for bb in ["vgg11", "resnet18"]:
    for m in METHODS:
        train("cifar10", bb, m, m)
full_backbone("cifar10", "vgg11", "resnet18")
full_backbone("cifar10", "resnet18", "vgg11")

# 2) ACAT ablations on ResNet-18 / CIFAR-10
for folder, extra in ABLATIONS.items():
    train("cifar10", "resnet18", folder, "acat", extra)
    evaluate("cifar10", "resnet18", folder, "vgg11")

# 3) other datasets with ResNet-18 (+ a standard VGG-11 surrogate for the transfer attack)
pending = []
for ds in ["cifar100", "fashionmnist", "tinyimagenet"]:
    if not AVAILABLE[ds]():
        pending.append(ds); print(f"{ds} not downloaded yet -> deferred", flush=True); continue
    train(ds, "vgg11", "standard", "standard")
    full_backbone(ds, "resnet18", "vgg11")

# 4) CIFAR-10 WideResNet-28-10
full_backbone("cifar10", "wrn28_10", "resnet18")

# 5) datasets whose download finished while the queue was running
for ds in pending:
    if AVAILABLE[ds]():
        train(ds, "vgg11", "standard", "standard")
        full_backbone(ds, "resnet18", "vgg11")
    else:
        print(f"{ds} still not available", flush=True)
print("ALL COMPLETE", flush=True)
