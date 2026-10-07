"""Resumable orchestrator: dataset x backbone x method, train then evaluate.

python gpu/run_matrix.py --dataset cifar10 --epochs 30
Re-running the same command skips finished cells (DONE marker / eval.json "done").
"""
import argparse
import json
import os
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="cifar10")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--out", default="results_gpu")
p.add_argument("--backbones", default="vgg11,resnet18,wrn28_10")
p.add_argument("--methods", default="standard,pgd_at,trades,mart,acat")
a = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
os.makedirs("logs_gpu", exist_ok=True)
backbones = a.backbones.split(",")
methods = a.methods.split(",")
# transfer-attack source: a standard-trained model of a *different* architecture
SURROGATE = {"vgg11": "resnet18", "resnet18": "vgg11", "wrn28_10": "resnet18"}


def sh(args, log):
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n==== {time.ctime()} :: {' '.join(args)}\n"); f.flush()
        r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT)
    status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {status} {' '.join(args[1:])}  {(time.time()-t0)/60:.1f} min", flush=True)
    return r.returncode == 0


def cell(bb, m):
    return os.path.join(a.out, a.dataset, bb, m)


def train(bb):
    for m in methods:
        if os.path.exists(os.path.join(cell(bb, m), "DONE")):
            print(f"skip train {bb}/{m}", flush=True); continue
        sh(["gpu/train.py", "--dataset", a.dataset, "--backbone", bb, "--method", m,
            "--epochs", str(a.epochs), "--out", a.out], f"logs_gpu/train_{a.dataset}_{bb}_{m}.log")


def evaluate(bb):
    sur = SURROGATE.get(bb)
    for m in methods:
        ej = os.path.join(cell(bb, m), "eval.json")
        if os.path.exists(ej) and json.load(open(ej)).get("done"):
            print(f"skip eval {bb}/{m}", flush=True); continue
        if not os.path.exists(os.path.join(cell(bb, m), "DONE")):
            print(f"cannot eval {bb}/{m}: not trained", flush=True); continue
        args = ["gpu/evaluate.py", "--dataset", a.dataset, "--backbone", bb, "--method", m, "--out", a.out]
        if sur and os.path.exists(os.path.join(cell(sur, "standard"), "DONE")):
            args += ["--surrogate", sur]
        sh(args, f"logs_gpu/eval_{a.dataset}_{bb}_{m}.log")


# fast backbones first (their surrogates are each other), then the big one
fast = [b for b in backbones if b != "wrn28_10"]
for bb in fast:
    train(bb)
for bb in fast:
    evaluate(bb)
if "wrn28_10" in backbones:
    train("wrn28_10")
    evaluate("wrn28_10")
print("MATRIX COMPLETE", flush=True)
