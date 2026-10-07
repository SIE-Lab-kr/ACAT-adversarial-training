"""
Orchestrates the full REAL-EXECUTED experimental matrix:
    3 datasets (CIFAR-10, Fashion-MNIST, Tiny-ImageNet-200 subset)
  x 5 methods  (Standard, PGD-AT, TRADES, MART, ACAT)
  x training-with-logging + full white-box/black-box evaluation.

Everything here runs on CPU with small, explicitly-labeled subsets so it
completes inside this sandboxed session. See EXECUTION_LOG.md for exact
sample counts / epoch counts / wall-clock time, and configs/*.yaml plus
README.md for how to rerun the same code at full paper scale on a GPU.
"""
import os
import sys
import json
import time
import traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train import run_training, resolve_device
from src.evaluate import evaluate_full
from src.data import DATASET_INFO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

METHODS = ["standard", "pgd_at", "trades", "mart", "acat"]

DATASET_RUN_CONFIG = {
    "cifar10":      dict(train_subset=640, test_subset=200, epochs=20, batch_size=32, lr=0.01, train_steps=7, seeds=[42, 123]),
    "fashionmnist": dict(train_subset=800, test_subset=200, epochs=18, batch_size=32, lr=0.01, train_steps=7, seeds=[42, 123]),
    "tinyimagenet": dict(train_subset=450, test_subset=100, epochs=10, batch_size=32, lr=0.01, train_steps=5, seeds=[42]),
}

EVAL_CONFIG = dict(max_examples=150, pgd_steps=15, square_queries=100)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    master_log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    device = resolve_device()  # auto: GPU if visible, else CPU

    for dataset in DATASET_RUN_CONFIG:
        cfg = DATASET_RUN_CONFIG[dataset]
        dataset_dir = os.path.join(RESULTS_DIR, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        for seed in cfg["seeds"]:
            trained = {}
            test_ds_ref = None

            for method in METHODS:
                out_dir = os.path.join(dataset_dir, method, f"seed{seed}")
                print(f"\n=== TRAIN {dataset} / {method} / seed={seed} ===", flush=True)
                t0 = time.time()
                try:
                    model, history, class_names, test_ds = run_training(
                        dataset, method,
                        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                        out_dir=out_dir, train_subset=cfg["train_subset"], test_subset=cfg["test_subset"],
                        train_steps=cfg["train_steps"], seed=seed)
                    elapsed = time.time() - t0
                    trained[method] = model
                    test_ds_ref = test_ds
                    master_log["runs"].append({
                        "dataset": dataset, "method": method, "seed": seed, "phase": "train",
                        "elapsed_sec": elapsed, "status": "ok",
                        "final_clean_acc": history["clean_acc"][-1],
                        "final_pgd10_acc": history["pgd10_robust_acc"][-1],
                    })
                    print(f"    done in {elapsed:.1f}s  clean={history['clean_acc'][-1]:.3f} "
                          f"pgd10={history['pgd10_robust_acc'][-1]:.3f}", flush=True)
                except Exception as e:
                    print(f"    FAILED: {e}", flush=True)
                    traceback.print_exc()
                    master_log["runs"].append({"dataset": dataset, "method": method, "seed": seed,
                                                "phase": "train", "status": "failed", "error": str(e)})

            # -- Evaluation: use the 'standard' model (if trained) as an
            # independent surrogate for the black-box transfer attack.
            surrogate = trained.get("standard", None)
            for method, model in trained.items():
                print(f"\n=== EVAL {dataset} / {method} / seed={seed} ===", flush=True)
                t0 = time.time()
                try:
                    res = evaluate_full(model, test_ds_ref, device, surrogate_model=surrogate,
                                         max_examples=EVAL_CONFIG["max_examples"],
                                         pgd_steps=EVAL_CONFIG["pgd_steps"],
                                         square_queries=EVAL_CONFIG["square_queries"])
                    elapsed = time.time() - t0
                    out_path = os.path.join(dataset_dir, method, f"seed{seed}", "eval_results.json")
                    with open(out_path, "w") as f:
                        json.dump(res, f, indent=2)
                    master_log["runs"].append({"dataset": dataset, "method": method, "seed": seed,
                                                "phase": "eval", "elapsed_sec": elapsed, "status": "ok",
                                                "MES": res["MES"], "accuracy": res["accuracy"]})
                    print(f"    done in {elapsed:.1f}s  MES={res['MES']:.3f}  acc={res['accuracy']}", flush=True)
                except Exception as e:
                    print(f"    FAILED: {e}", flush=True)
                    traceback.print_exc()
                    master_log["runs"].append({"dataset": dataset, "method": method, "seed": seed,
                                                "phase": "eval", "status": "failed", "error": str(e)})

            with open(os.path.join(RESULTS_DIR, "master_log.json"), "w") as f:
                json.dump(master_log, f, indent=2)

    master_log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(RESULTS_DIR, "master_log.json"), "w") as f:
        json.dump(master_log, f, indent=2)
    print("\nALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
