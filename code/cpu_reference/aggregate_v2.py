"""
Aggregates the v2 (increased-epoch, multi-seed) reruns in results/ into:
  - results/results_summary_v2_per_seed.csv   (one row per dataset,method,seed)
  - results/results_summary_v2.csv            (mean +/- std across seeds)

MES here is recomputed using the SAME ideal-point / relative-gap formula as
Eq. (10) in the paper (not the harmonic-mean formula src/evaluate.py returns
internally), computed per seed over the 3-dim vector (clean, mean_whitebox,
mean_blackbox) with p* = the best value observed among the methods present
for that dataset+seed -- exactly matching how Table V is defined in the text.
"""
import os
import re
import json
import glob
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

WHITEBOX_KEYS = ["fgsm", "pgd20", "apgd20", "cw_linf"]
BLACKBOX_KEYS = ["square_bbox", "transfer_bbox"]

METHOD_LABEL = {
    "standard": "Standard",
    "pgd_at": "PGD-AT",
    "trades": "TRADES",
    "mart": "MART",
    "acat": "ACAT (ours)",
}
METHOD_ORDER = ["standard", "pgd_at", "trades", "mart", "acat"]


def load_run(dataset, method, seed):
    d = os.path.join(RESULTS_DIR, dataset, method, f"seed{seed}")
    eval_path = os.path.join(d, "eval_results.json")
    hist_path = os.path.join(d, "history.json")
    if not os.path.exists(eval_path) or not os.path.exists(hist_path):
        return None
    with open(eval_path) as f:
        ev = json.load(f)
    with open(hist_path) as f:
        hist = json.load(f)
    acc = ev["accuracy"]
    clean = acc["clean"]
    mean_wb = float(np.mean([acc[k] for k in WHITEBOX_KEYS]))
    mean_bb = float(np.mean([acc[k] for k in BLACKBOX_KEYS if k in acc]))
    return {
        "clean": clean, "mean_wb": mean_wb, "mean_bb": mean_bb,
        "raw": {k: acc.get(k) for k in WHITEBOX_KEYS + BLACKBOX_KEYS},
        "total_time_sec": hist["total_time_sec"],
        "n_train": hist["n_train"], "n_test": hist["n_test"],
        "epochs": len(hist["epoch"]),
    }


def discover_seeds(dataset, method):
    pattern = os.path.join(RESULTS_DIR, dataset, method, "seed*")
    seeds = []
    for p in glob.glob(pattern):
        m = re.search(r"seed(\d+)$", p)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def mes_eq10(vec, pstar):
    """paper Eq. (10): MES = 1 - mean_j (p*_j - p_j)/p*_j over j=0..m (m+1 dims)."""
    gaps = [(pstar[j] - vec[j]) / pstar[j] if pstar[j] > 0 else 0.0 for j in range(len(vec))]
    return 1.0 - float(np.mean(gaps))


def main():
    datasets = sorted(d for d in os.listdir(RESULTS_DIR)
                       if os.path.isdir(os.path.join(RESULTS_DIR, d)))
    per_seed_rows = []
    summary_rows = []

    for dataset in datasets:
        # collect per-seed per-method point vectors first (needed for p*)
        per_seed_data = {}  # seed -> {method: {..}}
        for method in METHOD_ORDER:
            for seed in discover_seeds(dataset, method):
                run = load_run(dataset, method, seed)
                if run is None:
                    continue
                per_seed_data.setdefault(seed, {})[method] = run

        for seed, methods_here in sorted(per_seed_data.items()):
            present = list(methods_here.keys())
            pstar = [
                max(methods_here[m]["clean"] for m in present),
                max(methods_here[m]["mean_wb"] for m in present),
                max(methods_here[m]["mean_bb"] for m in present),
            ]
            for method in present:
                r = methods_here[method]
                vec = [r["clean"], r["mean_wb"], r["mean_bb"]]
                mes = mes_eq10(vec, pstar)
                per_seed_rows.append({
                    "dataset": dataset, "method": METHOD_LABEL[method], "seed": seed,
                    "epochs": r["epochs"], "n_train": r["n_train"], "n_test": r["n_test"],
                    "clean": r["clean"], "mean_whitebox": r["mean_wb"],
                    "mean_blackbox": r["mean_bb"], "MES": mes,
                    "total_time_sec": r["total_time_sec"],
                    **{f"acc_{k}": r["raw"][k] for k in WHITEBOX_KEYS + BLACKBOX_KEYS},
                })

        # now aggregate mean/std across seeds per method
        by_method = {}
        for row in per_seed_rows:
            if row["dataset"] != dataset:
                continue
            by_method.setdefault(row["method"], []).append(row)
        for method in METHOD_ORDER:
            label = METHOD_LABEL[method]
            rows = by_method.get(label, [])
            if not rows:
                continue
            def agg(key):
                vals = [r[key] for r in rows]
                return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)
            clean_m, clean_s = agg("clean")
            wb_m, wb_s = agg("mean_whitebox")
            bb_m, bb_s = agg("mean_blackbox")
            mes_m, mes_s = agg("MES")
            time_m, _ = agg("total_time_sec")
            summary_rows.append({
                "dataset": dataset, "method": label, "n_seeds": len(rows),
                "epochs": rows[0]["epochs"], "n_train": rows[0]["n_train"], "n_test": rows[0]["n_test"],
                "clean_mean": clean_m * 100, "clean_std": clean_s * 100,
                "mean_whitebox_mean": wb_m * 100, "mean_whitebox_std": wb_s * 100,
                "mean_blackbox_mean": bb_m * 100, "mean_blackbox_std": bb_s * 100,
                "MES_mean": mes_m * 100, "MES_std": mes_s * 100,
                "avg_total_time_sec": time_m,
            })

    # write CSVs
    import csv
    per_seed_path = os.path.join(RESULTS_DIR, "results_summary_v2_per_seed.csv")
    if per_seed_rows:
        keys = list(per_seed_rows[0].keys())
        with open(per_seed_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in per_seed_rows:
                w.writerow(row)

    summary_path = os.path.join(RESULTS_DIR, "results_summary_v2.csv")
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    print(f"Wrote {len(per_seed_rows)} per-seed rows -> {per_seed_path}")
    print(f"Wrote {len(summary_rows)} summary rows -> {summary_path}")
    for row in summary_rows:
        print(row["dataset"], row["method"], row["n_seeds"], "seeds",
              f"clean={row['clean_mean']:.2f}+-{row['clean_std']:.2f}",
              f"wb={row['mean_whitebox_mean']:.2f}+-{row['mean_whitebox_std']:.2f}",
              f"bb={row['mean_blackbox_mean']:.2f}+-{row['mean_blackbox_std']:.2f}",
              f"MES={row['MES_mean']:.2f}+-{row['MES_std']:.2f}",
              f"time={row['avg_total_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
