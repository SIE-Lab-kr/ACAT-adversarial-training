"""
Builds every reviewer-facing figure/table from the real logged results in
results/<dataset>/<method>/{history.json, eval_results.json}.

Outputs into figures/:
  loss_curves_<dataset>.png          -- training loss vs epoch, all methods
  accuracy_curves_<dataset>.png      -- clean & PGD-10 acc vs epoch, all methods
  robustness_bars_<dataset>.png      -- final accuracy under every attack, all methods
  mes_comparison.png                 -- Multi-threat Efficiency Score, all datasets x methods
  attack_heatmap_<dataset>.png       -- accuracy-drop heatmap (method x attack)
  results_summary.csv / .md          -- full numeric table
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METHODS = ["standard", "pgd_at", "trades", "mart", "acat"]
METHOD_LABELS = {"standard": "Standard", "pgd_at": "PGD-AT", "trades": "TRADES",
                  "mart": "MART", "acat": "ACAT (proposed)"}
COLORS = {"standard": "#888888", "pgd_at": "#1f77b4", "trades": "#2ca02c",
          "mart": "#9467bd", "acat": "#d62728"}
ATTACK_ORDER = ["clean", "fgsm", "pgd20", "apgd20", "cw_linf", "square_bbox", "transfer_bbox"]
ATTACK_LABELS = ["Clean", "FGSM", "PGD-20", "APGD-20", "CW-Linf", "Square\n(black-box)", "Transfer\n(black-box)"]


def load_run(dataset, method):
    hist_path = os.path.join(RESULTS_DIR, dataset, method, "history.json")
    eval_path = os.path.join(RESULTS_DIR, dataset, method, "eval_results.json")
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else None
    eva = json.load(open(eval_path)) if os.path.exists(eval_path) else None
    return hist, eva


def plot_loss_curves(dataset):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in METHODS:
        hist, _ = load_run(dataset, m)
        if hist is None:
            continue
        ax.plot(hist["epoch"], hist["train_loss"], label=METHOD_LABELS[m],
                color=COLORS[m], marker="o", markersize=3, linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(f"Training loss — {dataset}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"loss_curves_{dataset}.png"), dpi=200)
    plt.close(fig)


def plot_accuracy_curves(dataset):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in METHODS:
        hist, _ = load_run(dataset, m)
        if hist is None:
            continue
        axes[0].plot(hist["epoch"], hist["clean_acc"], label=METHOD_LABELS[m],
                      color=COLORS[m], marker="o", markersize=3, linewidth=1.6)
        axes[1].plot(hist["epoch"], hist["pgd10_robust_acc"], label=METHOD_LABELS[m],
                      color=COLORS[m], marker="o", markersize=3, linewidth=1.6)
    axes[0].set_title(f"Clean accuracy — {dataset}")
    axes[1].set_title(f"PGD-10 robust accuracy (in-loop) — {dataset}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"accuracy_curves_{dataset}.png"), dpi=200)
    plt.close(fig)


def plot_robustness_bars(dataset):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    n_attacks = len(ATTACK_ORDER)
    width = 0.15
    x = np.arange(n_attacks)
    for i, m in enumerate(METHODS):
        _, eva = load_run(dataset, m)
        if eva is None:
            continue
        vals = [eva["accuracy"].get(a, np.nan) for a in ATTACK_ORDER]
        ax.bar(x + (i - 2) * width, vals, width=width, label=METHOD_LABELS[m], color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_LABELS, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"White-box & black-box robustness — {dataset}")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"robustness_bars_{dataset}.png"), dpi=200)
    plt.close(fig)


def plot_attack_heatmap(dataset):
    data = []
    for m in METHODS:
        _, eva = load_run(dataset, m)
        if eva is None:
            data.append([np.nan] * len(ATTACK_ORDER))
            continue
        data.append([eva["accuracy"].get(a, np.nan) for a in ATTACK_ORDER])
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ATTACK_ORDER)))
    ax.set_xticklabels(ATTACK_LABELS, fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_title(f"Accuracy under each attack — {dataset}")
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"attack_heatmap_{dataset}.png"), dpi=200)
    plt.close(fig)


def plot_mes_comparison(datasets):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.15
    x = np.arange(len(datasets))
    for i, m in enumerate(METHODS):
        vals = []
        for d in datasets:
            _, eva = load_run(d, m)
            vals.append(eva["MES"] if eva else np.nan)
        ax.bar(x + (i - 2) * width, vals, width=width, label=METHOD_LABELS[m], color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Multi-threat Efficiency Score (MES)")
    ax.set_title("MES across datasets and methods (real-executed runs)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mes_comparison.png"), dpi=200)
    plt.close(fig)


def build_summary_table(datasets):
    rows = []
    for d in datasets:
        for m in METHODS:
            hist, eva = load_run(d, m)
            if hist is None or eva is None:
                continue
            row = {"dataset": d, "method": METHOD_LABELS[m],
                   "n_train": hist["n_train"], "n_test": hist["n_test"],
                   "epochs": len(hist["epoch"]), "total_time_sec": round(hist["total_time_sec"], 1)}
            row.update({f"acc_{k}": round(v, 4) for k, v in eva["accuracy"].items()})
            row["mean_robust_acc"] = round(eva["mean_robust_acc"], 4)
            row["MES"] = round(eva["MES"], 4)
            rows.append(row)

    if not rows:
        print("No results found yet.")
        return

    cols = list(rows[0].keys())
    csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    md_path = os.path.join(RESULTS_DIR, "results_summary.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"Wrote {csv_path} and {md_path} ({len(rows)} rows)")


if __name__ == "__main__":
    datasets = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    datasets = sorted(datasets)
    for d in datasets:
        plot_loss_curves(d)
        plot_accuracy_curves(d)
        plot_robustness_bars(d)
        plot_attack_heatmap(d)
        print(f"Figures done for {d}")
    plot_mes_comparison(datasets)
    build_summary_table(datasets)
    print("ALL FIGURES DONE")
