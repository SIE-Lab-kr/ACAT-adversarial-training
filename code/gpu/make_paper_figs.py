"""Aggregate results_gpu/ into paper figures (figs_gpu/) and LaTeX tables (tables_gpu/).

Safe to run at any time: missing cells are skipped.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_gpu")
FIG = os.path.join(ROOT, "figs_gpu"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(ROOT, "tables_gpu"); os.makedirs(TAB, exist_ok=True)

METHODS = ["standard", "pgd_at", "trades", "mart", "acat"]
AT = ["pgd_at", "trades", "mart", "acat"]
MNAME = {"standard": "Standard", "pgd_at": "PGD-AT", "trades": "TRADES", "mart": "MART", "acat": "ACAT (ours)"}
COL = {"standard": "#8c8c8c", "pgd_at": "#3a73b5", "trades": "#43a04a", "mart": "#8657b0", "acat": "#d1495b"}
BB = ["vgg11", "resnet18", "wrn28_10"]
BNAME = {"vgg11": "VGG-11", "resnet18": "ResNet-18", "wrn28_10": "WRN-28-10", "smallcnn": "SmallCNN"}
DNAME = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "fashionmnist": "Fashion-MNIST", "tinyimagenet": "Tiny-ImageNet (15 cl.)"}
ATTACKS = [("clean", "Clean"), ("fgsm", "FGSM"), ("pgd20", "PGD-20"), ("cw_linf", "CW-$\\ell_\\infty$"),
           ("autoattack", "AutoAttack"), ("square", "Square\n(black-box)"), ("transfer", "Transfer\n(black-box)")]
ABL = [("acat", "ACAT (full)"), ("acat_hardness_only", "Hardness only ($\\lambda\\equiv0$)"),
       ("acat_boundary_only", "Boundary only ($\\lambda\\equiv1$)"), ("acat_no_ema", "No EMA ($\\alpha=0$)"),
       ("acat_cosine", "Cosine schedule"), ("acat_adv_only", "Adv.-only loss ($\\beta=1$)"),
       ("acat_orig_hardfirst", "Hard-first ($1-h^{n}$)"), ("acat_orig_hardfirst_adv_only", "Hard-first, $\\beta=1$"),
       ("pgd_at", "PGD-AT"), ("trades", "TRADES"), ("mart", "MART")]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(ds, bb, m):
    d = os.path.join(RES, ds, bb, m)
    ev = os.path.join(d, "eval.json"); hi = os.path.join(d, "history.json")
    out = {}
    if os.path.exists(ev):
        out["eval"] = json.load(open(ev))
    if os.path.exists(hi) and os.path.exists(os.path.join(d, "DONE")):
        out["hist"] = json.load(open(hi))
    return out


def acc(e, k):
    v = e.get(k)
    if v is None:
        return None
    return 100 * (v["acc"] if isinstance(v, dict) else v)


def vec(e):
    return [acc(e, k) for k, _ in ATTACKS]


def mes(group):
    """group: {method: vector}; MES relative to best per dimension (Eq. 10); dims with any None dropped."""
    keys = list(group)
    if not keys:
        return {}
    dims = [j for j in range(len(ATTACKS)) if all(group[k][j] is not None for k in keys)]
    best = {j: max(group[k][j] for k in keys) for j in dims}
    return {k: 100 * (1 - np.mean([(best[j] - group[k][j]) / best[j] for j in dims if best[j] > 0])) for k in keys}


# ------------------------------------------------------------------ collect
DATA = {}
for ds in DNAME:
    for bb in list(BNAME):
        for m in METHODS + [a for a, _ in ABL]:
            r = load(ds, bb, m)
            if r:
                DATA[(ds, bb, m)] = r


def evals(ds, bb, methods):
    return {m: DATA[(ds, bb, m)]["eval"] for m in methods if (ds, bb, m) in DATA and "eval" in DATA[(ds, bb, m)]}


def bar_panel(ax, ds, bb, title):
    ev = evals(ds, bb, METHODS)
    if not ev:
        ax.set_visible(False); return False
    x = np.arange(len(ATTACKS)); w = 0.16
    for j, m in enumerate(METHODS):
        if m in ev:
            v = [np.nan if a is None else a for a in vec(ev[m])]
            ax.bar(x + (j - 2) * w, v, w, color=COL[m], label=MNAME[m], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in ATTACKS], fontsize=8.5)
    ax.set_title(title, fontweight="bold"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 100)
    return True


# ------------------------------------------------------------------ F1: per-attack bars, CIFAR-10, 3 backbones
fig, axs = plt.subplots(1, 3, figsize=(19, 4.8))
ok = [bar_panel(ax, "cifar10", bb, f"CIFAR-10, {BNAME[bb]}") for ax, bb in zip(axs, BB)]
if any(ok):
    axs[0].set_ylabel("Accuracy (%)")
    h, l = next(ax for ax, o in zip(axs, ok) if o).get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(FIG, "gpu_cifar10_per_attack.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ F2: other datasets, ResNet-18
others = [ds for ds in ["cifar100", "fashionmnist", "tinyimagenet"] if evals(ds, "resnet18", METHODS)]
if others:
    fig, axs = plt.subplots(1, len(others), figsize=(6.4 * len(others), 4.8), squeeze=False)
    for ax, ds in zip(axs[0], others):
        bar_panel(ax, ds, "resnet18", f"{DNAME[ds]}, ResNet-18")
    axs[0][0].set_ylabel("Accuracy (%)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(FIG, "gpu_other_datasets_per_attack.png"), dpi=200)
    plt.close(fig)

# ------------------------------------------------------------------ MES table data
MES = {}
for ds in DNAME:
    for bb in BNAME:
        ev = evals(ds, bb, AT)
        if len(ev) == len(AT):
            MES[(ds, bb)] = mes({m: vec(e) for m, e in ev.items()})

# F3: MES bars per (dataset, backbone) group
groups = [(ds, bb) for (ds, bb) in MES]
if groups:
    fig, ax = plt.subplots(figsize=(max(6, 2.3 * len(groups)), 4))
    x = np.arange(len(groups)); w = 0.2
    for j, m in enumerate(AT):
        v = [MES[g][m] for g in groups]
        ax.bar(x + (j - 1.5) * w, v, w, color=COL[m], label=MNAME[m], edgecolor="white")
        for xi, vi in zip(x + (j - 1.5) * w, v):
            ax.text(xi, vi + 0.4, f"{vi:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{DNAME[d]}\n{BNAME[b]}" for d, b in groups], fontsize=9)
    lo = min(min(MES[g].values()) for g in groups)
    ax.set_ylim(max(0, lo - 8), 101); ax.set_ylabel("MES (%)"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gpu_mes.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ F4: training dynamics (3 backbones x 3 metrics)
rows = [bb for bb in BB if any(("cifar10", bb, m) in DATA and "hist" in DATA[("cifar10", bb, m)] for m in METHODS)]
if rows:
    fig, axs = plt.subplots(len(rows), 3, figsize=(15, 3.3 * len(rows)), squeeze=False)
    for r, bb in enumerate(rows):
        for m in METHODS:
            d = DATA.get(("cifar10", bb, m), {}).get("hist")
            if not d:
                continue
            ep = [e["epoch"] + 1 for e in d["epochs"]]
            axs[r][0].plot(ep, [e["train_loss"] for e in d["epochs"]], color=COL[m], label=MNAME[m])
            axs[r][1].plot(ep, [100 * e["clean_acc"] for e in d["epochs"]], color=COL[m])
            axs[r][2].plot(ep, [100 * e["pgd10_acc"] for e in d["epochs"]], color=COL[m])
        axs[r][0].set_ylabel(f"{BNAME[bb]}\ntraining loss"); axs[r][1].set_ylabel("clean acc. (%)"); axs[r][2].set_ylabel("PGD-10 acc. (%)")
        for c in range(3):
            axs[r][c].grid(alpha=0.3)
    for c in range(3):
        axs[-1][c].set_xlabel("Epoch")
    axs[0][0].set_title("(a) Training loss"); axs[0][1].set_title("(b) Clean accuracy (1k test images)"); axs[0][2].set_title("(c) PGD-10 robust accuracy (1k test images)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(FIG, "gpu_training_dynamics.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ F5: ACAT curriculum dynamics
cur = [bb for bb in BB if ("cifar10", bb, "acat") in DATA and "hist" in DATA[("cifar10", bb, "acat")]]
if cur:
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    ls = {"vgg11": ":", "resnet18": "-", "wrn28_10": "--"}
    for bb in cur:
        E = DATA[("cifar10", bb, "acat")]["hist"]["epochs"]
        ep = [e["epoch"] + 1 for e in E]
        axs[0].plot(ep, [e["corr_w_h"] for e in E], color="#3a73b5", ls=ls[bb], label=f"corr($w$, $h$) {BNAME[bb]}")
        axs[0].plot(ep, [e["corr_w_b"] for e in E], color="#d1495b", ls=ls[bb], label=f"corr($w$, $b^{{s}}$) {BNAME[bb]}")
        axs[1].fill_between(ep, [e["w_p10"] for e in E], [e["w_p90"] for e in E], alpha=0.15, color="gray")
        axs[1].plot(ep, [e["w_std"] for e in E], color="k", ls=ls[bb], label=f"std($w$) {BNAME[bb]}")
        axs[2].plot(ep, [e["h_mean"] for e in E], color="#3a73b5", ls=ls[bb], label=f"mean $h$ {BNAME[bb]}")
        axs[2].plot(ep, [e["b_smooth_mean"] for e in E], color="#d1495b", ls=ls[bb], label=f"mean $b^{{s}}$ {BNAME[bb]}")
    E0 = DATA[("cifar10", cur[0], "acat")]["hist"]["epochs"]
    ax2 = axs[0].twinx(); ax2.plot([e["epoch"] + 1 for e in E0], [e["lambda"] for e in E0], color="orange", lw=2.5, alpha=0.6)
    ax2.set_ylabel("$\\lambda_t$", color="orange"); ax2.spines["right"].set_visible(True)
    axs[0].set_title("(a) What the weights follow"); axs[0].set_ylabel("Pearson correlation")
    axs[1].set_title("(b) Spread of weights (band: 10th-90th pct.)"); axs[1].set_ylabel("$w_i$")
    axs[2].set_title("(c) Mean difficulty signals"); axs[2].set_ylabel("value")
    for ax in axs:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gpu_acat_curriculum_dynamics.png"), dpi=200); plt.close(fig)

# F6: weight vs hardness at early / middle / final epoch (ResNet-18 preferred)
for bb in ["resnet18", "wrn28_10", "vgg11"]:
    files = sorted(glob.glob(os.path.join(RES, "cifar10", bb, "acat", "acat_signals_ep*.npz")),
                   key=lambda f: int(f.split("ep")[-1].split(".")[0]))
    if len(files) >= 3:
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
        for ax, f, name in zip(axs, files, ["early", "middle", "final"]):
            d = np.load(f)["data"]; ep = int(f.split("ep")[-1].split(".")[0]) + 1
            idx = np.random.default_rng(0).choice(len(d), min(15000, len(d)), replace=False)
            sc = ax.scatter(d[idx, 1], d[idx, 4], c=d[idx, 3], s=2, cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
            ax.set_title(f"{name} (epoch {ep})"); ax.set_xlabel("margin-based hardness $h_i$"); ax.grid(alpha=0.3)
        axs[0].set_ylabel("curriculum weight $w_i$")
        cb = fig.colorbar(sc, ax=axs, fraction=0.02); cb.set_label("smoothed boundary proximity $b_i^{s}$")
        fig.savefig(os.path.join(FIG, f"gpu_acat_weights_vs_hardness_{bb}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
        break

# ------------------------------------------------------------------ F7/F8: eps curve and PGD-steps curve
for key, fname, xlabel, xf in [("eps_curve", "gpu_eps_curve.png", "$\\epsilon$ (x/255)", float),
                               ("steps_curve", "gpu_pgd_steps_curve.png", "PGD iterations", float)]:
    have = [bb for bb in BB if evals("cifar10", bb, METHODS) and all(key in e for e in evals("cifar10", bb, METHODS).values())]
    if not have:
        continue
    fig, axs = plt.subplots(1, len(have), figsize=(5.2 * len(have), 3.8), squeeze=False)
    for ax, bb in zip(axs[0], have):
        for m, e in evals("cifar10", bb, METHODS).items():
            ks = sorted(e[key], key=float)
            ax.plot([xf(k) for k in ks], [100 * e[key][k] for k in ks], marker="o", ms=3, color=COL[m], label=MNAME[m])
        ax.set_title(f"CIFAR-10, {BNAME[bb]}"); ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
        if key == "steps_curve":
            ax.set_xscale("log")
        if key == "eps_curve":
            ax.axvline(8, color="gray", ls=":", lw=1)
    axs[0][0].set_ylabel("Robust accuracy (%)"); axs[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ F9: clean vs robust trade-off scatter
pts = [(ds, bb, m, e) for ds in DNAME for bb in BNAME for m, e in evals(ds, bb, AT).items() if e.get("autoattack") is not None]
if pts:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    mk = {"vgg11": "s", "resnet18": "o", "wrn28_10": "D"}
    for ds, bb, m, e in pts:
        if ds != "cifar10":
            continue
        ax.scatter(acc(e, "clean"), acc(e, "autoattack"), color=COL[m], marker=mk[bb], s=60, edgecolor="k", lw=0.5)
    for bb in BB:
        ax.scatter([], [], marker=mk[bb], color="w", edgecolor="k", label=BNAME[bb])
    for m in AT:
        ax.scatter([], [], marker="o", color=COL[m], label=MNAME[m])
    ax.set_xlabel("Clean accuracy (%)"); ax.set_ylabel("AutoAttack accuracy (%)"); ax.grid(alpha=0.3)
    ax.set_title("CIFAR-10: clean vs. robust accuracy (upper right is better)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gpu_tradeoff_scatter.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ F10: per-class PGD-20 accuracy heatmap (ResNet-18)
ev = evals("cifar10", "resnet18", METHODS)
if ev and all("preds" in e.get("pgd20", {}) for e in ev.values()):
    import pickle
    y = []
    with open(os.path.join(os.path.expanduser("~"), ".cache", "torch", "cifar-10-batches-py", "test_batch"), "rb") as fh:
        y = np.array(pickle.load(fh, encoding="latin1")["labels"])
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    M = np.array([[100 * np.mean(np.array(ev[m]["pgd20"]["preds"])[y == c] == c) for c in range(10)] for m in METHODS if m in ev])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(10):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="w" if M[i, j] < 50 else "k", fontsize=8)
    ax.set_xticks(range(10)); ax.set_xticklabels(classes); ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([MNAME[m] for m in METHODS if m in ev])
    fig.colorbar(im, ax=ax, label="PGD-20 accuracy (%)")
    ax.set_title("Per-class PGD-20 accuracy, CIFAR-10 / ResNet-18")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gpu_per_class_pgd20.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ F11: training cost
cost = {(bb, m): DATA[("cifar10", bb, m)]["hist"]["train_time_sec"] / 60 for bb in BB for m in METHODS
        if ("cifar10", bb, m) in DATA and "hist" in DATA[("cifar10", bb, m)] and "train_time_sec" in DATA[("cifar10", bb, m)]["hist"]}
cbb = [bb for bb in BB if all((bb, m) in cost for m in METHODS)]
if cbb:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(cbb)); w = 0.16
    for j, m in enumerate(METHODS):
        v = [cost[(bb, m)] for bb in cbb]
        ax.bar(x + (j - 2) * w, v, w, color=COL[m], label=MNAME[m], edgecolor="white")
        if m == "acat":
            for xi, bb in zip(x + (j - 2) * w, cbb):
                ax.text(xi, cost[(bb, m)] * 1.03, f"{cost[(bb, 'acat')] / cost[(bb, 'pgd_at')]:.2f}x", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([BNAME[b] for b in cbb]); ax.set_ylabel("Training time (min, 30 epochs, RTX 3090)")
    ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=3)
    ax.set_title("Training cost (label: ACAT time / PGD-AT time)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gpu_training_cost.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ F12: ablation (ResNet-18)
abl = [(k, n) for k, n in ABL if ("cifar10", "resnet18", k) in DATA and "eval" in DATA[("cifar10", "resnet18", k)]]
if len(abl) >= 2:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    style = {"acat": ("*", COL["acat"], 280), "pgd_at": ("o", COL["pgd_at"], 90),
             "trades": ("o", COL["trades"], 90), "mart": ("o", COL["mart"], 90)}
    for k, n in abl:
        e = DATA[("cifar10", "resnet18", k)]["eval"]
        beta1 = "adv_only" in k
        mk, c, sz = style.get(k, ("^" if beta1 else "D", "#6b6b6b" if beta1 else "#e08a2c", 80))
        ax.scatter(acc(e, "clean"), acc(e, "autoattack"), marker=mk, color=c, s=sz, edgecolor="k", lw=0.5, zorder=3)
        off = {"acat_orig_hardfirst_adv_only": (-4, -13), "acat_adv_only": (7, 2)}.get(k, (6, 3))
        ax.annotate(n, (acc(e, "clean"), acc(e, "autoattack")), textcoords="offset points", xytext=off, fontsize=7.5,
                    ha="right" if k == "acat_orig_hardfirst_adv_only" else "left")
    ax.scatter([], [], marker="D", color="#e08a2c", edgecolor="k", label="ACAT variant, $\\beta=0.5$")
    ax.scatter([], [], marker="^", color="#6b6b6b", edgecolor="k", label="ACAT variant, $\\beta=1$")
    ax.scatter([], [], marker="o", color="w", edgecolor="k", label="baseline")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Clean accuracy (%)"); ax.set_ylabel("AutoAttack accuracy (%)"); ax.grid(alpha=0.3)
    ax.set_xlim(76.5, 89.5)
    ax.set_title("CIFAR-10 / ResNet-18: ACAT variants and baselines")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gpu_ablation_resnet18.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ LaTeX tables
def fmt(v, best):
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return f"\\textbf{{{s}}}" if best is not None and abs(v - best) < 1e-9 else s


def table(ds, bbs, label, caption, path):
    lines = [r"\begin{table*}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\footnotesize",
             r"\begin{tabular}{llcccccccc}", r"\toprule",
             r"Backbone & Method & Clean & FGSM & PGD-20 & C\&W-$\ell_\infty$ & AA & Square & Transfer & MES \\", r"\midrule"]
    any_row = False
    for bb in bbs:
        ev = evals(ds, bb, METHODS)
        if not ev:
            continue
        any_row = True
        best = {k: max((acc(ev[m], k) for m in AT if m in ev and acc(ev[m], k) is not None), default=None) for k, _ in ATTACKS}
        mesd = MES.get((ds, bb), {})
        bm = max(mesd.values()) if mesd else None
        rows = [m for m in METHODS if m in ev]
        for i, m in enumerate(rows):
            cells = [fmt(acc(ev[m], k), best[k] if m != "standard" else None) for k, _ in ATTACKS]
            mm = fmt(mesd[m], bm) if m in mesd else "--"
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{BNAME[bb]}}}" if i == 0 else ""
            lines.append(f"{first} & {MNAME[m]} & " + " & ".join(cells) + f" & {mm} \\\\")
        lines.append(r"\midrule")
    if not any_row:
        return
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(lines))


table("cifar10", BB, "tab:gpu_cifar10",
      "CIFAR-10 results for three backbones (30 epochs, full training set, one seed). FGSM, PGD-20 and C\\&W-$\\ell_\\infty$ (30 steps) and the transfer attack use the full 10{,}000-image test set; AutoAttack (standard) and Square Attack (1{,}000 queries) use the first 1{,}000 test images. MES is computed over the four adversarially trained methods; bold marks the best adversarially trained method per column.",
      os.path.join(TAB, "table_gpu_cifar10.tex"))
table_others = [ds for ds in ["cifar100", "fashionmnist", "tinyimagenet"] if evals(ds, "resnet18", METHODS)]
for ds in table_others:
    table(ds, ["resnet18"], f"tab:gpu_{ds}", f"{DNAME[ds]} results with ResNet-18 (same protocol as Table~\\ref{{tab:gpu_cifar10}}).",
          os.path.join(TAB, f"table_gpu_{ds}.tex"))

# summary json for writing the paper
summary = {f"{ds}/{bb}/{m}": {**{k: acc(DATA[(ds, bb, m)]["eval"], k) for k, _ in ATTACKS},
                              "mes": MES.get((ds, bb), {}).get(m),
                              "train_min": DATA[(ds, bb, m)].get("hist", {}).get("train_time_sec", 0) / 60}
           for (ds, bb, m) in DATA if "eval" in DATA[(ds, bb, m)]}
json.dump(summary, open(os.path.join(TAB, "summary.json"), "w"), indent=1)
print("figures:", sorted(os.listdir(FIG)))
print("tables:", sorted(os.listdir(TAB)))
