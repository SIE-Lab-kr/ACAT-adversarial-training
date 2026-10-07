# ACAT: Adaptive Curriculum Adversarial Training

Code, results, logs and figures for the manuscript

> **Adaptive Curriculum Adversarial Training: Dynamically Balancing Sample Hardness and Boundary Proximity for Robustness**
> Zhengyu Shang, Manzoor Hussain, Ahmed D. M. Ibrahum, Jang-Eui Hong
> Software Intelligence Engineering Lab, Chungbuk National University (submitted to the *Journal of Information Security and Applications*)

ACAT reweights adversarial-training samples with two signals computed from one forward pass — a margin-based
hardness score *h* and a boundary-proximity score *b* — smooths *b* with a per-sample EMA, and blends them with a
schedule λ<sub>t</sub> that moves the emphasis from easy, high-margin samples to boundary-adjacent samples:

```
w_i = (1 - λ_t) · h_i^n + λ_t · b_i^n ,   loss = mean_i  w_i · [ (1-β) CE(x_i) + β CE(x_i^adv) ]
```

## Repository layout

| Path | Contents |
|---|---|
| `code/gpu/` | GPU pipeline used for the extended study: `train.py`, `evaluate.py`, `common.py` (data, PGD, all five objectives), `run_all_gpu.py` (resumable queue), `make_paper_figs.py`, `export_excel.py` |
| `code/src/`, `code/configs/` | Model definitions (SmallCNN, VGG-11, ResNet-18, WRN-28-10, MobileNet), dataset registry, configs |
| `code/cpu_reference/` | Reference (CPU) implementation used for the reduced-scale supplementary study |
| `results/gpu/<dataset>/<backbone>/<run>/` | Per-run `history.json` (every epoch), `eval.json` (all attacks, ε- and PGD-step curves, per-sample predictions), and `acat_signals_ep*.npz` (per-sample *h*, *b*, smoothed *b*, *w* at early/middle/final epochs) |
| `results/cpu_study/` | Supplementary reduced-scale study (SmallCNN, 2 seeds), plus earlier smoke tests in `earlier_smoke_tests/` |
| `results/tables/` | One CSV per figure/table and `ACAT_results_data.xlsx` (all results, with charts) |
| `logs/` | Raw training / evaluation logs of both studies |
| `figures/paper/` | Figures used in the manuscript |
| `figures/gpu_study/`, `figures/cpu_study/` | All figures generated from the two studies |

Model checkpoints (8.4 GB) are not included; they are available from the authors on request.

## Experimental protocols

| Study | Data | Backbones | Epochs | Seeds | Attacks (ℓ∞, ε = 8/255) |
|---|---|---|---|---|---|
| GPU study (`results/gpu`) | CIFAR-10, CIFAR-100, Fashion-MNIST, Tiny-ImageNet (15 classes), full sets | VGG-11, ResNet-18, WRN-28-10 (CIFAR-10); ResNet-18 (others) | 30 | 1 (42) | FGSM, PGD-20, C&W-ℓ∞, AutoAttack (1k images), Square (1k queries), transfer |
| Supplementary CPU study (`results/cpu_study`) | 640 / 800 / 450-image training subsets | SmallCNN (0.14M) | 20 / 18 / 10 | 2 / 2 / 1 | FGSM, PGD-15, APGD-15, C&W-ℓ∞, Square (100 queries), transfer |

The WideResNet-28-10 results trained for 150 epochs that are reported in the main tables of the manuscript
(CIFAR-10/100 comparison with TRADES, MART and SPAT, and the two-signal ablation) were produced with a separate
implementation that is **not** part of this repository, and they have not been reproduced with the code here.

## Summary of the GPU-study results (accuracy %, CIFAR-10)

| Backbone | Method | Clean | PGD-20 | AutoAttack | Square | Transfer |
|---|---|---|---|---|---|---|
| ResNet-18 | PGD-AT | 82.04 | 50.11 | 46.00 | 58.20 | 79.32 |
| ResNet-18 | TRADES | 80.34 | 51.13 | 46.70 | 57.50 | 77.52 |
| ResNet-18 | MART | 77.67 | 53.63 | 46.30 | 57.10 | 75.32 |
| ResNet-18 | ACAT (β = 0.5) | **86.63** | 42.06 | 37.80 | 57.60 | **82.00** |
| ResNet-18 | ACAT (β = 1) | 82.25 | 50.37 | 45.60 | 58.30 | 79.55 |
| WRN-28-10 | PGD-AT | 86.91 | 51.75 | 48.90 | 61.80 | 85.44 |
| WRN-28-10 | TRADES | 84.59 | 54.55 | 51.90 | 62.30 | 83.17 |
| WRN-28-10 | ACAT (β = 0.5) | **87.22** | 48.09 | 44.30 | 62.10 | 85.34 |

Under this 30-epoch protocol, ACAT with β = 0.5 gives the highest clean accuracy and the best robustness at small
budgets (ε ≤ 2/255 on VGG-11 and ResNet-18), but lower white-box robustness at ε = 8/255; with β = 1 it performs on
par with PGD-AT. Complete numbers for every run are in `results/tables/` and `results/gpu/`.

## Reproducing the GPU study

```bash
cd code
pip install -r cpu_reference/requirements.txt      # torch, torchvision, autoattack, numpy, pandas, matplotlib, openpyxl
python gpu/run_all_gpu.py        # trains + evaluates every cell (resumable; ~24 h on one RTX 3090)
python gpu/make_paper_figs.py    # figures and LaTeX tables
python gpu/export_excel.py       # Excel workbook + CSVs
```

Datasets are read from `~/.cache/torch` (CIFAR-10/100 and Fashion-MNIST via torchvision; Tiny-ImageNet from the
official `tiny-imagenet-200.zip`, first 15 classes of `wnids.txt`).

## Contact

Jang-Eui Hong (corresponding author) — jehong@chungbuk.ac.kr
