# Config files

Each `<dataset>.yaml` gives the **full paper-scale** training configuration
(architecture, epochs, optimizer, PGD budget) for that dataset, matching the
revision plan's expanded generalization study. These are the settings to use
on your own GPU machine — they are intentionally *not* what ran in this
sandboxed CPU-only session (see `../EXECUTION_LOG.md` for the actual
small-scale settings that were executed here).

| Dataset | Status in this session | Real data downloaded here? |
|---|---|---|
| CIFAR-10 | Trained + evaluated (small-scale, real data) | Yes |
| Fashion-MNIST | Trained + evaluated (small-scale, real data) | Yes |
| Tiny-ImageNet-200 | Trained + evaluated (15-class subset, real data) | Yes |
| CIFAR-100 | Code-ready only | No (host blocked from sandbox; `torchvision.datasets.CIFAR100(download=True)` works normally on your machine) |
| SVHN | Code-ready only | No (same reason; `torchvision.datasets.SVHN(download=True)`) |
| GTSRB | Code-ready only | No (same reason; `torchvision.datasets.GTSRB(download=True)`) |
| STL-10 | Code-ready only | No (same reason; `torchvision.datasets.STL10(download=True)`) |

To run any dataset at full scale:

```bash
python run_experiment.py --dataset cifar100 --method acat --config configs/cifar100.yaml
```
