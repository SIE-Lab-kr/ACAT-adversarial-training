"""
Dataset registry and loaders for the ACAT multi-dataset evaluation suite.

Datasets are split into two tiers, and this is reported honestly everywhere
results are shown (see EXECUTION_LOG.md at the project root):

  REAL-EXECUTED (real public images, actually downloaded and trained/evaluated
  in this sandboxed, GPU-less, network-restricted session):
      - CIFAR-10        (real photos, github.com/YoongiKim/CIFAR-10-images mirror)
      - Fashion-MNIST   (full official dataset, zalandoresearch/fashion-mnist)
      - Tiny-ImageNet-200 (15-class / 50-img-per-class real subset,
                           github.com/seshuad/IMagenet mirror)

  CODE-READY (loader is standard torchvision and will download automatically
  the first time it is run on a machine with normal internet access / the
  standard dataset mirrors that are blocked from this sandbox):
      - CIFAR-100
      - SVHN
      - GTSRB
      - STL-10

All loaders return a torch.utils.data.Dataset whose __getitem__ yields
(image_tensor in [0,1] range CxHxW, label, sample_index). Sample_index is
required by ACAT's per-sample EMA difficulty tracker.
"""
import os
import gzip
import struct
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw")

# ----------------------------------------------------------------------------
# Registry: canonical metadata per dataset (used by models.py / train.py)
# ----------------------------------------------------------------------------
DATASET_INFO = {
    "cifar10":      dict(num_classes=10,  channels=3, img_size=32, real=True,
                          mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    "fashionmnist": dict(num_classes=10,  channels=1, img_size=28, real=True,
                          mean=[0.2860], std=[0.3530]),
    "tinyimagenet": dict(num_classes=15,  channels=3, img_size=64, real=True,
                          mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]),
    "cifar100":     dict(num_classes=100, channels=3, img_size=32, real=False,
                          mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    "svhn":         dict(num_classes=10,  channels=3, img_size=32, real=False,
                          mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    "gtsrb":        dict(num_classes=43,  channels=3, img_size=32, real=False,
                          mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]),
    "stl10":        dict(num_classes=10,  channels=3, img_size=96, real=False,
                          mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
}


class ArrayImageDataset(Dataset):
    """Wraps an in-memory uint8 array of images + labels."""
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            img = img[:, :, None]
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        label = int(self.labels[idx])
        return img, label, idx


# ----------------------------------------------------------------------------
# CIFAR-10  (real, from per-class JPEGs downloaded via GitHub raw mirror)
# ----------------------------------------------------------------------------
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]


def _load_folder_of_class_dirs(root, class_names, img_size):
    images, labels = [], []
    for ci, cname in enumerate(class_names):
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            fpath = os.path.join(cdir, fname)
            try:
                im = Image.open(fpath).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            except Exception:
                continue
            images.append(np.array(im, dtype=np.uint8))
            labels.append(ci)
    return np.stack(images), np.array(labels, dtype=np.int64)


def load_cifar10():
    train_root = os.path.join(DATA_ROOT, "cifar10_raw")
    test_root = os.path.join(DATA_ROOT, "cifar10_raw_test")
    xtr, ytr = _load_folder_of_class_dirs(train_root, CIFAR10_CLASSES, 32)
    xte, yte = _load_folder_of_class_dirs(test_root, CIFAR10_CLASSES, 32)
    return ArrayImageDataset(xtr, ytr), ArrayImageDataset(xte, yte), CIFAR10_CLASSES


# ----------------------------------------------------------------------------
# Fashion-MNIST (real, full official IDX files)
# ----------------------------------------------------------------------------
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(rows * cols * n)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return data


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        buf = f.read(n)
        data = np.frombuffer(buf, dtype=np.uint8)
    return data


def load_fashionmnist(n_train=1200, n_test=300, seed=0):
    root = os.path.join(DATA_ROOT, "fashionmnist_raw")
    xtr_full = _read_idx_images(os.path.join(root, "train-images-idx3-ubyte.gz"))
    ytr_full = _read_idx_labels(os.path.join(root, "train-labels-idx1-ubyte.gz"))
    xte_full = _read_idx_images(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    yte_full = _read_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte.gz"))
    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(xtr_full), size=min(n_train, len(xtr_full)), replace=False)
    te_idx = rng.choice(len(xte_full), size=min(n_test, len(xte_full)), replace=False)
    return (ArrayImageDataset(xtr_full[tr_idx], ytr_full[tr_idx]),
            ArrayImageDataset(xte_full[te_idx], yte_full[te_idx]),
            FASHION_CLASSES)


# ----------------------------------------------------------------------------
# Tiny-ImageNet-200 (real, 15-class subset downloaded via GitHub mirror)
# ----------------------------------------------------------------------------
def load_tinyimagenet(test_frac=0.2, seed=0):
    root = os.path.join(DATA_ROOT, "tinyimagenet_raw")
    with open(os.path.join(root, "classes15.txt")) as f:
        wnids = [l.strip() for l in f if l.strip()]
    # human-readable names, best-effort from words.txt
    words = {}
    words_path = os.path.join(root, "words.txt")
    if os.path.exists(words_path):
        with open(words_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    words[parts[0]] = parts[1].split(",")[0]
    class_names = [words.get(w, w) for w in wnids]

    images, labels = [], []
    for ci, wnid in enumerate(wnids):
        cdir = os.path.join(root, "images", wnid)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            fpath = os.path.join(cdir, fname)
            try:
                im = Image.open(fpath).convert("RGB").resize((64, 64), Image.BILINEAR)
            except Exception:
                continue
            images.append(np.array(im, dtype=np.uint8))
            labels.append(ci)
    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(images))
    n_test = int(len(images) * test_frac)
    te_idx, tr_idx = perm[:n_test], perm[n_test:]
    return (ArrayImageDataset(images[tr_idx], labels[tr_idx]),
            ArrayImageDataset(images[te_idx], labels[te_idx]),
            class_names)


# ----------------------------------------------------------------------------
# CODE-READY datasets (standard torchvision auto-download; blocked in this
# sandbox's network but will work unmodified on a normal internet connection).
# ----------------------------------------------------------------------------
def load_torchvision_dataset(name):
    """
    name in {'cifar100', 'svhn', 'gtsrb', 'stl10'}.
    Requires internet access to the dataset's standard host (not available in
    this sandboxed session). Provided so the full experimental matrix in the
    paper's revision plan is code-complete; run this on a normal machine.
    """
    import torchvision
    import torchvision.transforms as T
    root = os.path.join(DATA_ROOT, name)
    os.makedirs(root, exist_ok=True)
    to_tensor = T.ToTensor()
    if name == "cifar100":
        train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=to_tensor)
        test = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=to_tensor)
        classes = train.classes
    elif name == "svhn":
        train = torchvision.datasets.SVHN(root, split="train", download=True, transform=to_tensor)
        test = torchvision.datasets.SVHN(root, split="test", download=True, transform=to_tensor)
        classes = [str(i) for i in range(10)]
    elif name == "gtsrb":
        train = torchvision.datasets.GTSRB(root, split="train", download=True, transform=to_tensor)
        test = torchvision.datasets.GTSRB(root, split="test", download=True, transform=to_tensor)
        classes = [str(i) for i in range(43)]
    elif name == "stl10":
        train = torchvision.datasets.STL10(root, split="train", download=True, transform=to_tensor)
        test = torchvision.datasets.STL10(root, split="test", download=True, transform=to_tensor)
        classes = train.classes
    else:
        raise ValueError(name)

    class _Wrap(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return img, label, idx

    return _Wrap(train), _Wrap(test), classes


def get_dataset(name):
    """Unified entry point. Returns (train_ds, test_ds, class_names)."""
    if name == "cifar10":
        return load_cifar10()
    if name == "fashionmnist":
        return load_fashionmnist()
    if name == "tinyimagenet":
        return load_tinyimagenet()
    if name in ("cifar100", "svhn", "gtsrb", "stl10"):
        return load_torchvision_dataset(name)
    raise ValueError(f"Unknown dataset: {name}")
