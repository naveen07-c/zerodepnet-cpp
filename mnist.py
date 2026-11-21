"""Minimal MNIST loader: download the raw IDX gzip files and parse them.

Avoids torchvision entirely so the only heavy dependency is PyTorch itself.
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

URLS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _download(name: str, root: Path) -> Path:
    path = root / URLS[name]
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URLS[name]} ...")
        urllib.request.urlretrieve(BASE_URL + URLS[name], path)
    return path


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"bad image magic in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"bad label magic in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_split(root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images [n,784] float32 in [0,1], labels [n] int64)."""
    x = _read_images(_download(f"{split}_images", root))
    y = _read_labels(_download(f"{split}_labels", root))
    return torch.from_numpy(x), torch.from_numpy(y)
