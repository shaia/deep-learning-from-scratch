"""
MNIST download + loader, shared by the modules that need a real dataset
(Module 01 MLP onward).

Downloads the four classic IDX files once into data/mnist/ (gitignored) from a
stable mirror, caches them, and parses them into NumPy arrays. Images are
returned as float64 in [0, 1] with shape (n, 784); labels as int64 in [0, 10).

Use it from anywhere in the repo:

    import sys, os
    sys.path.insert(0, os.path.join(REPO_ROOT, "data"))
    from get_mnist import load_mnist
    X_train, y_train, X_test, y_test = load_mnist()

Run directly to fetch + report shapes:
    python data/get_mnist.py
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "mnist")

# Mirrors that still serve the original IDL files. We try them in order; the
# first two are the mirrors torchvision/TF fall back to.
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(fname: str) -> str:
    """Fetch fname into CACHE_DIR if not already present; return its path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    dest = os.path.join(CACHE_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    last_err = None
    for base in MIRRORS:
        url = base + fname
        try:
            print(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            return dest
        except Exception as e:  # try the next mirror
            last_err = e
            print(f"  failed ({type(e).__name__}: {e}); trying next mirror")
    raise RuntimeError(f"could not download {fname} from any mirror: {last_err}")


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad image magic {magic} in {path}"
        buf = f.read(n * rows * cols)
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
    return images.reshape(n, rows * cols) / 255.0  # normalize to [0, 1]


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad label magic {magic} in {path}"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_mnist():
    """Return (X_train, y_train, X_test, y_test); downloads + caches on first call."""
    X_train = _read_idx_images(_download(FILES["train_images"]))
    y_train = _read_idx_labels(_download(FILES["train_labels"]))
    X_test = _read_idx_images(_download(FILES["test_images"]))
    y_test = _read_idx_labels(_download(FILES["test_labels"]))
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"train: X {Xtr.shape} {Xtr.dtype}  y {ytr.shape} {ytr.dtype}")
    print(f"test:  X {Xte.shape} {Xte.dtype}  y {yte.shape} {yte.dtype}")
    print(f"pixel range [{Xtr.min():.3f}, {Xtr.max():.3f}]  labels {np.unique(ytr)}")
