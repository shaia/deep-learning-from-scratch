"""
CIFAR-10 download + loader, shared by the modules that train on color photos
(Module 04 AlexNet onward).

CIFAR-10 (Krizhevsky, 2009) is 60,000 32x32 RGB images in 10 classes -- 50,000
for training, 10,000 for test -- cut from the "80 Million Tiny Images"
collection. It is the small-photo stand-in for ImageNet that this book can
train on a laptop in NumPy.

Downloads the official python-format archive once into data/cifar10/
(gitignored), verifies its MD5, converts the pickled batches into one
data/cifar10/cifar10.npz cache, and loads from that cache afterwards.

Images come back as **uint8** in NCHW layout, shape (n, 3, 32, 32): the
archive already stores each image as its R plane, then G, then B, so a plain
reshape yields channels-first with no transpose. They stay uint8 on purpose --
50,000 images as float64 would be 1.2 GB; convert batch by batch instead (see
`to_float`). Labels are int64 in [0, 10).

Use it from anywhere in the repo:

    import sys, os
    sys.path.insert(0, os.path.join(REPO_ROOT, "data"))
    from get_cifar10 import load_cifar10, CLASSES
    X_train, y_train, X_test, y_test = load_cifar10()

Run directly to fetch + report shapes:
    python data/get_cifar10.py
"""

import hashlib
import os
import pickle
import tarfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cifar10")
ARCHIVE = os.path.join(CACHE_DIR, "cifar-10-python.tar.gz")
NPZ = os.path.join(CACHE_DIR, "cifar10.npz")

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# The archive's published MD5 (the same value torchvision pins). Checked before
# the pickles inside are ever loaded, so a corrupted or substituted download
# fails loudly instead of being unpickled.
MD5 = "c58f30108f718f92721af3b95e74349a"

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download() -> str:
    """Fetch the archive into CACHE_DIR if not already present; return its path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not (os.path.exists(ARCHIVE) and os.path.getsize(ARCHIVE) > 0):
        print(f"downloading {URL} (~170 MB, once)")
        tmp = ARCHIVE + ".part"
        with urllib.request.urlopen(URL, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, ARCHIVE)
    digest = _md5(ARCHIVE)
    if digest != MD5:
        os.remove(ARCHIVE)              # so the next call downloads afresh
        raise RuntimeError(f"{ARCHIVE}: MD5 {digest} != expected {MD5}; "
                           "removed it -- re-run to download again")
    return ARCHIVE


def _read_batches(archive: str):
    """Read the five training batches + the test batch straight out of the
    tarball (no extraction to disk). Each batch is a pickled dict holding
    b'data' (10000, 3072) uint8 and b'labels' (a list of 10000 ints)."""
    batches = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if not (name.startswith("data_batch_") or name == "test_batch"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                raise RuntimeError(f"{archive}: {member.name} is not a file")
            # The MD5 check above vouches for these bytes before unpickling.
            batch = pickle.load(fh, encoding="bytes")
            X = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
            y = np.asarray(batch[b"labels"], dtype=np.int64)
            batches[name] = (X, y)
    train = [batches[f"data_batch_{k}"] for k in range(1, 6)]
    Xtr = np.concatenate([X for X, _ in train])
    ytr = np.concatenate([y for _, y in train])
    Xte, yte = batches["test_batch"]
    return Xtr, ytr, Xte, yte


def load_cifar10():
    """Return (X_train, y_train, X_test, y_test); downloads + caches on first call.

    X_* are uint8 (n, 3, 32, 32); y_* are int64 class indices into CLASSES.
    """
    if not os.path.exists(NPZ):
        Xtr, ytr, Xte, yte = _read_batches(_download())
        # Save under a temporary name, then rename: an interrupted save must
        # not leave a truncated cache that every later call trips over.
        tmp = NPZ[:-len(".npz")] + ".tmp.npz"
        np.savez(tmp, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        os.replace(tmp, NPZ)
    with np.load(NPZ, allow_pickle=False) as z:
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]


def to_float(X_uint8, mean=None, std=None, dtype=np.float64):
    """uint8 images -> floats in [0, 1], optionally standardized per channel.

    mean/std are length-3 arrays (per RGB channel), typically measured on the
    training set; pass them to center the data the way every CIFAR pipeline
    does. Convert a batch at a time -- the whole set as float64 is 1.2 GB.
    """
    X = X_uint8.astype(dtype) / 255.0
    if mean is not None:
        X = (X - np.asarray(mean, dtype=dtype)[None, :, None, None]) \
            / np.asarray(std, dtype=dtype)[None, :, None, None]
    return X


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_cifar10()
    print(f"train: X {Xtr.shape} {Xtr.dtype}  y {ytr.shape} {ytr.dtype}")
    print(f"test:  X {Xte.shape} {Xte.dtype}  y {yte.shape} {yte.dtype}")
    counts = np.bincount(ytr, minlength=10)
    print("per-class train counts:", dict(zip(CLASSES, counts.tolist())))
    mean = Xtr.mean(axis=(0, 2, 3)) / 255.0
    std = (Xtr / 255.0).std(axis=(0, 2, 3))
    print(f"channel mean {np.round(mean, 4)}  std {np.round(std, 4)}")
