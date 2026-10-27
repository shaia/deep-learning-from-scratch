"""
Tiny Shakespeare download + loader, shared by the modules that model text one
character at a time (Module 06 RNN onward).

Tiny Shakespeare is 1,115,394 characters -- about 40,000 lines -- of
Shakespeare's plays concatenated into one plain-text file, speaker names and
all. Andrej Karpathy assembled it for char-rnn (2015), and it has been the
standard laptop-sized benchmark for character-level language models ever
since. Only 65 distinct characters occur: newline, space, a dozen punctuation
marks, the digit 3, and the upper- and lower-case alphabets.

Downloads the file once into data/shakespeare/ (gitignored) and verifies its
SHA-256. The text is split the way nanoGPT and most later work split it: the
first 90% for training, the last 10% for validation -- contiguous, so the
validation text is passages the model never saw, not shuffled lines.

Use it from anywhere in the repo:

    import sys, os
    sys.path.insert(0, os.path.join(REPO_ROOT, "data"))
    from get_shakespeare import load_shakespeare
    train_ids, val_ids, chars = load_shakespeare()
    text = "".join(chars[i] for i in train_ids[:100])

Run directly to fetch + report sizes:
    python data/get_shakespeare.py
"""

import hashlib
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "shakespeare")
TXT = os.path.join(CACHE_DIR, "input.txt")

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Checked on every load, so a truncated or substituted download fails loudly.
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

TRAIN_FRACTION = 0.9


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download() -> str:
    """Fetch the text into CACHE_DIR if not already present; return its path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not (os.path.exists(TXT) and os.path.getsize(TXT) > 0):
        print(f"downloading {URL} (~1.1 MB, once)")
        tmp = TXT + ".part"
        with urllib.request.urlopen(URL, timeout=120) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        os.replace(tmp, TXT)
    digest = _sha256(TXT)
    if digest != SHA256:
        os.remove(TXT)                  # so the next call downloads afresh
        raise RuntimeError(f"{TXT}: SHA-256 {digest} != expected {SHA256}; "
                           "removed it -- re-run to download again")
    return TXT


def load_text() -> str:
    """The whole corpus as one string (downloads on first call)."""
    with open(_download(), encoding="utf-8") as f:
        return f.read()


def load_shakespeare():
    """Return (train_ids, val_ids, chars).

    chars is the sorted tuple of the 65 distinct characters -- the vocabulary;
    a character's id is its index there. train_ids / val_ids are int64 arrays
    of ids: the first 90% of the text and the last 10%.
    """
    text = load_text()
    chars = tuple(sorted(set(text)))
    index = {c: i for i, c in enumerate(chars)}
    ids = np.array([index[c] for c in text], dtype=np.int64)
    cut = int(TRAIN_FRACTION * len(ids))
    return ids[:cut], ids[cut:], chars


if __name__ == "__main__":
    train_ids, val_ids, chars = load_shakespeare()
    print(f"vocabulary: {len(chars)} characters {''.join(chars)!r}")
    print(f"train: {train_ids.size:,} characters   val: {val_ids.size:,} characters")
    print("first 200 training characters:")
    print("".join(chars[i] for i in train_ids[:200]))
