"""Inspect the CMU-MOSI aligned_50.pkl feature file used by EMOE.

The file is only unpickled after its SHA-256 matches the value published by
MMSA, because unpickling an untrusted file can execute arbitrary code.
"""
import hashlib
import pickle
import sys
from collections import Counter

import numpy as np

PKL_PATH = "dataset/MOSI/Processed/aligned_50.pkl"
EXPECTED_SHA256 = "d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9"


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(value):
    if isinstance(value, np.ndarray):
        return f"shape={value.shape}, dtype={value.dtype}"
    if isinstance(value, (list, tuple)):
        inner = type(value[0]).__name__ if len(value) else "-"
        return f"{type(value).__name__} of {len(value)} x {inner}"
    return type(value).__name__


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PKL_PATH

    # (a) checksum before opening
    digest = sha256_of(path)
    print(f"SHA-256  : {digest}")
    print(f"expected : {EXPECTED_SHA256}")
    if digest != EXPECTED_SHA256:
        sys.exit("Checksum MISMATCH - refusing to unpickle this file.")
    print("Checksum OK\n")

    with open(path, "rb") as f:
        data = pickle.load(f)

    # (b) keys, shapes and dtypes per split
    for split in ("train", "valid", "test"):
        print(f"=== {split} ===")
        for key, value in data[split].items():
            print(f"  {key:25s} {describe(value)}")
        print()

    # (c) label statistics per split
    for split in ("train", "valid", "test"):
        labels = np.asarray(data[split]["regression_labels"], dtype=np.float64).reshape(-1)
        rounded = np.clip(np.round(labels), -3, 3).astype(int)
        counts = Counter(rounded.tolist())
        print(f"=== {split} labels (N={len(labels)}) ===")
        print(f"  range           : [{labels.min():.4f}, {labels.max():.4f}]")
        print("  rounded score   : " + "  ".join(f"{s:+d}:{counts.get(s, 0)}" for s in range(-3, 4)))
        print(f"  exactly 0.0     : {int(np.sum(labels == 0))}")
        print()

    # (d) one training clip
    train = data["train"]
    i = 0
    print("=== example training clip (index 0) ===")
    print(f"  id        : {train['id'][i]}")
    print(f"  sentence  : {train['raw_text'][i]}")
    print(f"  label     : {float(np.asarray(train['regression_labels'][i]).reshape(-1)[0]):.4f}")
    tb = train["text_bert"][i]
    print(f"  text_bert row 0 (first 12): {tb[0][:12].astype(int).tolist()}")
    print(f"  text_bert row 1 (first 12): {tb[1][:12].astype(int).tolist()}")
    print(f"  text_bert row 2 (first 12): {tb[2][:12].astype(int).tolist()}")
    print(f"  non-padding tokens        : {int(tb[1].sum())} of {tb.shape[1]}")


if __name__ == "__main__":
    main()
