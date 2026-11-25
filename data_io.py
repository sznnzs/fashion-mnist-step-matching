"""Read the published Fashion-MNIST IDX files and construct one fixed split."""
import gzip
import hashlib
import os
from pathlib import Path
import struct
import tempfile
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SPLIT_SEED = 6704
COMMIT = "b2617bb6d3ffa2e429640350f613e3291e10b141"
DIGESTS = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def fetch():
    DATA.mkdir(exist_ok=True)
    bases = [
        f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{COMMIT}/data/fashion",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets",
    ]
    for name, expected in DIGESTS.items():
        path = DATA / name
        if path.is_file() and hashlib.md5(path.read_bytes()).hexdigest() == expected:
            continue
        errors = []
        for base in bases:
            try:
                with urlopen(f"{base}/{name}", timeout=45) as response:
                    body = response.read()
                if hashlib.md5(body).hexdigest() != expected:
                    raise ValueError("published MD5 mismatch")
                with tempfile.NamedTemporaryFile(dir=DATA, delete=False) as tmp:
                    tmp.write(body)
                os.replace(tmp.name, path)
                print(f"Verified {name} ({len(body):,} bytes)", flush=True)
                break
            except (OSError, ValueError) as error:
                errors.append(str(error))
        else:
            raise RuntimeError(f"{name}: {errors}")


def read_idx(name):
    raw = (DATA / name).read_bytes()
    if hashlib.md5(raw).hexdigest() != DIGESTS[name]:
        raise ValueError(f"Checksum mismatch: {name}")
    buf = gzip.decompress(raw)
    magic, count = struct.unpack_from(">II", buf)
    if magic == 2051:
        rows, cols = struct.unpack_from(">II", buf, 8)
        assert (rows, cols) == (28, 28)
        return np.frombuffer(buf, np.uint8, offset=16).reshape(count, rows * cols)
    assert magic == 2049 and len(buf) == count + 8
    return np.frombuffer(buf, np.uint8, offset=8).astype(np.int64)


def split(labels):
    rng = np.random.default_rng(SPLIT_SEED)
    valid = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == category), 600, replace=False)
        for category in range(10)
    ]))
    train = np.setdiff1d(np.arange(len(labels)), valid)
    assert np.all(np.bincount(labels[train]) == 5400)
    return train, valid


def load_arrays():
    images = read_idx("train-images-idx3-ubyte.gz")
    labels = read_idx("train-labels-idx1-ubyte.gz")
    train, valid = split(labels)
    train_images = images[train]
    meta = {
        "split_seed": SPLIT_SEED,
        "train_indices_sha256": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
        "valid_indices_sha256": hashlib.sha256(valid.astype("<i8").tobytes()).hexdigest(),
        "train_pixel_mean": float(train_images.mean()),
        "train_pixel_std": float(train_images.std()),
    }
    arrays = (train_images, labels[train], images[valid], labels[valid])
    return arrays, meta


if __name__ == "__main__":
    fetch()
