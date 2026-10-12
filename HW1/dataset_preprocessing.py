import argparse
import hashlib
import io
import json
from pathlib import Path
from functools import lru_cache
from zipfile import ZipFile

import numpy as np
from PIL import Image
from datasets import Dataset, Image as DatasetImage, load_dataset

SOURCE = "DamarJati/Face-Mask-Detection"
REVISION = "f3eebe9b4c4f4ef28c7ff8b3b20bf77eb030272d"
SEED = 2203


def pixels(image):
    image = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    return np.asarray(image).transpose(2, 0, 1)


@lru_cache(maxsize=4)
def archive(path):
    # Reuse the ZIP directory instead of reopening a 12,000-image archive per row.
    return ZipFile(path)


def row_pixels(value):
    if value["bytes"] is not None:
        source = io.BytesIO(value["bytes"])
    elif value["path"].startswith("zip://"):
        member, path = value["path"][6:].split("::", 1)
        source = io.BytesIO(archive(path).read(member))
    else:
        source = value["path"]
    with Image.open(source) as image:
        return pixels(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arrow", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.source_arrow:
        dataset = Dataset.from_file(str(args.source_arrow), in_memory=True)
    else:
        dataset = load_dataset(SOURCE, split="train", revision=REVISION, keep_in_memory=True,
                               cache_dir=str(root / ".cache" / "datasets"))
    if dataset.features["label"].names != ["WithMask", "WithoutMask"]:
        raise ValueError("Unexpected label mapping; inspect source before regenerating")
    dataset = dataset.cast_column("image", DatasetImage(decode=False))
    # Preserve the original 80/20 split, 1/50 small train set, and class-0 extra.
    source_rows = len(dataset)
    dataset = dataset.add_column("source_id", list(range(len(dataset))))
    # The source has repeated images, including across its original folders.
    # Remove exact resized-image duplicates before any split or CV can leak them.
    seen, unique_indices = {}, []
    for i, row in enumerate(dataset):
        digest = hashlib.sha256(row_pixels(row["image"]).tobytes()).digest()
        if digest in seen:
            if seen[digest] != row["label"]:
                raise ValueError("Identical image has conflicting labels; inspect source")
        else:
            seen[digest] = row["label"]
            unique_indices.append(i)
        if (i+1) % 2000 == 0:
            print(f"Checked {i+1}/{source_rows} source images", flush=True)
    dataset = dataset.select(unique_indices, keep_in_memory=True)
    print(f"Source rows: {source_rows}; unique resized images: {len(dataset)}", flush=True)
    split = dataset.train_test_split(test_size=0.2, seed=SEED,
                                    keep_in_memory=True)
    split = split.shuffle(seed=SEED, keep_in_memory=True)
    train = split["train"]
    n_small = len(train) // 50
    remaining = train.select(range(n_small, len(train)), keep_in_memory=True)
    extra_indices = np.flatnonzero(np.asarray(remaining["label"]) == 0)
    subsets = {
        "train": train.select(range(n_small), keep_in_memory=True),
        "extra": remaining.select(extra_indices, keep_in_memory=True),
        "test": split["test"],
    }
    out = root / "assets"
    out.mkdir(exist_ok=True)
    manifest = {"source": SOURCE, "revision": REVISION, "seed": SEED,
                "source_rows": source_rows, "unique_images": len(dataset),
                "duplicates_removed": source_rows-len(dataset),
                "deduplication": "SHA256 of 128x128 RGB uint8 pixels, before splitting",
                "labels": {"0": "WithMask", "1": "WithoutMask"},
                "image_shape": [3, 128, 128], "dtype": "uint8", "files": {}}
    for name, subset in subsets.items():
        X = np.empty((len(subset), 3, 128, 128), dtype=np.uint8)
        for i, row in enumerate(subset):
            X[i] = row_pixels(row["image"])
        y = np.asarray(subset["label"], dtype=np.int64)
        ids = np.asarray(subset["source_id"], dtype=np.int64)
        path = out / f"{name}.npz"
        np.savez_compressed(path, X=X, y=y, source_id=ids)
        manifest["files"][path.name] = {
            "n": len(y), "class_counts": np.bincount(y, minlength=2).tolist(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(name, X.shape, manifest["files"][path.name], flush=True)
    # Deliberately do not expose train_full: it bypasses the extra-data exercise.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
