"""Data loading and a clearly labelled local final-test evaluation."""
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parent


def load_file(name):
    if name not in {"train", "extra", "test"}:
        raise ValueError("Use train, extra, or test")
    with np.load(ROOT / "assets" / f"{name}.npz") as data:
        return data["X"], data["y"]


def image_features(images):
    """Fixed 4x4 block averages: (n,3,128,128) -> (n,3072), scaled to [0,1].

    This transformation has no fitted parameters and is safe before splitting.
    Every model, including final evaluation, uses exactly this representation.
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
        raise ValueError("Expected (n, 3, 128, 128) uint8 RGB images")
    return (images.reshape(-1, 3, 32, 4, 32, 4)
            .mean(axis=(3, 5), dtype=np.float32).reshape(len(images), -1) / 255.0)


def scores(model, X, y):
    pred = model.predict(X)
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "recall_no_mask": recall_score(y, pred, pos_label=1, zero_division=0)}


def evaluate(model):
    """Call only after selecting and freezing the final model in Question 29.

    This is an accessible local test set, not a hidden grading service.
    """
    images, y = load_file("test")
    result = scores(model, image_features(images), y)
    print("Final test metrics:", result)
    return result
