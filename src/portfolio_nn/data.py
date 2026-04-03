from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetSplit:
    features: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetBundle:
    train: DatasetSplit
    test: DatasetSplit
    input_size: int
    num_classes: int
    label_names: list[str]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray


def _digit_templates() -> dict[int, np.ndarray]:
    raw_templates = {
        0: [
            "01110",
            "10001",
            "10011",
            "10101",
            "11001",
            "10001",
            "01110",
        ],
        1: [
            "00100",
            "01100",
            "00100",
            "00100",
            "00100",
            "00100",
            "01110",
        ],
        2: [
            "01110",
            "10001",
            "00001",
            "00010",
            "00100",
            "01000",
            "11111",
        ],
        3: [
            "11110",
            "00001",
            "00001",
            "01110",
            "00001",
            "00001",
            "11110",
        ],
        4: [
            "00010",
            "00110",
            "01010",
            "10010",
            "11111",
            "00010",
            "00010",
        ],
        5: [
            "11111",
            "10000",
            "11110",
            "00001",
            "00001",
            "10001",
            "01110",
        ],
        6: [
            "00110",
            "01000",
            "10000",
            "11110",
            "10001",
            "10001",
            "01110",
        ],
        7: [
            "11111",
            "00001",
            "00010",
            "00100",
            "01000",
            "01000",
            "01000",
        ],
        8: [
            "01110",
            "10001",
            "10001",
            "01110",
            "10001",
            "10001",
            "01110",
        ],
        9: [
            "01110",
            "10001",
            "10001",
            "01111",
            "00001",
            "00010",
            "11100",
        ],
    }
    return {
        label: np.array([[int(pixel) for pixel in row] for row in rows], dtype=np.float64)
        for label, rows in raw_templates.items()
    }


def _augment_digit(template: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = template.copy()
    padded = np.pad(image, 1, mode="constant")
    row_shift = int(rng.integers(-1, 2))
    col_shift = int(rng.integers(-1, 2))
    shifted = padded[1 + row_shift : 1 + row_shift + image.shape[0], 1 + col_shift : 1 + col_shift + image.shape[1]]

    noise_mask = rng.random(shifted.shape) < 0.04
    noisy = np.where(noise_mask, 1.0 - shifted, shifted)
    brightness = rng.uniform(0.85, 1.15)
    sample = np.clip(noisy * brightness, 0.0, 1.0)
    return sample


def _standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def make_synthetic_digit_dataset(
    samples_per_class: int = 500,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    templates = _digit_templates()

    features: list[np.ndarray] = []
    labels: list[int] = []

    for label, template in templates.items():
        for _ in range(samples_per_class):
            sample = _augment_digit(template, rng)
            features.append(sample.reshape(-1))
            labels.append(label)

    X = np.vstack(features)
    y = np.array(labels, dtype=np.int64)

    indices = rng.permutation(len(X))
    split_index = int(len(X) * (1.0 - test_ratio))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    X_train, X_test, mean, std = _standardize_train_test(X_train, X_test)

    return DatasetBundle(
        train=DatasetSplit(features=X_train, labels=y[train_idx]),
        test=DatasetSplit(features=X_test, labels=y[test_idx]),
        input_size=X_train.shape[1],
        num_classes=10,
        label_names=[str(label) for label in range(10)],
        normalization_mean=mean,
        normalization_std=std,
    )


def load_mnist_csv(path: str | Path, test_ratio: float = 0.2, seed: int = 42) -> DatasetBundle:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV dataset not found at {csv_path}. Place a Kaggle-style MNIST train.csv file there first."
        )

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    y = data[:, 0].astype(np.int64)
    X = data[:, 1:].astype(np.float64) / 255.0

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split_index = int(len(X) * (1.0 - test_ratio))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    X_train, X_test, mean, std = _standardize_train_test(X[train_idx], X[test_idx])

    return DatasetBundle(
        train=DatasetSplit(features=X_train, labels=y[train_idx]),
        test=DatasetSplit(features=X_test, labels=y[test_idx]),
        input_size=X.shape[1],
        num_classes=10,
        label_names=[str(label) for label in range(10)],
        normalization_mean=mean,
        normalization_std=std,
    )


def load_dataset(
    dataset_name: str,
    *,
    samples_per_class: int = 500,
    csv_path: str | None = None,
    seed: int = 42,
) -> DatasetBundle:
    if dataset_name == "synthetic_digits":
        return make_synthetic_digit_dataset(samples_per_class=samples_per_class, seed=seed)
    if dataset_name == "mnist_csv":
        if not csv_path:
            raise ValueError("csv_path is required when dataset_name='mnist_csv'.")
        return load_mnist_csv(csv_path, seed=seed)
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def split_train_validation(
    split: DatasetSplit,
    validation_ratio: float,
    seed: int = 42,
) -> tuple[DatasetSplit, DatasetSplit | None]:
    if validation_ratio <= 0.0:
        return split, None
    if validation_ratio >= 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(split.features.shape[0])
    val_size = max(1, int(split.features.shape[0] * validation_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_split = DatasetSplit(features=split.features[train_indices], labels=split.labels[train_indices])
    val_split = DatasetSplit(features=split.features[val_indices], labels=split.labels[val_indices])
    return train_split, val_split


def sample_synthetic_digit(
    label: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    chosen_label = int(rng.integers(0, 10)) if label is None else label
    template = _digit_templates()[chosen_label]
    return _augment_digit(template, rng).reshape(-1), chosen_label


def load_feature_row_from_csv(path: str | Path, row_index: int = 0) -> tuple[np.ndarray, int | None]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV input not found at {csv_path}.")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if row_index < 0 or row_index >= len(data):
        raise IndexError(f"row_index {row_index} is out of bounds for {csv_path}.")

    row = data[row_index]
    if row.shape[0] == 785:
        return row[1:].astype(np.float64) / 255.0, int(row[0])
    return row.astype(np.float64), None
