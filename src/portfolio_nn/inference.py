from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .data import load_feature_row_from_csv, sample_synthetic_digit
from .model import NeuralNetwork


def _normalize_features(features: np.ndarray, normalization_mean: np.ndarray, normalization_std: np.ndarray) -> np.ndarray:
    return (features.reshape(1, -1) - normalization_mean) / normalization_std


def predict_from_saved_model(
    model_path: str | Path,
    *,
    source: str = "synthetic",
    label: int | None = None,
    seed: int = 42,
    csv_path: str | None = None,
    row_index: int = 0,
) -> dict:
    model, extra = NeuralNetwork.load(str(model_path))
    normalization_mean = extra.get("normalization_mean")
    normalization_std = extra.get("normalization_std")

    if normalization_mean is None or normalization_std is None:
        raise ValueError("Saved model does not contain normalization stats required for inference.")

    expected_label = None
    if source == "synthetic":
        features, expected_label = sample_synthetic_digit(label=label, seed=seed)
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path is required when source='csv'.")
        features, expected_label = load_feature_row_from_csv(csv_path, row_index=row_index)
    else:
        raise ValueError(f"Unsupported source '{source}'.")

    normalized = _normalize_features(features, normalization_mean, normalization_std)
    probabilities = model.predict_proba(normalized)[0]
    predicted_label = int(np.argmax(probabilities))

    return {
        "source": source,
        "expected_label": expected_label,
        "predicted_label": predicted_label,
        "confidence": float(probabilities[predicted_label]),
        "probabilities": {str(index): float(value) for index, value in enumerate(probabilities)},
    }


def save_prediction(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
