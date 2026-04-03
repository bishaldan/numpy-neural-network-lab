from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> str:
    lines = ["label,precision,recall,f1,support"]
    for index, label_name in enumerate(label_names):
        true_positive = np.sum((y_true == index) & (y_pred == index))
        false_positive = np.sum((y_true != index) & (y_pred == index))
        false_negative = np.sum((y_true == index) & (y_pred != index))
        support = np.sum(y_true == index)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        lines.append(f"{label_name},{precision:.4f},{recall:.4f},{f1:.4f},{int(support)}")
    return "\n".join(lines)


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: str | Path, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
