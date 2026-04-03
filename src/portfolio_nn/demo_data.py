from __future__ import annotations

from pathlib import Path
import csv
import json


def list_run_directories(root: str | Path = "outputs") -> list[Path]:
    output_root = Path(root)
    if not output_root.exists():
        return []
    candidates = []
    for path in output_root.rglob("metrics.json"):
        candidates.append(path.parent)
    return sorted(candidates)


def load_metrics(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def load_training_history(run_dir: str | Path) -> list[dict]:
    history_path = Path(run_dir) / "training_history.csv"
    with history_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_confusion_matrix(run_dir: str | Path) -> list[list[int]]:
    matrix_path = Path(run_dir) / "confusion_matrix.csv"
    with matrix_path.open("r", encoding="utf-8") as handle:
        return [[int(value) for value in row] for row in csv.reader(handle)]


def load_classification_report(run_dir: str | Path) -> str:
    return (Path(run_dir) / "classification_report.csv").read_text(encoding="utf-8")
