from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv

import numpy as np

from .data import DatasetSplit, load_dataset, split_train_validation
from .metrics import accuracy_score, classification_report, confusion_matrix, write_json, write_markdown
from .model import NeuralNetwork, TrainingResult
from .reporting import build_run_summary
from .visualization import maybe_generate_run_visuals


@dataclass
class TrainingConfig:
    dataset: str = "synthetic_digits"
    csv_path: str = "data/raw/mnist_train.csv"
    samples_per_class: int = 500
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 0.05
    validation_ratio: float = 0.1
    dropout: float = 0.0
    l2_lambda: float = 0.0
    early_stopping_patience: int = 0
    lr_decay: float = 1.0
    seed: int = 42
    hidden_sizes: tuple[int, ...] = (64, 32)
    output_dir: str | None = None
    run_name: str | None = None
    quiet: bool = False


def resolve_output_dir(output_dir: str | None, run_name: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    if run_name:
        return Path("outputs") / run_name
    return Path("outputs/latest_run")


def _save_history(path: Path, result: TrainingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"])
        for item in result.history:
            writer.writerow(
                [
                    item.epoch,
                    f"{item.train_loss:.6f}",
                    f"{item.train_accuracy:.6f}",
                    "" if item.val_loss is None else f"{item.val_loss:.6f}",
                    "" if item.val_accuracy is None else f"{item.val_accuracy:.6f}",
                    f"{item.learning_rate:.8f}",
                ]
            )


def _save_confusion_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.tolist())


def _final_metrics(split: DatasetSplit, model: NeuralNetwork) -> tuple[float, float]:
    probabilities = model.predict_proba(split.features)
    predictions = np.argmax(probabilities, axis=1)
    return model.compute_loss(split.labels, probabilities), accuracy_score(split.labels, predictions)


def run_training(config: TrainingConfig) -> dict:
    dataset = load_dataset(
        config.dataset,
        samples_per_class=config.samples_per_class,
        csv_path=config.csv_path,
        seed=config.seed,
    )
    train_split, val_split = split_train_validation(dataset.train, config.validation_ratio, seed=config.seed)

    model = NeuralNetwork(
        input_size=dataset.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=dataset.num_classes,
        learning_rate=config.learning_rate,
        dropout_rate=config.dropout,
        l2_lambda=config.l2_lambda,
        seed=config.seed,
    )

    result = model.fit(
        train_split.features,
        train_split.labels,
        X_val=None if val_split is None else val_split.features,
        y_val=None if val_split is None else val_split.labels,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=not config.quiet,
        early_stopping_patience=config.early_stopping_patience,
        lr_decay=config.lr_decay,
        seed=config.seed,
    )

    output_dir = resolve_output_dir(config.output_dir, config.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_predictions = model.predict(dataset.test.features)
    test_probabilities = model.predict_proba(dataset.test.features)
    test_accuracy = accuracy_score(dataset.test.labels, test_predictions)
    test_loss = model.compute_loss(dataset.test.labels, test_probabilities)
    test_matrix = confusion_matrix(dataset.test.labels, test_predictions, dataset.num_classes)
    test_report = classification_report(dataset.test.labels, test_predictions, dataset.label_names)

    final_train_loss, final_train_accuracy = _final_metrics(train_split, model)
    final_val_loss = None
    final_val_accuracy = None
    if val_split is not None:
        final_val_loss, final_val_accuracy = _final_metrics(val_split, model)

    model.save(
        str(output_dir / "model_weights.npz"),
        extra={
            "normalization_mean": dataset.normalization_mean,
            "normalization_std": dataset.normalization_std,
        },
    )
    _save_history(output_dir / "training_history.csv", result)
    _save_confusion_matrix(output_dir / "confusion_matrix.csv", test_matrix)
    (output_dir / "classification_report.csv").write_text(test_report + "\n", encoding="utf-8")

    summary = {
        "run_name": config.run_name or output_dir.name,
        "dataset": config.dataset,
        "samples_per_class": config.samples_per_class if config.dataset == "synthetic_digits" else None,
        "epochs_requested": config.epochs,
        "epochs_completed": len(result.history),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "lr_decay": config.lr_decay,
        "dropout": config.dropout,
        "l2_lambda": config.l2_lambda,
        "validation_ratio": config.validation_ratio,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_triggered": result.stopped_early,
        "hidden_sizes": list(config.hidden_sizes),
        "train_size": int(train_split.features.shape[0]),
        "validation_size": 0 if val_split is None else int(val_split.features.shape[0]),
        "test_size": int(dataset.test.features.shape[0]),
        "best_epoch": result.best_epoch,
        "best_monitored_loss": round(result.best_loss, 6),
        "final_train_loss": round(final_train_loss, 6),
        "final_train_accuracy": round(final_train_accuracy, 4),
        "final_val_loss": None if final_val_loss is None else round(final_val_loss, 6),
        "final_val_accuracy": None if final_val_accuracy is None else round(final_val_accuracy, 4),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 4),
    }
    generated_visuals = maybe_generate_run_visuals(output_dir)
    summary["generated_visuals"] = generated_visuals

    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "training_config.json", asdict(config))
    write_markdown(output_dir / "run_summary.md", build_run_summary(summary))

    return {
        "summary": summary,
        "output_dir": output_dir,
        "confusion_matrix": test_matrix,
    }
