from __future__ import annotations

from pathlib import Path

import csv


def maybe_generate_run_visuals(output_dir: str | Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    output_path = Path(output_dir)
    generated: list[str] = []

    history_path = output_path / "training_history.csv"
    if history_path.exists():
        epochs: list[int] = []
        train_loss: list[float] = []
        val_loss: list[float] = []
        train_accuracy: list[float] = []
        val_accuracy: list[float] = []

        with history_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                train_accuracy.append(float(row["train_accuracy"]))
                val_loss.append(float(row["val_loss"])) if row["val_loss"] else val_loss.append(float("nan"))
                val_accuracy.append(float(row["val_accuracy"])) if row["val_accuracy"] else val_accuracy.append(float("nan"))

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, train_loss, label="train_loss", linewidth=2)
        if not all(np.isnan(val_loss)):
            axes[0].plot(epochs, val_loss, label="val_loss", linewidth=2)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, train_accuracy, label="train_accuracy", linewidth=2)
        if not all(np.isnan(val_accuracy)):
            axes[1].plot(epochs, val_accuracy, label="val_accuracy", linewidth=2)
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        figure.tight_layout()
        curves_path = output_path / "training_curves.png"
        figure.savefig(curves_path, dpi=150)
        plt.close(figure)
        generated.append(curves_path.name)

    confusion_matrix_path = output_path / "confusion_matrix.csv"
    if confusion_matrix_path.exists():
        matrix = np.loadtxt(confusion_matrix_path, delimiter=",")
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_title("Confusion Matrix")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                axis.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center", fontsize=8)

        figure.tight_layout()
        matrix_image_path = output_path / "confusion_matrix.png"
        figure.savefig(matrix_image_path, dpi=150)
        plt.close(figure)
        generated.append(matrix_image_path.name)

    return generated
