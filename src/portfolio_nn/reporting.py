from __future__ import annotations


def build_run_summary(summary: dict) -> str:
    samples_per_class = summary.get("samples_per_class")
    samples_line = (
        f"- Samples per class: {samples_per_class}"
        if samples_per_class is not None
        else "- Samples per class: n/a"
    )

    hidden_sizes = ", ".join(str(size) for size in summary["hidden_sizes"])
    visuals = summary.get("generated_visuals", [])
    visuals_lines = [f"- `{name}`" for name in visuals] if visuals else ["- No image artifacts generated in this environment."]

    return "\n".join(
        [
            "# Training Run Summary",
            "",
            "## Overview",
            f"- Run name: {summary['run_name']}",
            f"- Dataset: {summary['dataset']}",
            f"- Train size: {summary['train_size']}",
            f"- Validation size: {summary['validation_size']}",
            f"- Test size: {summary['test_size']}",
            f"- Best epoch: {summary['best_epoch']}",
            f"- Test loss: {summary['test_loss']:.4f}",
            f"- Test accuracy: {summary['test_accuracy']:.4f}",
            "",
            "## Hyperparameters",
            f"- Epochs requested: {summary['epochs_requested']}",
            f"- Epochs completed: {summary['epochs_completed']}",
            f"- Batch size: {summary['batch_size']}",
            f"- Learning rate: {summary['learning_rate']}",
            f"- Learning-rate decay: {summary['lr_decay']}",
            f"- Dropout: {summary['dropout']}",
            f"- L2 lambda: {summary['l2_lambda']}",
            f"- Validation ratio: {summary['validation_ratio']}",
            f"- Early stopping patience: {summary['early_stopping_patience']}",
            f"- Early stopping triggered: {summary['early_stopping_triggered']}",
            f"- Hidden sizes: {hidden_sizes}",
            samples_line,
            "",
            "## Final Metrics",
            f"- Final train loss: {summary['final_train_loss']:.4f}",
            f"- Final train accuracy: {summary['final_train_accuracy']:.4f}",
            (
                "- Final validation loss: n/a"
                if summary['final_val_loss'] is None
                else f"- Final validation loss: {summary['final_val_loss']:.4f}"
            ),
            (
                "- Final validation accuracy: n/a"
                if summary['final_val_accuracy'] is None
                else f"- Final validation accuracy: {summary['final_val_accuracy']:.4f}"
            ),
            f"- Best monitored loss: {summary['best_monitored_loss']:.4f}",
            "",
            "## Generated Artifacts",
            "- `metrics.json`",
            "- `training_config.json`",
            "- `training_history.csv`",
            "- `classification_report.csv`",
            "- `confusion_matrix.csv`",
            "- `model_weights.npz`",
            "- `run_summary.md`",
            "",
            "## Generated Visuals",
            *visuals_lines,
            "",
            "## Portfolio Angle",
            "- Neural network implemented from scratch with NumPy.",
            "- Advanced training workflow includes validation monitoring, regularization, and early stopping.",
            "- Reproducible CLI workflow suitable for Dockerized ML demos.",
            "- Reviewable output artifacts support GitHub screenshots and project discussion.",
            "",
        ]
    )


def build_experiment_comparison_markdown(rows: list[dict]) -> str:
    lines = [
        "# Experiment Comparison",
        "",
        "| Run | Best Epoch | Test Accuracy | Test Loss | Dropout | L2 | LR Decay | Early Stop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {best_epoch} | {test_accuracy:.4f} | {test_loss:.4f} | {dropout:.2f} | {l2_lambda:.5f} | {lr_decay:.4f} | {early_stopping_triggered} |".format(
                **row
            )
        )

    best_row = max(rows, key=lambda item: item["test_accuracy"])
    lines.extend(
        [
            "",
            "## Best Run",
            f"- Run: {best_row['run_name']}",
            f"- Test accuracy: {best_row['test_accuracy']:.4f}",
            f"- Best epoch: {best_row['best_epoch']}",
        ]
    )
    return "\n".join(lines) + "\n"
