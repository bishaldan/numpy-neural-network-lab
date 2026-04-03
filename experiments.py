from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from portfolio_nn.metrics import write_markdown
from portfolio_nn.reporting import build_experiment_comparison_markdown
from portfolio_nn.workflow import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a curated set of NumPy neural network experiments.")
    parser.add_argument("--dataset", choices=["synthetic_digits", "mnist_csv"], default="synthetic_digits")
    parser.add_argument("--csv-path", default="data/raw/mnist_train.csv")
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/experiments/v2_showcase")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def curated_configs(args: argparse.Namespace) -> list[TrainingConfig]:
    output_root = Path(args.output_root)
    return [
        TrainingConfig(
            dataset=args.dataset,
            csv_path=args.csv_path,
            samples_per_class=args.samples_per_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_ratio=0.1,
            seed=args.seed,
            run_name="baseline",
            output_dir=str(output_root / "baseline"),
            quiet=args.quiet,
        ),
        TrainingConfig(
            dataset=args.dataset,
            csv_path=args.csv_path,
            samples_per_class=args.samples_per_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_ratio=0.15,
            dropout=0.15,
            seed=args.seed,
            run_name="dropout_regularized",
            output_dir=str(output_root / "dropout_regularized"),
            quiet=args.quiet,
        ),
        TrainingConfig(
            dataset=args.dataset,
            csv_path=args.csv_path,
            samples_per_class=args.samples_per_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_ratio=0.15,
            l2_lambda=0.0005,
            early_stopping_patience=8,
            seed=args.seed,
            run_name="l2_early_stopping",
            output_dir=str(output_root / "l2_early_stopping"),
            quiet=args.quiet,
        ),
        TrainingConfig(
            dataset=args.dataset,
            csv_path=args.csv_path,
            samples_per_class=args.samples_per_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_ratio=0.15,
            dropout=0.10,
            l2_lambda=0.0005,
            lr_decay=0.985,
            early_stopping_patience=10,
            seed=args.seed,
            run_name="scheduled_robust",
            output_dir=str(output_root / "scheduled_robust"),
            quiet=args.quiet,
        ),
    ]


def save_comparison_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_name",
        "best_epoch",
        "test_accuracy",
        "test_loss",
        "dropout",
        "l2_lambda",
        "lr_decay",
        "early_stopping_triggered",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fieldnames} for row in rows])


def main() -> None:
    args = parse_args()
    rows = []
    for config in curated_configs(args):
        result = run_training(config)
        rows.append(result["summary"])
        print(f"Completed {result['summary']['run_name']} with test_accuracy={result['summary']['test_accuracy']:.4f}")

    output_root = Path(args.output_root)
    save_comparison_csv(output_root / "experiment_comparison.csv", rows)
    write_markdown(output_root / "experiment_comparison.md", build_experiment_comparison_markdown(rows))
    best_run = max(rows, key=lambda item: item["test_accuracy"])
    print(f"Best run: {best_run['run_name']} ({best_run['test_accuracy']:.4f})")


if __name__ == "__main__":
    main()
