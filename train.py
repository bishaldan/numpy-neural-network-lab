from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from portfolio_nn.workflow import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy neural network with portfolio-ready ML artifacts.")
    parser.add_argument("--dataset", choices=["synthetic_digits", "mnist_csv"], default="synthetic_digits")
    parser.add_argument("--csv-path", default="data/raw/mnist_train.csv")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        dataset=args.dataset,
        csv_path=args.csv_path,
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        dropout=args.dropout,
        l2_lambda=args.l2_lambda,
        early_stopping_patience=args.early_stopping_patience,
        lr_decay=args.lr_decay,
        seed=args.seed,
        hidden_sizes=tuple(args.hidden_sizes),
        output_dir=args.output_dir,
        run_name=args.run_name,
        quiet=args.quiet,
    )

    result = run_training(config)
    summary = result["summary"]
    print(f"Run name: {summary['run_name']}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Artifacts saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
