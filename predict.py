from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from portfolio_nn.inference import predict_from_saved_model, save_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a saved NumPy neural network model.")
    parser.add_argument("--model-path", default="outputs/latest_run/model_weights.npz")
    parser.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = predict_from_saved_model(
        args.model_path,
        source=args.source,
        label=args.label,
        seed=args.seed,
        csv_path=args.csv_path,
        row_index=args.row_index,
    )

    print(f"Predicted label: {prediction['predicted_label']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    if prediction["expected_label"] is not None:
        print(f"Expected label: {prediction['expected_label']}")

    if args.output_json:
        save_prediction(args.output_json, prediction)
        print(f"Prediction saved to: {args.output_json}")


if __name__ == "__main__":
    main()
