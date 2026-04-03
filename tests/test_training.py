from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from portfolio_nn.demo_data import load_confusion_matrix, load_metrics, load_training_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = PROJECT_ROOT / "outputs" / "test_runs"
        if self.output_root.exists():
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def run_command(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [PYTHON, *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    def test_smoke_training_run_writes_expected_artifacts(self) -> None:
        target_dir = "outputs/test_runs/smoke"
        result = self.run_command(
            "train.py",
            "--epochs",
            "16",
            "--samples-per-class",
            "60",
            "--run-name",
            "smoke",
            "--output-dir",
            target_dir,
            "--quiet",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Test accuracy:", result.stdout)
        self.assertIn("Best epoch:", result.stdout)

        run_dir = PROJECT_ROOT / target_dir
        expected_files = {
            "metrics.json",
            "training_config.json",
            "training_history.csv",
            "classification_report.csv",
            "confusion_matrix.csv",
            "model_weights.npz",
            "run_summary.md",
        }
        self.assertTrue(expected_files.issubset({path.name for path in run_dir.iterdir()}))

    def test_validation_metrics_and_early_stopping_are_recorded(self) -> None:
        target_dir = "outputs/test_runs/validation"
        result = self.run_command(
            "train.py",
            "--epochs",
            "40",
            "--samples-per-class",
            "50",
            "--validation-ratio",
            "0.2",
            "--dropout",
            "0.1",
            "--l2-lambda",
            "0.0005",
            "--early-stopping-patience",
            "4",
            "--lr-decay",
            "0.99",
            "--output-dir",
            target_dir,
            "--quiet",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        metrics = json.loads((PROJECT_ROOT / target_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("validation_size", metrics)
        self.assertGreater(metrics["validation_size"], 0)
        self.assertLessEqual(metrics["epochs_completed"], metrics["epochs_requested"])
        self.assertIn("best_epoch", metrics)

        with (PROJECT_ROOT / target_dir / "training_history.csv").open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader)
        self.assertIn("val_loss", row)
        self.assertIn("val_accuracy", row)
        self.assertIn("learning_rate", row)

    def test_mnist_mode_fails_with_clear_message_when_csv_is_missing(self) -> None:
        result = self.run_command(
            "train.py",
            "--dataset",
            "mnist_csv",
            "--csv-path",
            "data/raw/missing_train.csv",
            "--output-dir",
            "outputs/test_runs/mnist_missing",
            "--quiet",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CSV dataset not found", result.stderr)

    def test_predict_cli_generates_json_output(self) -> None:
        train_dir = "outputs/test_runs/predict_train"
        train_result = self.run_command(
            "train.py",
            "--epochs",
            "12",
            "--samples-per-class",
            "40",
            "--output-dir",
            train_dir,
            "--quiet",
        )
        self.assertEqual(train_result.returncode, 0, msg=train_result.stderr)

        prediction_json = "outputs/test_runs/predict_output.json"
        predict_result = self.run_command(
            "predict.py",
            "--model-path",
            f"{train_dir}/model_weights.npz",
            "--label",
            "5",
            "--output-json",
            prediction_json,
        )
        self.assertEqual(predict_result.returncode, 0, msg=predict_result.stderr)
        self.assertIn("Predicted label:", predict_result.stdout)

        prediction = json.loads((PROJECT_ROOT / prediction_json).read_text(encoding="utf-8"))
        self.assertEqual(prediction["source"], "synthetic")
        self.assertIn("probabilities", prediction)

    def test_demo_helpers_load_run_artifacts(self) -> None:
        target_dir = "outputs/test_runs/demo_source"
        result = self.run_command(
            "train.py",
            "--epochs",
            "10",
            "--samples-per-class",
            "35",
            "--output-dir",
            target_dir,
            "--quiet",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        run_dir = PROJECT_ROOT / target_dir
        metrics = load_metrics(run_dir)
        history = load_training_history(run_dir)
        matrix = load_confusion_matrix(run_dir)

        self.assertIn("test_accuracy", metrics)
        self.assertGreater(len(history), 0)
        self.assertEqual(len(matrix), 10)

    def test_experiment_runner_writes_comparison_reports(self) -> None:
        output_root = "outputs/test_runs/experiments"
        result = self.run_command(
            "experiments.py",
            "--epochs",
            "8",
            "--samples-per-class",
            "30",
            "--output-root",
            output_root,
            "--quiet",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue((PROJECT_ROOT / output_root / "experiment_comparison.csv").exists())
        self.assertTrue((PROJECT_ROOT / output_root / "experiment_comparison.md").exists())


if __name__ == "__main__":
    unittest.main()
