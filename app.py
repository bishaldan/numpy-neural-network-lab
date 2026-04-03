from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from portfolio_nn.demo_data import (
    list_run_directories,
    load_classification_report,
    load_confusion_matrix,
    load_metrics,
    load_training_history,
)
from portfolio_nn.inference import predict_from_saved_model


st.set_page_config(page_title="NumPy Classifier Demo", page_icon="NN", layout="wide")
st.title("NumPy Classifier Portfolio Demo")
st.caption("Interactive viewer for training artifacts, metrics, and saved-model inference.")

run_directories = list_run_directories("outputs")
if not run_directories:
    st.warning("No run directories found under outputs/. Run training first.")
    st.stop()

selected_dir = st.sidebar.selectbox("Saved run", run_directories, format_func=lambda path: str(path))
metrics = load_metrics(selected_dir)
history_rows = load_training_history(selected_dir)
history_frame = pd.DataFrame(history_rows)
confusion = pd.DataFrame(load_confusion_matrix(selected_dir))

st.subheader("Run overview")
left, middle, right = st.columns(3)
left.metric("Run name", metrics["run_name"])
middle.metric("Best epoch", metrics["best_epoch"])
right.metric("Test accuracy", f"{metrics['test_accuracy']:.4f}")

st.write(
    "This project demonstrates a from-scratch NumPy neural network with validation tracking, "
    "regularization options, Docker workflows, and portfolio-oriented reporting."
)

history_frame["epoch"] = history_frame["epoch"].astype(int)
for column in ["train_loss", "train_accuracy", "learning_rate"]:
    history_frame[column] = history_frame[column].astype(float)
for column in ["val_loss", "val_accuracy"]:
    history_frame[column] = pd.to_numeric(history_frame[column], errors="coerce")

curve_left, curve_right = st.columns(2)
with curve_left:
    st.markdown("### Training curves")
    st.line_chart(history_frame.set_index("epoch")[["train_loss", "val_loss"]])
with curve_right:
    st.markdown("### Accuracy curves")
    st.line_chart(history_frame.set_index("epoch")[["train_accuracy", "val_accuracy"]])

matrix_col, report_col = st.columns(2)
with matrix_col:
    st.markdown("### Confusion matrix")
    st.dataframe(confusion, use_container_width=True)
with report_col:
    st.markdown("### Classification report")
    st.code(load_classification_report(selected_dir), language="text")

st.markdown("### Demo prediction")
prediction_label = st.slider("Synthetic digit label", min_value=0, max_value=9, value=3)
if st.button("Run inference"):
    prediction = predict_from_saved_model(
        Path(selected_dir) / "model_weights.npz",
        source="synthetic",
        label=prediction_label,
    )
    st.success(
        f"Predicted {prediction['predicted_label']} with confidence {prediction['confidence']:.4f} "
        f"(expected {prediction['expected_label']})"
    )
    st.json(prediction["probabilities"])
