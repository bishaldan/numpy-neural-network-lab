# Training Run Summary

## Overview
- Run name: mnist_run
- Dataset: mnist_csv
- Train size: 43200
- Validation size: 4799
- Test size: 12000
- Best epoch: 12
- Test loss: 0.1632
- Test accuracy: 0.9578

## Hyperparameters
- Epochs requested: 12
- Epochs completed: 12
- Batch size: 128
- Learning rate: 0.03
- Learning-rate decay: 0.995
- Dropout: 0.1
- L2 lambda: 0.0001
- Validation ratio: 0.1
- Early stopping patience: 4
- Early stopping triggered: False
- Hidden sizes: 128, 64
- Samples per class: n/a

## Final Metrics
- Final train loss: 0.0725
- Final train accuracy: 0.9787
- Final validation loss: 0.1478
- Final validation accuracy: 0.9564
- Best monitored loss: 0.1478

## Generated Artifacts
- `metrics.json`
- `training_config.json`
- `training_history.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `model_weights.npz`
- `run_summary.md`

## Generated Visuals
- `training_curves.png`
- `confusion_matrix.png`

## Portfolio Angle
- Neural network implemented from scratch with NumPy.
- Advanced training workflow includes validation monitoring, regularization, and early stopping.
- Reproducible CLI workflow suitable for Dockerized ML demos.
- Reviewable output artifacts support GitHub screenshots and project discussion.
