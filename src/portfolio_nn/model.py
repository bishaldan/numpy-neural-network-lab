from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingSnapshot:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None
    learning_rate: float


@dataclass
class TrainingResult:
    history: list[TrainingSnapshot]
    best_epoch: int
    best_loss: float
    stopped_early: bool


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        output_size: int = 10,
        learning_rate: float = 0.05,
        dropout_rate: float = 0.0,
        l2_lambda: float = 0.0,
        seed: int = 42,
    ) -> None:
        layer_sizes = (input_size, *hidden_sizes, output_size)
        rng = np.random.default_rng(seed)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.weights = []
        self.biases = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
            weight = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            bias = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(weight.astype(np.float64))
            self.biases.append(bias)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
        encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
        encoded[np.arange(labels.shape[0]), labels] = 1.0
        return encoded

    def forward(
        self,
        X: np.ndarray,
        *,
        training: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray | None], np.ndarray]:
        activations = [X]
        pre_activations = []
        dropout_masks: list[np.ndarray | None] = []
        current = X

        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            z = current @ weight + bias
            current = self._relu(z)
            mask = None
            if training and self.dropout_rate > 0.0:
                if rng is None:
                    rng = np.random.default_rng()
                keep_probability = 1.0 - self.dropout_rate
                mask = (rng.random(current.shape) < keep_probability).astype(np.float64) / keep_probability
                current = current * mask
            pre_activations.append(z)
            activations.append(current)
            dropout_masks.append(mask)

        output_logits = current @ self.weights[-1] + self.biases[-1]
        probabilities = self._softmax(output_logits)
        pre_activations.append(output_logits)
        activations.append(probabilities)
        dropout_masks.append(None)
        return activations, pre_activations, dropout_masks, probabilities

    def compute_loss(self, y_true: np.ndarray, probabilities: np.ndarray) -> float:
        clipped = np.clip(probabilities, 1e-12, 1.0)
        data_loss = float(-np.mean(np.log(clipped[np.arange(y_true.shape[0]), y_true])))
        if self.l2_lambda <= 0.0:
            return data_loss
        l2_penalty = sum(np.sum(weight**2) for weight in self.weights)
        return data_loss + (self.l2_lambda / (2.0 * y_true.shape[0])) * float(l2_penalty)

    def backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        dropout_masks: list[np.ndarray | None],
        y_true: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch_size = y_true.shape[0]
        grad_weights = [np.zeros_like(weight) for weight in self.weights]
        grad_biases = [np.zeros_like(bias) for bias in self.biases]

        delta = activations[-1] - self._one_hot(y_true, activations[-1].shape[1])
        delta /= batch_size

        for layer_index in range(len(self.weights) - 1, -1, -1):
            grad_weights[layer_index] = activations[layer_index].T @ delta
            if self.l2_lambda > 0.0:
                grad_weights[layer_index] += (self.l2_lambda / batch_size) * self.weights[layer_index]
            grad_biases[layer_index] = np.sum(delta, axis=0, keepdims=True)

            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * self._relu_derivative(pre_activations[layer_index - 1])
                mask = dropout_masks[layer_index - 1]
                if mask is not None:
                    delta = delta * mask

        return grad_weights, grad_biases

    def update_parameters(self, grad_weights: list[np.ndarray], grad_biases: list[np.ndarray]) -> None:
        for index in range(len(self.weights)):
            self.weights[index] -= self.learning_rate * grad_weights[index]
            self.biases[index] -= self.learning_rate * grad_biases[index]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 120,
        batch_size: int = 64,
        verbose: bool = True,
        early_stopping_patience: int = 0,
        lr_decay: float = 1.0,
        seed: int = 42,
    ) -> TrainingResult:
        rng = np.random.default_rng(seed)
        history: list[TrainingSnapshot] = []
        best_epoch = 1
        best_loss = float("inf")
        best_weights = deepcopy(self.weights)
        best_biases = deepcopy(self.biases)
        patience_counter = 0
        stopped_early = False

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start_index in range(0, X_train.shape[0], batch_size):
                end_index = start_index + batch_size
                X_batch = X_shuffled[start_index:end_index]
                y_batch = y_shuffled[start_index:end_index]

                activations, pre_activations, dropout_masks, _ = self.forward(X_batch, training=True, rng=rng)
                grad_weights, grad_biases = self.backward(activations, pre_activations, dropout_masks, y_batch)
                self.update_parameters(grad_weights, grad_biases)

            _, _, _, train_probabilities = self.forward(X_train)
            train_predictions = np.argmax(train_probabilities, axis=1)
            train_loss = self.compute_loss(y_train, train_probabilities)
            train_accuracy = float(np.mean(train_predictions == y_train))

            val_loss = None
            val_accuracy = None
            monitored_loss = train_loss
            if X_val is not None and y_val is not None:
                _, _, _, val_probabilities = self.forward(X_val)
                val_predictions = np.argmax(val_probabilities, axis=1)
                val_loss = self.compute_loss(y_val, val_probabilities)
                val_accuracy = float(np.mean(val_predictions == y_val))
                monitored_loss = val_loss

            snapshot = TrainingSnapshot(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                learning_rate=self.learning_rate,
            )
            history.append(snapshot)

            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_epoch = epoch
                best_weights = deepcopy(self.weights)
                best_biases = deepcopy(self.biases)
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                if val_loss is None:
                    print(
                        f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                        f"train_accuracy={train_accuracy:.4f} lr={self.learning_rate:.5f}"
                    )
                else:
                    print(
                        f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                        f"train_accuracy={train_accuracy:.4f} val_loss={val_loss:.4f} "
                        f"val_accuracy={val_accuracy:.4f} lr={self.learning_rate:.5f}"
                    )

            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                stopped_early = True
                break

            self.learning_rate *= lr_decay

        self.weights = best_weights
        self.biases = best_biases
        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_loss=best_loss,
            stopped_early=stopped_early,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, _, _, probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, _, probabilities = self.forward(X)
        return probabilities

    def save(self, path: str, extra: dict[str, np.ndarray] | None = None) -> None:
        payload: dict[str, np.ndarray] = {}
        for index, weight in enumerate(self.weights):
            payload[f"weight_{index}"] = weight
        for index, bias in enumerate(self.biases):
            payload[f"bias_{index}"] = bias
        if extra:
            payload.update(extra)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> tuple["NeuralNetwork", dict[str, np.ndarray]]:
        data = np.load(path, allow_pickle=True)
        weight_keys = sorted([key for key in data.files if key.startswith("weight_")], key=lambda item: int(item.split("_")[1]))
        bias_keys = sorted([key for key in data.files if key.startswith("bias_")], key=lambda item: int(item.split("_")[1]))
        weights = [data[key] for key in weight_keys]
        biases = [data[key] for key in bias_keys]

        hidden_sizes = tuple(weight.shape[1] for weight in weights[:-1])
        model = cls(
            input_size=weights[0].shape[0],
            hidden_sizes=hidden_sizes,
            output_size=weights[-1].shape[1],
        )
        model.weights = [weight.astype(np.float64) for weight in weights]
        model.biases = [bias.astype(np.float64) for bias in biases]

        extra = {key: data[key] for key in data.files if not key.startswith("weight_") and not key.startswith("bias_")}
        return model, extra
