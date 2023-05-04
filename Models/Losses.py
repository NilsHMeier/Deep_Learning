import abc
import numpy as np


class Loss(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, signal_derivative: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.evaluate(y_true, y_pred)


class SquaredError(Loss):

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Check that the shapes of the arrays match
        assert y_true.shape == y_pred.shape, 'Shapes do not match'

        # Compute the loss
        return np.mean(np.square(y_pred - y_true))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, signal_derivative: np.ndarray) -> np.ndarray:
        # Compute the gradient
        return 2 * (y_pred - y_true) * signal_derivative
