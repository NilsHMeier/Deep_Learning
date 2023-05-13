import abc
import numpy as np


class Activation(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, z: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def gradient(self, z) -> np.ndarray:
        pass

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.evaluate(z)


class ReLU(Activation):
    def evaluate(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1, 0)


class Sigmoid(Activation):
    def evaluate(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        return self.evaluate(z) * (1 - self.evaluate(z))


class Tanh(Activation):
    def evaluate(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        return 1 - np.square(self.evaluate(z))


class Softmax(Activation):
    def evaluate(self, z: np.ndarray) -> np.ndarray:
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        # Compute the jacobi matrix
        pred = self.evaluate(z)
        J = np.zeros((z.shape[0], z.shape[0]))
        for i in range(z.shape[0]):
            for j in range(z.shape[0]):
                if i == j:
                    J[i, j] = pred[i] * (1 - pred[i])
                else:
                    J[i, j] = -pred[i] * pred[j]
        return J


class ELU(Activation):
    def __init__(self, alpha):
        self.alpha = alpha

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1, self.evaluate(z) + self.alpha)
