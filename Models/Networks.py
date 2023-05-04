from .Losses import SquaredError, Loss
from .Activations import Sigmoid, Activation
from typing import List
import numpy as np


class NeuralNetwork:
    """
    # The NeuralNetwork class is supposed to work as follows:
    We can define a network, i.e., with 2 input, 3 hidden, and 1 output neurons as net = NeuralNetwork([2,3,1]).
    A forward pass (prediction) can be computed using net.forward(x) and the gradients of all parameters can be obtained
    via net.backprop(x, y). Finally, the network can be trained by calling net.train(x, y).
    Other activation functions can be provided when creating the object:
    net = NeuralNetwork([2,3,1], activations=[ReLU(), sigmoid()])
    """

    def __init__(self, neurons_per_layer: List[int], activations: List[Activation] = None,
                 loss: Loss = None):
        # Neurons per layer (incl. input and output)
        self.neurons_per_layer = np.array(neurons_per_layer)
        # List holding the weight matrices
        self.weight = []
        # List holding the bias vectors
        self.bias = []
        # List holding the activation functions
        self.activation = []
        # Intermediate states s (they get set by each forward pass)
        self._s = []
        # Intermediate states x(they get set by each forward pass)
        self._x = []

        # Initialize the model parameters
        for d_in, d_out in zip(neurons_per_layer[:-1], neurons_per_layer[1:]):
            self.weight.append(np.random.normal(loc=0, scale=np.sqrt(2 / (d_in + d_out)), size=(d_out, d_in)))
            self.bias.append(np.zeros((d_out, 1)))
        for d in neurons_per_layer:
            self._s.append(np.zeros((d, 1)))
            self._x.append(np.zeros((d, 1)))
        if activations is None:
            # If no activations list is given, use sigmoid() as default
            for _ in range(len(neurons_per_layer) - 1):
                self.activation.append(Sigmoid())
        else:
            self.activation = activations

        if loss is None or not isinstance(loss, Loss):
            # If no loss is given, use squared error as default
            self.loss = SquaredError()
        else:
            self.loss = loss

    @property
    def s(self):
        return self._s

    @property
    def x(self):
        return self._x

    def forward(self, x: np.ndarray):
        # Check the shape of the input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        assert x.shape[1] == self.neurons_per_layer[0], "Input dimension does not match the number of input neurons"

        # Save the input as the output of the first layer
        self._x[0] = x

        # Loop over all layers
        for i in range(len(self.neurons_per_layer) - 1):
            # Calculate the signal of the current layer and store it in self._s
            signal = np.matmul(x, self.weight[i].T) + self.bias[i].T
            self._s[i + 1] = signal

            # Calculate the activation of the current layer and store it in self._x
            output = self.activation[i](signal)
            self._x[i + 1] = output

            # Set the input of the next layer to the output of the current layer
            x = output

        # Return the output of the last layer
        return x

    def loss(self, y_prediction, y_true: np.ndarray) -> float:
        return self.loss(y_true, y_prediction)

    def backprop(self, x: np.ndarray, y):
        # Start by computing the forward pass
        self.forward(x)

        # Initialize a list for the deltas
        deltas = [np.nan] * (len(self.weight))

        # Compute the delta of the last layer using the gradient of the loss
        deltas[-1] = self.loss.gradient(y, self._x[-1], self.activation[-1].gradient(self._s[-1]))

        # Loop over all layers in reverse order
        for i in range(len(self.neurons_per_layer) - 2, 0, -1):
            # Sum over the deltas of the next layer
            deltas[i - 1] = (deltas[i]) @ self.weight[i] * self.activation[i - 1].gradient(self._s[i])

        # Initialize a list for the gradients
        weight_gradients = [np.zeros(shape=w.shape) for w in self.weight]
        bias_gradients = [np.zeros(shape=b.shape) for b in self.bias]

        # Loop over all layers
        for i in range(len(self.neurons_per_layer) - 1):
            # Compute the gradients of the weights and biases
            weight_gradients[i] = deltas[i].T @ self._x[i]
            bias_gradients[i] = deltas[i].T

        # Return the gradients
        return weight_gradients, bias_gradients

    def train(self, x: np.ndarray, y: np.ndarray, eta: float = 0.1, iterations: int = 100, mode: str = 'sgd'):
        # Check the shape of the input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        assert x.shape[1] == self.neurons_per_layer[0], "Input dimension does not match the number of input neurons"

        # Loop over the iterations and save the loss after each iteration
        loss = [self.loss(y, self.forward(x))]
        for i in range(iterations):
            if mode == 'sgd':
                # Compute the gradients for each individual sample
                for j in range(x.shape[0]):
                    weight_gradients, bias_gradients = self.backprop(x[[j]], y[[j]])

            elif mode == 'batch':
                # Compute the gradients for the whole batch
                weight_gradients, bias_gradients = self.backprop(x, y)
                bias_gradients = [b.mean(axis=1, keepdims=True) for b in bias_gradients]

            else:
                raise ValueError("Mode must be either 'sgd' or 'batch'")

            # Update the weights and biases
            for k in range(len(self.neurons_per_layer) - 1):
                self.weight[k] -= eta * weight_gradients[k]
                self.bias[k] -= eta * bias_gradients[k]

            # Compute the loss and save it
            loss.append(self.loss(self.forward(x), y))

        # Return the loss
        return loss
