import abc
import numpy as np
import matplotlib.pylab as plt
from typing import List, Tuple


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


class NeuralNetwork:
    """
    # The NeuralNetwork class is supposed to work as follows:
    We can define a network, i.e., with 2 input, 3 hidden, and 1 output neurons as net = NeuralNetwork([2,3,1]).
    A forward pass (prediction) can be computed using net.forward(x) and the gradients of all parameters can be obtained
    via net.backprop(x, y). Finally, the network can be trained by calling net.train(x, y).
    Other activation functions can be provided when creating the object:
    net = NeuralNetwork([2,3,1], activations=[ReLU(), ReLU(), sigmoid()])
    """

    def __init__(self, neurons_per_layer: List[int], activations: List[Activation] = None):
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
        for d in neurons_per_layer[:-1]:
            self._s.append(np.zeros((d, 1)))
        for d in neurons_per_layer:
            self._x.append(np.zeros((d, 1)))
        if activations is None:
            # If no activations list is given, use sigmoid() as default
            for _ in range(len(neurons_per_layer) - 1):
                self.activation.append(Sigmoid())
        else:
            self.activation = activations

    @property
    def s(self):
        return self._s

    @property
    def x(self):
        return self._x

    def forward(self, x: np.ndarray):
        # Check the shape of the input
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        assert x.shape[0] == self.neurons_per_layer[0], "Input dimension does not match the number of input neurons"

        # Loop over all layers
        for i in range(len(self.neurons_per_layer) - 1):
            # Calculate the signal of the current layer and store it in self._s
            signal = self.weight[i] @ x + self.bias[i]
            self._s[i] = signal

            # Calculate the activation of the current layer and store it in self._x
            output = self.activation[i](signal)
            self._x[i] = output

            # Set the input of the next layer to the output of the current layer
            x = output

        # Return the output of the last layer
        return self._x[-1]


def task_7():
    # you can also overwrite the parameters yourself
    # to confirm that your implementation is correct,
    # i.e., that it matches what you computed by hand
    net = NeuralNetwork([2, 3, 2])
    # set the weights and biases
    net.weight[0] = np.array([[.2, -.3], [.1, -1.2], [.4, .3]])
    net.bias[0] = np.array([[.3], [-.1], [1.2]])
    net.weight[1] = np.array([[.4, .2, .2], [.1, .3, .5]])
    net.bias[1] = np.array([[-0.6], [0.5]])
    # set the input
    x0 = np.array([[1.], [2.]])
    # compute the forward pass
    net.forward(x0)
    # compare the intermediate states
    print(net.s[0])
    print(net.x[0])
    print(net.s[1])
    print(net.x[1])


def generate_target_function(x):
    return np.sin(x)


def approximate_function(x_values: np.ndarray, lambda_: np.ndarray, b_: np.ndarray):
    # Calculate the approximation
    predicted_values = []
    for x in x_values:
        predicted_values.append(sum([lambda_[i] * max(0, x - b_[i]) for i in range(len(lambda_))]))
    return np.array(predicted_values)


def task_8(number_of_points: int = 5, interval: Tuple[float, float] = (0, 6), save_plot: bool = False):
    # Set the number of functions and the interval
    n = number_of_points
    a, b = interval

    # Distribute the points evenly and calculate the spacing
    x = np.linspace(a, b, n)
    h = x[1] - x[0]

    # Use the spacing to get the biases
    b_ = np.array([a + h * i for i in range(-1, n - 1)])
    print('b=', b_)

    # Get the target values
    f_ = generate_target_function(x)

    # Set up the matrix used in the linear system of equations
    A = np.zeros((n, n))
    for i in range(n):
        A[i:, i] = np.arange(1, n - i + 1) * h

    # Solve the linear system of equations to get the coefficients
    lambda_ = np.linalg.solve(A, f_)

    # plot the function and the approximation
    dom = np.linspace(-1, 8, 100)
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(dom, generate_target_function(x=dom), 'b-', label='Target function')
    ax.plot(dom, approximate_function(x_values=dom, lambda_=lambda_, b_=b_), 'r-', label='Approximation')
    ax.scatter(x, f_, c='k', label='Points')
    ax.set(title=f'Approximation of the function with {n} points')
    ax.legend()
    fig.show()

    # Save the plot if a filename is given
    if save_plot:
        fig.savefig(f'Figures/exercise_03/{number_of_points}_Points.png')


if __name__ == '__main__':
    task_7()
    task_8(number_of_points=5, save_plot=True)
    task_8(number_of_points=10, save_plot=True)
    task_8(number_of_points=20, save_plot=True)
