import numpy as np
import matplotlib.pyplot as plt
import torchvision
from Models.Networks import NeuralNetwork
from Models.Activations import Softmax
from typing import Tuple


def load_and_preprocess_mnist(train: bool, root_dir: str = './data') -> Tuple[np.ndarray, np.ndarray]:
    # Load the MNIST dataset using torchvision
    mnist = torchvision.datasets.MNIST(root_dir, download=True, train=train)
    X = mnist.data.numpy()
    y = mnist.targets.numpy()

    # Reshape and normalize the features and encode the labels
    X = X.reshape(X.shape[0], -1) / 255.0
    encoded_y = np.zeros((y.shape[0], 10))
    encoded_y[np.arange(y.shape[0]), y] = 1

    return X, encoded_y


def task_13():
    # Load the MNIST train and test sets
    X_train, y_train = load_and_preprocess_mnist(train=True)
    X_test, y_test = load_and_preprocess_mnist(train=False)

    # Print the shapes of the train and test sets
    print(f'Train features shape: {X_train.shape}', f'Train labels shape: {y_train.shape}')
    print(f'Test features shape: {X_test.shape}', f'Test labels shape: {y_test.shape}')

    # Create a NeuralNetwork instance and train it on the dataset
    model = NeuralNetwork(neurons_per_layer=[X_train.shape[1], 10], activations=[Softmax()])
    losses = model.train(x=X_train, y=y_train, eta=0.001, iterations=200, mode='sgd')

    # Plot the loss over the iterations
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(losses)
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss over iterations')
    fig.show()


if __name__ == '__main__':
    task_13()
