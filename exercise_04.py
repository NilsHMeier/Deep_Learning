from Models import NeuralNetwork
from Models.Activations import Sigmoid
from exercise_02 import make_binary_clusters
import torchvision
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def task_9():
    # Create a NeuralNetwork instance and set the weights and biases
    net = NeuralNetwork([2, 3, 2])
    net.weight[0] = np.array([[.2, -.3], [.1, -1.2], [.4, .3]])
    net.bias[0] = np.array([[.3], [-.1], [1.2]])
    net.weight[1] = np.array([[.4, .2, .2], [.1, .3, .5]])
    net.bias[1] = np.array([[-0.6], [0.5]])

    # Set the input and target
    x0 = np.array([[1., 2.]])
    y = np.array([[0.0, 1.0]])

    # Compute the forward pass
    result = net.forward(x0)
    print(f'Result: {result}')

    # Get the gradients of the weights and biases
    weight_gradients, bias_gradients = net.backprop(x0, y)

    # Print the gradients
    for i in range(len(weight_gradients)):
        print(f'Weight gradient {i}:')
        print(weight_gradients[i])
        print(f'Bias gradient {i}:')
        print(bias_gradients[i])


def task_10():
    # Generate a dataset and replace the negative labels with 0
    features, labels = make_binary_clusters(n_points=100, linearly_separable=True)
    labels[labels == -1] = 0

    # Reshape the labels to be a column vector
    labels = labels.reshape(-1, 1)

    # Print the shapes of the features and labels
    print(f'Features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')

    # Create a NeuralNetwork instance and train it on the dataset
    model = NeuralNetwork([2, 4, 1])
    print(f'Initial Weights: {model.weight}')
    losses = model.train(x=features, y=labels, eta=0.1, iterations=50, mode='batch')
    print(f'Final Weights: {model.weight}')

    # Plot the losses
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(losses)
    ax.set(title='Loss per Epoch', xlabel='Iteration', ylabel='Loss', yscale='log')
    fig.savefig('Figures/exercise_04/Training_Loss.png')
    fig.show()

    # Predict the labels of the features and plot them
    predictions = model.forward(features)
    fig, ax = plt.subplots(tight_layout=True)
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=predictions.reshape(-1), ax=ax)
    ax.set(title='Predictions', xlabel='Feature 1', ylabel='Feature 2')
    ax.legend(title='Probability', loc='upper right')
    fig.savefig('Figures/exercise_04/Predictions.png')
    fig.show()


def load_and_preprocess_mnist(train: bool, root_dir: str = './data') -> Tuple[np.ndarray, np.ndarray]:
    # Load the MNIST dataset using torchvision
    mnist = torchvision.datasets.MNIST(root_dir, download=True, train=train)
    X = mnist.data
    y = mnist.targets

    # Obtain the indices for the data points labeled with 6 or 9
    subset_indices = np.where(np.logical_or(y.numpy() == 6, y.numpy() == 9))

    # Transform the dataset to the required shape
    X = X[subset_indices].numpy()
    y = y[subset_indices].numpy()
    y = np.where(y == 6, 0, 1)
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(-1, 1)

    # Normalize the features
    X = X / 255.0

    return X, y


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    return ndimage.rotate(image, angle, reshape=False)


def task_11():
    # Load the MNIST train and test sets
    X_train, y_train = load_and_preprocess_mnist(train=True)
    X_test, y_test = load_and_preprocess_mnist(train=False)

    # Print the shapes of the train and test sets
    print(f'Train features shape: {X_train.shape}', f'Train labels shape: {y_train.shape}')
    print(f'Test features shape: {X_test.shape}', f'Test labels shape: {y_test.shape}')

    # Create a NeuralNetwork instance and train it on the dataset
    model = NeuralNetwork(neurons_per_layer=[X_train.shape[1], 1], activations=[Sigmoid()])
    model.train(x=X_train, y=y_train, eta=0.001, iterations=200, mode='batch')

    # Let the model predict both the train and test sets
    train_predictions = model.forward(X_train)
    train_predictions = np.where(train_predictions > 0.5, 1, 0)
    test_predictions = model.forward(X_test)
    test_predictions = np.where(test_predictions > 0.5, 1, 0)

    # Print the accuracy of the model on both the train and test sets
    print(f'Train accuracy: {np.mean(train_predictions == y_train)}')
    print(f'Test accuracy: {np.mean(test_predictions == y_test)}')

    # Let the model predict a rotated samples with a 9
    angles = np.arange(0, 361, 15)
    selected_image = X_test[np.where(y_test == 1)[0][5]].reshape(28, 28)
    rotated_samples = np.array([rotate_image(selected_image, angle).flatten() for angle in angles])
    predictions = model.forward(rotated_samples).squeeze()

    # Plot some of the rotated samples
    fig, axs = plt.subplots(nrows=2, ncols=3, tight_layout=True)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(rotated_samples[3*i].reshape(28, 28), cmap='gray')
        ax.set(title=f'Angle: {angles[3*i]}° \n Prediction: {predictions[3*i]:.2f}')
    fig.savefig('Figures/exercise_04/Rotation_Samples.png')
    fig.show()

    # Plot the predictions
    fig, ax = plt.subplots(tight_layout=True)
    sns.lineplot(x=angles, y=predictions, ax=ax)
    ax.set(title='Predictions', xlabel='Angle', ylabel='Probability')
    fig.savefig('Figures/exercise_04/Rotation_Probabilities.png')
    fig.show()


if __name__ == '__main__':
    task_9()
    task_10()
    task_11()
