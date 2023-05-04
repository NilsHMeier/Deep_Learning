from Models import NeuralNetwork
from exercise_02 import make_binary_clusters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def task_9():
    # Create a NeuralNetwork instance and set the weights and biases
    net = NeuralNetwork([2, 3, 2])
    net.weight[0] = np.array([[.2, -.3], [.1, -1.2], [.4, .3]])
    net.bias[0] = np.array([[.3], [-.1], [1.2]])
    net.weight[1] = np.array([[.4, .2, .2], [.1, .3, .5]])
    net.bias[1] = np.array([[-0.6], [0.5]])

    # Set the input and target
    x0 = np.array([[1.], [2.]])
    y = np.array([[0.0], [1.0]])

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


if __name__ == '__main__':
    # task_9()
    task_10()
