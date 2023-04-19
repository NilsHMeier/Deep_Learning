import torchvision.datasets
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn import datasets
from typing import Tuple, List


def make_binary_clusters(n_points=100, blob_centers: List[Tuple[float, float]] = None, cluster_std: float = 0.5,
                         linearly_separable: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # Generate data based on the given parameters
    if blob_centers is None:
        x, y = datasets.make_blobs(n_samples=n_points, n_features=2, centers=2, random_state=0, cluster_std=cluster_std)
    else:
        x, y = datasets.make_blobs(n_samples=n_points, n_features=2, centers=blob_centers, random_state=0,
                                   cluster_std=cluster_std)

    # Check the linear separability of the data and adjust if necessary
    if linearly_separable and not is_linearly_separable(x[y == 1], x[y == 0]):
        return make_binary_clusters(n_points, blob_centers, cluster_std - 0.1, linearly_separable)
    elif not linearly_separable and is_linearly_separable(x[y == 1], x[y == 0]):
        return make_binary_clusters(n_points, blob_centers, cluster_std + 0.1, linearly_separable)

    # Return the generated data
    y = np.where(y == 0, -1, y)
    # Subtract the mean of X from every data point => X has a mean of 0
    x = x - x.mean(0)
    return x, y


def is_linearly_separable(pos_class: np.ndarray, neg_class: np.ndarray) -> bool:
    pos_hull = ConvexHull(pos_class)
    neg_hull = ConvexHull(neg_class)
    return not Polygon(pos_hull.points).intersects(Polygon(neg_hull.points))


def plot_dataset(features: np.ndarray, labels: np.ndarray, filename: str = None):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_aspect(1)
    ax.plot(features[labels == +1, 0], features[labels == +1, 1], 'ro', label="positive class")
    ax.plot(features[labels == -1, 0], features[labels == -1, 1], 'bo', label="negative class")
    ax.legend()
    fig.show()

    # Save file if a filename is given
    if filename is not None:
        fig.savefig(filename)


def calculate_boundary(x1, w):
    x2 = -w[1] / w[2] * x1 - w[0] / w[2]
    return x2


def train_perceptron(data_points, labels, weights, max_iterations: int = 1000, lr: float = 1.0):
    num_iterations = 0
    mistakes_made = True
    while num_iterations < max_iterations and mistakes_made:
        mistakes_made = False
        for i in range(data_points.shape[0]):
            # If the current data point is misclassified, update weights
            if labels[i] * np.dot(weights, data_points[i]) <= 0:
                weights += lr * labels[i] * data_points[i]
                mistakes_made = True
                num_iterations += 1

    return weights, num_iterations


def find_maximal_margin(features, labels, filename: str = None) -> Tuple[float, float]:
    # Compute the maximum norm
    norms = np.linalg.norm(features, axis=1)
    R = max(norms)

    # Compute gamma
    svm = LinearSVC(C=1000, loss="hinge", tol=1e-5, random_state=0)
    svm.fit(features, labels)
    margin = 1 / np.sqrt(np.sum(svm.coef_ ** 2))

    # Get the separating hyperplane
    w = svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (svm.intercept_[0]) / w[1]
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # Plot toy data
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_aspect(1)
    ax.plot(features[labels == +1, 0], features[labels == +1, 1], 'ro', label="positive class")
    ax.plot(features[labels == -1, 0], features[labels == -1, 1], 'bo', label="negative class")
    ax.plot(xx, yy, 'k-')
    ax.plot(xx, yy_down, 'k--')
    ax.plot(xx, yy_up, 'k--')
    ax.plot([0, features[np.argmax(norms), 0]], [0, features[np.argmax(norms), 1]], 'k-')
    circle = plt.Circle((0, 0), R, color='k', fill=False)
    ax.add_artist(circle)
    ax.legend()
    fig.show()

    # Save the figure if a filename is provided
    if filename is not None:
        fig.savefig(filename)

    # Return the maximum norm and the margin
    return R, margin


def plot_decision_hyperplane(features, labels, weights, domain, filename=None):
    # Plot the dataset and the decision boundary
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_aspect(1)
    ax.set_xlim(features[:, 1].min() - 1, features[:, 1].max() + 1)
    ax.set_ylim(features[:, 2].min() - 1, features[:, 2].max() + 1)
    ax.plot(features[labels == +1, 1], features[labels == +1, 2], 'ro', label="positive class")
    ax.plot(features[labels == -1, 1], features[labels == -1, 2], 'bo', label="negative class")
    ax.plot(domain, calculate_boundary(domain, weights), 'k-')
    ax.legend()
    fig.show()

    # Save the figure if a filename is given
    if filename is not None:
        fig.savefig(filename)


def task_4():
    # Generate toy data
    X, y = make_binary_clusters(n_points=100, linearly_separable=True)

    # Plot toy data
    plot_dataset(features=X, labels=y, filename='Figures/exercise_02_Dataset.png')

    # Find the margin and the maximum norm
    R, margin = find_maximal_margin(features=X, labels=y, filename='Figures/exercise_02_Margin.png')

    # Prepend a 1 to each data point for the bias
    X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

    # Initialize the weight vector w
    w = np.array([0.0, -2.0, 1.0])

    # Get the domain of the features
    dom = np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 10)

    # Show the decision hyperplane for the current w
    plot_decision_hyperplane(features=X, labels=y, weights=w, domain=dom, filename='Figures/exercise_02_Initial.png')

    # Train the Perceptron and show the updated decision boundary
    w, total_updates = train_perceptron(X, y, w)

    # show the decision hyperplane for the learned w
    plot_decision_hyperplane(features=X, labels=y, weights=w, domain=dom, filename='Figures/exercise_02_Fitted.png')

    # Run the perceptron several times with random w initializations
    fig, ax = plt.subplots(figsize=(15, 10), tight_layout=True)
    ax.set_aspect(1)
    ax.set_xlim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_ylim(X[:, 2].min() - 1, X[:, 2].max() + 1)
    ax.plot(X[y == +1, 1], X[y == +1, 2], 'ro', label="positive class")
    ax.plot(X[y == -1, 1], X[y == -1, 2], 'bo', label="negative class")
    res = []
    for j in range(250):
        # Random initialization of w
        w = np.random.uniform(-5, 5, X.shape[1])
        w, total_updates = train_perceptron(X, y, w)
        res.append(total_updates)
        ax.plot(dom, calculate_boundary(dom, w), 'k-', alpha=0.1)  # k is short for black
    ax.legend(loc=2)
    fig.show()
    fig.savefig('Figures/exercise_02_Boundaries.png')

    # Set the bound: 4R² / margin²
    bound = round(4 * R ** 2 / margin ** 2)
    # Check if the bound holds for all experiments
    print(f'{bound = } iterations ')
    if np.max(res) <= bound:
        print('Bound holds for all experiments')

    fig, ax = plt.subplots()
    ax.hist(res, density=True)
    fig.show()
    fig.savefig('Figures/exercise_02_Iterations.png')


def task_5():
    # Load the MNIST dataset using torchvision
    mnist = torchvision.datasets.MNIST('./data', download=True)
    X = mnist.data
    y = mnist.targets

    # Plot the first entry of the training dataset
    fig, ax = plt.subplots()
    ax.imshow(X[0])
    fig.show()

    # Obtain the indices for the data points labeled with 0 or 1
    subset_indices = np.where(np.logical_or(y.numpy() == 0, y.numpy() == 1))

    # Transform the dataset to the required shape
    X = X[subset_indices].numpy()
    y = y[subset_indices].numpy()
    y[y == 0] = -1
    X = X.reshape(X.shape[0], -1)

    # Add 1 to each data point
    X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

    # Initialize the weight vector w
    w = np.full(X.shape[1], 0.0)

    # Train the perceptron
    w, total_updates = train_perceptron(X, y, w)

    # Check the classification error on train data
    predictions = np.sign(np.dot(X, w))
    accuracy = np.mean(predictions == y)
    print(f'Train Accuracy={accuracy}')

    # Load test data and apply the same transformations
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    X_test = mnist_test.data
    y_test = mnist_test.targets
    subset_indices_test = np.where(np.logical_or(y_test.numpy() == 0, y_test.numpy() == 1))
    X_test = X_test[subset_indices_test].numpy()
    y_test = y_test[subset_indices_test].numpy()
    y_test[y_test == 0] = -1
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = np.hstack((np.ones(shape=(X_test.shape[0], 1)), X_test))

    # check the classification error on train data
    test_predictions = np.sign(np.dot(X_test, w))
    accuracy = np.mean(test_predictions == y_test)
    print(f'Test Accuracy={accuracy}')

    # Show the misclassified instances
    misclassified_indices = np.where(test_predictions != y_test)[0]
    fig, ax = plt.subplots(ncols=len(misclassified_indices))
    for i, idx in enumerate(misclassified_indices):
        ax[i].imshow(X_test[idx, 1:].reshape(28, 28))
        ax[i].set(title=f'Actual Label is {int(y_test[idx])} \n Predicted Label is {int(test_predictions[idx])}')
    fig.show()
    fig.savefig('Figures/exercise_02_Mistakes.png')


if __name__ == '__main__':
    task_4()
    task_5()
