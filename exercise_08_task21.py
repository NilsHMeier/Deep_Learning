import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set parameters for training
BATCH_SIZE = 64
EPOCHS = 10


def plot_image_comparison(original_image: np.ndarray, transformed_image: np.ndarray,
                          title: str = '', filename: str = None):
    # Plot a comparison of the original and transformed image
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), tight_layout=True)
    fig.suptitle(title, fontsize=16)
    ax1.imshow(original_image)
    ax1.set_title('Original')
    ax2.imshow(transformed_image)
    ax2.set_title('Transformed')

    # Save the figure if a filename is provided
    if filename is not None:
        fig.savefig(f'Figures/exercise_08/{filename}.png')
    fig.show()


def task_21_part_i():
    # Random crop on CIFAR10
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10('./data', download=True, train=True)

    # Plot a comparison of the original and transformed image
    original_image, _ = dataset[0]
    transformed_image = transform(original_image)
    plot_image_comparison(original_image=original_image,
                          transformed_image=np.moveaxis(transformed_image.numpy(), 0, -1),
                          title='Random Crop on CIFAR10', filename='Random_Crop_CIFAR10')

    # Add noise to MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
    ])
    dataset = torchvision.datasets.MNIST('./data', download=True, train=True)

    # Plot a comparison of the original and transformed image
    original_image, _ = dataset[0]
    transformed_image = transform(original_image)
    plot_image_comparison(original_image=original_image,
                          transformed_image=np.moveaxis(transformed_image.numpy(), 0, -1),
                          title='Add Noise to MNIST', filename='Noise_MNIST')

    # Vertical flip on FashionMNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(p=1.0),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.FashionMNIST('./data', download=True, train=True)

    # Plot a comparison of the original and transformed image
    original_image, _ = dataset[0]
    transformed_image = transform(original_image)
    plot_image_comparison(original_image=original_image,
                          transformed_image=np.moveaxis(transformed_image.numpy(), 0, -1),
                          title='Vertical Flip on FashionMNIST', filename='Vertical_Flip_FashionMNIST')


class TwoMoonsDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int = 1000, noise: float = 0.1):
        self.X, self.y = self.generate_two_moons(n_samples=n_samples, noise=noise)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def generate_two_moons(n_samples: int = 1000, noise: float = 0.05):
        # Generate the two moons dataset
        X, y = datasets.make_moons(n_samples=n_samples, noise=noise)
        return X.astype(np.float32), y.astype(np.float32)


def task_21_part_ii():
    # Create a data loader from the TwoMoonsDataset
    dataset = TwoMoonsDataset()
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define a simple model
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    # Define the loss function and optimizer
    loss_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        # Iterate over the data loader
        for batch_X, batch_y in data_loader:
            # Reset the gradients
            optimizer.zero_grad()

            # Compute the predictions and loss
            predictions = model(batch_X)
            loss = loss_criterion(predictions, batch_y.unsqueeze(1))

            # Back-propagate the loss and update the model parameters
            loss.backward()
            optimizer.step()

    # Predict the labels for the entire dataset
    with torch.no_grad():
        predictions = model(torch.from_numpy(dataset.X))
        predictions = predictions.numpy().squeeze()

    # Plot the predictions
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    cmap = sns.cubehelix_palette(n_colors=10, as_cmap=True)
    sns.scatterplot(x=dataset.X[:, 0], y=dataset.X[:, 1], hue=predictions, ax=ax, palette=cmap)
    fig.suptitle('Predictions on TwoMoonsDataset', fontsize=16)
    ax.legend_.set_title('Prediction Probability')
    fig.savefig('Figures/exercise_08/TwoMoonsDataset_Predictions.png')
    fig.show()


if __name__ == '__main__':
    task_21_part_i()
    task_21_part_ii()
