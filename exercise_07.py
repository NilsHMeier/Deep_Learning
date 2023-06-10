import torch
import torchvision
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

# Set parameters for training
BATCH_SIZE = 64
EPOCHS = 5


def load_and_preprocess_mnist(train: bool, root_dir: str = './data') -> DataLoader:
    # Create a transform pipeline
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset using torchvision
    dataset = torchvision.datasets.MNIST(root_dir, download=True, train=train, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader


class ConvNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], num_conv_layers: int, num_classes: int,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.input_shape = input_shape

        # Create the convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_channels = 1 if i == 0 else 4 * (2 ** (i - 1))
            out_channels = 4 * (2 ** i)
            conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3))
        self.conv_layers = nn.ModuleList(conv_layers)

        # Create a 2x2 max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the shape of the output of the convolutional layers
        for _ in range(num_conv_layers):
            input_shape = (input_shape[0] - 2) // 2, (input_shape[1] - 2) // 2

        self.fc1 = nn.Linear(in_features=8 * torch.prod(torch.tensor(input_shape)), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = functional.relu(conv_layer(x))
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, epochs: int = EPOCHS) -> Tuple[nn.Module, float]:
    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs and labels
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 200}')
                running_loss = 0.0

    # Predict the labels for the training set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            # Get the inputs and labels
            inputs, labels = data

            # Get the predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Update the total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Return the trained model
    return model, correct / total


def test_model(model: nn.Module, test_loader: DataLoader) -> float:
    # Test the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            # Get the inputs and labels
            inputs, labels = data

            # Get the predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Update the total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Return the accuracy
    return correct / total


def task_19(train_loader: DataLoader, test_loader: DataLoader):
    # Set a range of different dropout rates
    dropout_rates = np.linspace(0.1, 0.9, 9)
    train_scores = []
    test_scores = []
    models = []

    # Train a model for each dropout rate
    for dropout_rate in dropout_rates:
        print(f'Training model with dropout rate {dropout_rate}')

        # Create a model and optimizer
        model = ConvNetwork(input_shape=(28, 28), num_conv_layers=2, num_classes=10, dropout_rate=dropout_rate)
        optimizer = optim.Adam(params=model.parameters(), lr=0.01)

        # Train the model
        model, train_score = train_model(model, train_loader, nn.CrossEntropyLoss(), optimizer)
        train_scores.append(train_score)

        # Test the model
        test_score = test_model(model, test_loader)
        test_scores.append(test_score)
        models.append(model)

    # Plot the train and test scores
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(dropout_rates, train_scores, label='Train Score')
    ax.plot(dropout_rates, test_scores, label='Test Score')
    ax.legend()
    ax.set(xlabel='Dropout Rate', ylabel='Accuracy', title='Train and Test Scores vs Dropout Rate')
    fig.savefig('Figures/exercise_07/Dropout_Experiments.png')
    fig.show()

    # Take the model with 0.5 dropout rate and visualize the first convolutional layer
    model = models[4]
    fig, ax = plt.subplots(2, 2, tight_layout=True)
    for i, ax in enumerate(ax.flatten()):
        sns.heatmap(model.conv_layers[0].weight[i, 0].detach().numpy(), ax=ax,
                    cmap='gray', annot=True, cbar=False, fmt='.2f')
        ax.axis('off')
    fig.suptitle('First Convolutional Layer Filters')
    fig.savefig('Figures/exercise_07/Conv_Layer_Filters.png')
    fig.show()


def main():
    # Load the MNIST dataset for training and testing
    train_loader = load_and_preprocess_mnist(train=True)
    test_loader = load_and_preprocess_mnist(train=False)

    # Print the shapes of the train and test sets
    print(f'Train features shape: {train_loader.dataset.data.shape}',
          f'Train labels shape: {train_loader.dataset.targets.shape}')
    print(f'Test features shape: {test_loader.dataset.data.shape}',
          f'Test labels shape: {test_loader.dataset.targets.shape}')

    # Run task 19
    task_19(train_loader, test_loader)


if __name__ == '__main__':
    main()
