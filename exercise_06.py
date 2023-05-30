import torch
import torchvision
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List

# Set parameters for training
BATCH_SIZE = 32
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
    def __init__(self, input_shape: Tuple[int, int], num_conv_layers: int, num_classes: int):
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

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = functional.relu(conv_layer(x))
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def task_18():
    # Load the MNIST dataset for training and testing
    train_loader = load_and_preprocess_mnist(train=True)
    test_loader = load_and_preprocess_mnist(train=False)

    # Print the shapes of the train and test sets
    print(f'Train features shape: {train_loader.dataset.data.shape}',
          f'Train labels shape: {train_loader.dataset.targets.shape}')
    print(f'Test features shape: {test_loader.dataset.data.shape}',
          f'Test labels shape: {test_loader.dataset.targets.shape}')

    # Create a ConvNetwork instance and train it on the dataset
    model = ConvNetwork(input_shape=(28, 28), num_conv_layers=2, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
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

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            # Get the inputs and labels
            inputs, labels = data

            # Predict the labels
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Update the statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')


if __name__ == '__main__':
    task_18()
