import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal

# Set parameters for training
BATCH_SIZE = 64
EPOCHS = 10


def prepare_dataset(normalize_input: bool = False, add_constant: float = 0.0) -> DataLoader:
    # Create a transform pipeline
    if not normalize_input:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x + add_constant)
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
            torchvision.transforms.Lambda(lambda x: x + add_constant)
        ])

    # Load the MNIST dataset using torchvision
    dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader


class DenseNetwork(nn.Module):
    def __init__(self, num_classes: int, batch_norm: Literal['none', 'before', 'after'] = 'none',
                 activation: Literal['relu', 'sigmoid'] = 'relu'):
        super().__init__()

        # Create the dense layers
        dense_layers = []
        for i in range(3):
            in_features = 28 * 28 if i == 0 else 128
            out_features = 128 if i < 2 else num_classes
            dense_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
        self.dense_layers = nn.ModuleList(dense_layers)

        # Create the batch normalization layers
        batch_norm_layers = []
        for i in range(3):
            if batch_norm == 'before' or batch_norm == 'after':
                batch_norm_layers.append(nn.BatchNorm1d(num_features=128))
        self.batch_norm_layers = nn.ModuleList(batch_norm_layers)

        # Save the activation function and batch normalization type
        self.activation = activation
        self.batch_norm = batch_norm

    def forward(self, x):
        for layer in self.dense_layers:
            # Pass the input through the layer
            x = layer(x)

            # Apply the batch normalization layer if specified
            if self.batch_norm == 'before':
                x = self.batch_norm_layers[0](x)

            # Apply the activation function
            if self.activation == 'relu':
                x = nn.functional.relu(x)
            elif self.activation == 'sigmoid':
                x = nn.functional.sigmoid(x)

            # Apply the batch normalization layer if specified
            if self.batch_norm == 'after':
                x = self.batch_norm_layers[0](x)

        return x
    

def get_gradients_from_training(model: nn.Module, train_loader: DataLoader, loss_function: nn.Module,
                                optimizer: optim.Optimizer) -> np.ndarray:
    # Create a list to store the gradients
    gradients = []

    # Train the model
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Store the gradients
            gradients.append(model[0].weight.grad.numpy().flatten())

    # Return the gradients
    return np.array(gradients)
