import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

# Set parameters for training
BATCH_SIZE = 64
EPOCHS = 10


def prepare_dataset(normalize_input: bool = False) -> DataLoader:
    # Create a transform pipeline
    if not normalize_input:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

    # Load the MNIST dataset using torchvision
    dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader


class Encoder(nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int], latent_dim: int, *args, **kwargs):
        # Make super call and store input_shape and latent_dim
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Create the encoder layers
        self.encoder_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape[1] * input_shape[2], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the encoder layers to the input
        return self.encoder_layers(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim: int, output_shape: Tuple[int, int, int], *args, **kwargs):
        # Make super call and store latent_dim and output_shape
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        # Create the decoder layers
        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_shape[1] * output_shape[2]),
            nn.Tanh(),
            nn.Unflatten(dim=-1, unflattened_size=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the decoder layers to the input
        return self.decoder_layers(x)


class AutoEncoder(nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int], latent_dim: int, *args, **kwargs):
        # Make super call and store latent_dim
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim

        # Create the encoder and decoder
        self.encoder = Encoder(input_shape=input_shape, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_shape=input_shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the encoder and decoder to the input
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    # Load the MNIST dataset
    data_loader = prepare_dataset(normalize_input=True)

    # Create the autoencoder
    autoencoder = AutoEncoder(input_shape=(1, 28, 28), latent_dim=2)

    # Create the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Train the autoencoder
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        # Iterate over the data
        for batch_idx, (data, _) in enumerate(data_loader):
            print(f'Batch {batch_idx + 1}/{len(data_loader)}', end='\r')

            # Reset the gradients
            optimizer.zero_grad()

            # Calculate the output and loss
            output = autoencoder(data)
            loss = loss_function(output, data)
            loss.backward()

            # Update the parameters
            optimizer.step()
    print('Training completed...')

    # Plot the first 5 images and their reconstructions
    with torch.no_grad():
        data, labels = next(iter(data_loader))
        output = autoencoder(data[:5])

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle('Sample Reconstruction', fontsize=16)
    for i in range(5):
        axes[0, i].imshow(data[i, 0], cmap='gray')
        axes[1, i].imshow(output[i, 0], cmap='gray')
    fig.savefig('Figures/exercise_09/Sample_Reconstruction.png')
    fig.show()

    # Encode the entire dataset to get the latent space
    latent_space, labels = None, None
    with torch.no_grad():
        for data, l in data_loader:
            encoded = autoencoder.encode(data).cpu().detach().numpy()
            if latent_space is None:
                latent_space = encoded
                labels = l.cpu().detach().numpy()
            else:
                latent_space = np.concatenate([latent_space, encoded], axis=0)
                labels = np.concatenate([labels, l], axis=0)

    # Plot the latent space
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    fig.suptitle('Latent Space', fontsize=16)
    sns.scatterplot(x=latent_space[:, 0], y=latent_space[:, 1], hue=labels.astype(str), ax=ax)
    fig.savefig('Figures/exercise_09/Latent_Space.png')
    fig.show()

    # Pick two random points from the latent space and interpolate with a line
    start_point, end_point = latent_space[np.random.choice(len(latent_space), 2)]
    interpolated_points = np.array(list(zip(np.linspace(start_point[0], end_point[0], 11),
                                            np.linspace(start_point[1], end_point[1], 11))))

    # Predict the interpolated points and plot the results
    interpolated_predictions = autoencoder.decode(torch.from_numpy(interpolated_points.astype(np.float32))
                                                  ).cpu().detach().numpy()
    fig, axs = plt.subplots(ncols=len(interpolated_points), tight_layout=True, figsize=(12, 2))
    fig.suptitle('Interpolated Transformation', fontsize=16)
    for i, image in enumerate(interpolated_predictions):
        axs[i].imshow(np.moveaxis(image, 0, -1), cmap='gray')
        axs[i].axis('off')
    fig.savefig('Figures/exercise_09/Interpolated_Transformation.png')
    fig.show()
