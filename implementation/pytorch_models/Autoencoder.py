import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 256, encoded_dimension: int = 32) -> None:
        """
        Initialize the Encoder class

        Args:
            input_dimension: The dimension of the input data.
            hidden_dimension: The dimension of the hidden layer.
            encoded_dimension: The dimension of the encoded layer
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dimension, out_features=hidden_dimension),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dimension, out_features=encoded_dimension),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data: The input data.
        """
        return self.model(data)


class Decoder(nn.Module):
    def __init__(self, output_dimension: int, hidden_dimension: int = 256, encoded_dimension: int = 32) -> None:
        """
        Initialize the Decoder class

        Args:
            output_dimension: The dimension of the output data.
            hidden_dimension: The dimension of the hidden layer.
            encoded_dimension: The dimension of the encoded layer
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=encoded_dimension, out_features=hidden_dimension),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dimension, out_features=output_dimension),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data: The input data.
        """
        return self.model(data)


class Autoencoder(nn.Module):
    def __init__(
        self, num_users: int = 10000, num_movies: int = 1000, hidden_dimension: int = 256, encoded_dimension: int = 32
    ) -> None:
        """
        Initialize the Autoencoder class

        Args:
            num_users: The number of users.
            num_movies: The number of movies.
            hidden_dimension: The dimension of the hidden layer.
            encoded_dimension: The dimension of the encoded layer
        """
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies

        hidden_dimension = hidden_dimension
        encoded_dimension = encoded_dimension

        self.encoder = Encoder(
            input_dimension=self.num_movies, hidden_dimension=hidden_dimension, encoded_dimension=encoded_dimension
        )
        self.decoder = Decoder(
            output_dimension=self.num_movies, hidden_dimension=hidden_dimension, encoded_dimension=encoded_dimension
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. This function normalizee, encodes and then decodes the input data.
        """
        data = F.normalize(data)
        data = self.decoder(self.encoder(data))

        return data

    @staticmethod
    def loss_function(original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Loss function for the autoencoder model.
        
        Args:
            original: The original data.
            reconstructed: The reconstructed data.
            mask: The mask for the data.
        """
        return torch.mean(mask * (original - reconstructed) ** 2)
