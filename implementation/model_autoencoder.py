import numpy as np
import torch
from model_base import BaseModel
from pytorch_models.Autoencoder import Autoencoder
from torch import optim
from utils import (MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS,
                   extract_matrix_predictions, get_dataloader, get_score)


class AutoencoderModel(BaseModel):
    def __init__(self, num_iterations: int = 200) -> None:
        """
        Initializes the model.

        Args:
            num_iterations: The number of iterations.
        """
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder().to(self.device)
        self.num_iterations = num_iterations
        self.model_name = "Autoencoder"
        self.users_train = None
        self.users_val = None
        self.movies_train = None
        self.movies_val = None
        self.ratings_train = None
        self.ratings_val = None

    def fit(
        self,
        users_train: np.ndarray,
        movies_train: np.ndarray,
        ratings_train: np.ndarray,
        users_val: np.ndarray = None,
        movies_val: np.ndarray = None,
        ratings_val: np.ndarray = None,
    ) -> None:
        """
        Fits the model to the given data.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
            users_val: The user ids of the validation data used for validation during training.
            movies_val: The movie ids of the validation data used for validation during training.
            ratings_val: The ratings of the validation data used for validation during training.
        """
        self.users_train = users_train
        self.users_val = users_val
        self.movies_train = movies_train
        self.movies_val = movies_val
        self.ratings_train = ratings_train
        self.ratings_val = ratings_val

        dataloader, _, _, _ = get_dataloader(users_train, movies_train, ratings_train)

        optimizer = optim.Adam(self.model.parameters())
        num_epochs = self.num_iterations

        for epoch in range(num_epochs):
            print(f"{self.model_name}: epoch {epoch+1}/{num_epochs}")
            for _, (data_batch, mask_batch, _) in enumerate(dataloader):
                optimizer.zero_grad()
                reconstructed_batch = self.model(data_batch)
                loss = self.model.loss_function(data_batch, reconstructed_batch, mask_batch)

                loss.backward()
                optimizer.step()

            if not any([users_val is None, movies_val is None, ratings_val is None]):
                self.model.eval()
                reconstructed_matrix = self.reconstruct()
                predictions = extract_matrix_predictions(reconstructed_matrix, users_val, movies_val)
                reconstruction_rmse = get_score(predictions, ratings_val)

                print(f"Reconstruction rmse {reconstruction_rmse}")

            self.model.train()

        self.reconstruct()

    def reconstruct(self) -> np.ndarray:
        """
        Returns the reconstructed matrix.
        """
        if any(matrix is None for matrix in [self.users_train, self.movies_train, self.ratings_train]):
            raise ValueError("Model has not been trained yet.")
        _, data_tensor, _, _ = get_dataloader(self.users_train, self.movies_train, self.ratings_train)
        data_reconstructed = self.model(data_tensor)
        reconstructed_matrix = data_reconstructed.detach().cpu().numpy().clip(self.min_rating, self.max_rating)
        self.reconstructed_matrix = reconstructed_matrix
        return reconstructed_matrix
