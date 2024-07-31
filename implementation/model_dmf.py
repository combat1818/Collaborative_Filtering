import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_base import BaseModel
from pytorch_models.DMF import DMF
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS,
                   RatingsDataset, create_ratings_matrix, get_score)


class DMFModel(BaseModel):
    def __init__(
        self, epochs: int = 10, batch_size: int = 512, neg_ratio: int = 7, mu: float = 10e-6, lr: float = 0.0001
    ) -> None:
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DMF(self.num_users, self.num_movies).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size

        self.mu = mu
        self.neg_ratio = neg_ratio
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def save_model(self, file_path: str) -> None:
        """
        Save the model state dict to the given file path.
        """
        torch.save(self.model.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        Load the model state dict from the given file path.
        """
        self.model.load_state_dict(torch.load(file_path))
        self.model.to(self.device)
        print(f"Model weights loaded from {file_path}")

    @staticmethod
    def nce_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the noise contrastive estimation loss.
        """
        y_true = y_true / MAX_RATING
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss.sum()

    def get_matrix_T_Y(
        self, users: np.ndarray, movies: np.ndarray, ratings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the matrix T and Y as described in the paper.

        Args:
            users: The user ids.
            movies: The movie ids.
            ratings: The ratings.
        """
        R, Y_bin = create_ratings_matrix(self.num_users, self.num_movies, users, movies, ratings, impute_value=np.nan)

        # create matrix described in equation (2) of paper
        Y = np.zeros(R.shape)
        Y[Y_bin > 0] = R[Y_bin > 0]

        Y_plus = np.argwhere(Y_bin > 0)  # positive interactions (numsamples x 2 dim array)
        Y_minus = np.argwhere(Y_bin == 0)  # negative interactions ( ((numusers x numitems) - numsamples) x 2 dim array)
        Y_minus_sampled = Y_minus[
            np.random.choice(Y_minus.shape[0], int(self.neg_ratio * Y_plus.shape[0]), replace=False)
        ]  # negative sampling
        T = np.vstack([Y_plus, Y_minus_sampled])

        return T, Y

    def fit(
        self,
        users_train: np.ndarray,
        movies_train: np.ndarray,
        ratings_train: np.ndarray,
        users_test: np.ndarray,
        movies_test: np.ndarray,
        ratings_test: np.ndarray,
    ) -> None:
        """
        Fit the model to the given data.
        
        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data.
            ratings_test: The ratings of the test data.
        """
        print("Creating ratings matrix...")

        T_train, Y_train = self.get_matrix_T_Y(users_train, movies_train, ratings_train)
        _, Y_test = self.get_matrix_T_Y(users_test, movies_test, ratings_test)

        train_dataset = RatingsDataset(T_train, Y_train)
        test_dataset = RatingsDataset(np.argwhere(Y_test > 0), Y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        criterion = nn.BCEWithLogitsLoss(reduction="sum")

        print("Training initialized.")

        for epoch in range(self.epochs):
            start_time = time.time()
            print(f"Epoch {epoch+1}/{self.epochs}")
            epoch_loss = 0

            self.model.train()
            with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
                for user_vect_batch, movie_vect_batch, ratings_batch in train_loader:
                    user_vect_batch = user_vect_batch.to(self.device)
                    movie_vect_batch = movie_vect_batch.to(self.device)
                    ratings_batch = ratings_batch.to(self.device)

                    Y_hat_ij = self.model(user_vect_batch, movie_vect_batch)
                    Y_hat_o_ij = torch.max(torch.tensor(self.mu, device=self.device), Y_hat_ij)

                    loss = self.nce_loss(ratings_batch, Y_hat_o_ij)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    pbar.update()

            epoch_time = time.time() - start_time
            print(
                f"End of Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / self.batch_size}, Time: {epoch_time:.2f}s"
            )

            self.model.eval()
            score = 0
            total_samples = 0
            with torch.no_grad():
                for user_vect_batch, movie_vect_batch, ratings_batch in test_loader:
                    user_vect_batch = user_vect_batch.to(self.device)
                    movie_vect_batch = movie_vect_batch.to(self.device)

                    Y_hat_ij = self.model(user_vect_batch, movie_vect_batch)
                    Y_hat_o_ij = torch.max(torch.tensor(self.mu, device=self.device), Y_hat_ij)
                    print(Y_hat_o_ij)

                    predictions = Y_hat_o_ij * (self.max_rating - self.min_rating) + self.min_rating

                    squared_errors = (ratings_batch.cpu().numpy() - predictions.cpu().numpy()) ** 2
                    score += np.sum(squared_errors)
                    total_samples += len(ratings_batch)

            rmse = np.sqrt(score / total_samples)

            self.save_model("dmf_model_weights.pth")
            print(f"Evaluation RMSE of Epoch: {rmse}")

    def predict(self, users_embedded: np.ndarray, movies_embedded: np.ndarray):
        """
        Predict the ratings for the given data.
        
        Args:
            users_embedded: The user vector.
            movies_embedded: The movie vector.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(users_embedded, movies_embedded)
            predictions = predictions.detach().cpu().numpy()
            predictions = predictions * (self.max_rating - self.min_rating) + self.min_rating
            predictions = np.clip(predictions, self.min_rating, self.max_rating)
        return predictions

    def evaluate(self, users_test: np.ndarray, movies_test: np.ndarray, ratings_test: np.ndarray) -> float:
        """
        Evaluate the model on the given data.
        
        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data.
            ratings_test: The ratings of the test data.
        """
        print("Evaluating model...")

        self.load_model("dmf_model_weights.pth")

        # embed data
        _, Y_test = self.get_matrix_T_Y(users_test, movies_test, ratings_test)
        users_embedded = torch.tensor(Y_test[users_test, :], dtype=torch.float, device=self.device)
        movies_embedded = torch.tensor(Y_test[:, movies_test], dtype=torch.float, device=self.device).T

        # predict
        predictions = self.predict(users_embedded, movies_embedded)
        target = ratings_test

        print(predictions)
        print(target)

        # compare
        score = get_score(predictions, target)
        return score
