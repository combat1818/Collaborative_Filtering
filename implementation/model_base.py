from abc import ABC, abstractmethod

import numpy as np
from utils import extract_matrix_predictions, get_score


class BaseModel(ABC):
    """
    Abstract base class for all models.
    """

    def __init__(self, num_users: int, num_movies: int, min_rating: int, max_rating: int) -> None:
        """
        Initializes the model.

        Args:
            num_users: The number of users.
            num_movies: The number of movies.
            min_rating: The minimum rating.
            max_rating: The maximum rating.
        """
        self.num_users = num_users
        self.num_movies = num_movies
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.model_name = None
        self.reconstructed_matrix = None
        self._params = []

    def set_params(self, params: list) -> None:
        """
        Sets the hyperparameters of the model.

        Args:
            params: List of hyperparameters.
        """
        assert len(params) == len(self._params), f"{self.model_name} takes {len(self._params)} hyperparameter"
        for param_name, param_value in zip(self._params, params):
            setattr(self, param_name, param_value)

    @abstractmethod
    def fit(self, users_train: np.ndarray, movies_train: np.ndarray, ratings_train: np.ndarray) -> None:
        """
        Fits the model to the given data.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
        """

    def predict(self, users_test: np.ndarray, movies_test: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings for the given data.

        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data
        """
        if self.reconstructed_matrix is None:
            self.reconstruct()
        predictions = extract_matrix_predictions(self.reconstructed_matrix, users_test, movies_test)
        return predictions

    @abstractmethod
    def reconstruct(self, *args, **kwargs) -> np.ndarray:
        """
        Returns the reconstructed matrix.
        """

    def evaluate(self, users_test: np.ndarray, movies_test: np.ndarray, ratings_test: np.ndarray) -> float:
        """
        Evaluates the model on the given data.

        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data.
            ratings_test: The ratings of the test data.
        """
        predictions = self.predict(users_test, movies_test)
        score = get_score(predictions, ratings_test)
        return score

    def save_full_matrix(self, path: str = "./reconstructions/") -> None:
        """
        Saves the model to the given path.

        Args:
            path: The path to save the model.
        """
        if self.reconstructed_matrix is None:
            self.reconstruct()
        np.save(path + f"{self.model_name}_full_matrix.npy", self.reconstructed_matrix)

    def save_predictions(self, users_test: np.ndarray, movies_test: np.ndarray, path: str = "./predictions/") -> None:
        """
        Saves the predictions to the given path.

        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data.
            path: The path to save the predictions.
        """
        predictions = self.predict(users_test, movies_test)
        np.save(path + f"{self.model_name}_predictions.npy", predictions)
