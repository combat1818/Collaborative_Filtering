import numpy as np
import pandas as pd
from model_base import BaseModel
from surprise import Dataset, KNNBaseline, KNNWithMeans, KNNWithZScore, Reader
from utils import MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS, get_score


class KNNMeansModel(BaseModel):
    def __init__(self, neighbors=300, knn_type="KNNBaseline") -> None:
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)
        self.neighbors = neighbors
        self.knn_type = knn_type
        self.model_name = "KNNMeans"
        self._params = ["neighbors", "knn_type"]
        self.set_params([self.neighbors, self.knn_type])

    def set_params(self, params: list) -> None:
        """
        Sets the hyperparameters of the model.

        Args:
            params: List of hyperparameters.
        """
        assert len(params) == len(self._params), f"{self.model_name} takes {len(self._params)} hyperparameter"
        for param_name, param_value in zip(self._params, params):
            setattr(self, param_name, param_value)
        if self.knn_type == "KNNWithMeans":
            self.model = KNNWithMeans(k=self.neighbors)
        elif self.knn_type == "KNNWithZScore":
            self.model = KNNWithZScore(k=self.neighbors)
        else:
            self.model = KNNBaseline(k=self.neighbors)

    def fit(self, users_train: np.ndarray, movies_train: np.ndarray, ratings_train: np.ndarray) -> None:
        """
        Fits the model to the given data.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
        """
        data = pd.DataFrame({"userID": users_train, "itemID": movies_train, "rating": ratings_train})

        reader = Reader(rating_scale=(self.min_rating, self.max_rating))

        data = Dataset.load_from_df(data[["userID", "itemID", "rating"]], reader)
        trainset = data.build_full_trainset()

        self.model.fit(trainset)

    def predict_single(self, user_id: int, movie_id: int) -> float:
        """
        Predicts the rating for a single user-movie pair.

        Args:
            user_id: The user id.
            movie_id: The movie id.
        """
        return np.clip(self.model.predict(user_id, movie_id).est, self.min_rating, self.max_rating)

    def predict(self, users_test: np.ndarray, movies_test: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings for a given set of user-movie pairs.
        
        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data.
        """
        return np.array([self.predict_single(user, movie) for user, movie in zip(users_test, movies_test)])

    def reconstruct(self) -> np.ndarray:
        """
        Returns the reconstructed matrix.
        """
        reconstructed_matrix = np.zeros((self.num_users, self.num_movies))

        for user in range(self.num_users):
            for movie in range(self.num_movies):
                prediction = self.predict_single(user, movie)
                reconstructed_matrix[user, movie] = prediction
        self.reconstructed_matrix = reconstructed_matrix
        return self.reconstructed_matrix
