import numpy as np
import pandas as pd
from model_base import BaseModel
from sklearn.utils import shuffle
from surprise import Dataset, Reader, SVDpp
from utils import (MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS,
                   create_ratings_matrix, denorm, impute, norm)


class SVDModel(BaseModel):
    def __init__(self, rank: int = 8) -> None:
        """
        Initializes the model.

        Args:
            rank: The rank of the SVD.
        """
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)
        self.svd_rank = rank  # 7 or 8 is best - cross validated
        self.model_name = "SVD"
        self._params = ["svd_rank"]
        self.u_matrix, self.s_matrix, self.vt_matrix = None, None, None

    @staticmethod
    def compute_svd(matrix: np.ndarray, num_movies: int, k: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the SVD of the given matrix.

        Args:
            matrix: The matrix to compute the SVD of.
            num_movies: The number of movies.
            k: The rank of the SVD.
        """
        u_matrix, s_matrix, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
        sk_matrix = np.zeros((num_movies, num_movies))
        sk_matrix[:k, :k] = np.diag(s_matrix[:k])
        return u_matrix, sk_matrix, vt_matrix

    def fit(self, users_train: np.ndarray, movies_train: np.ndarray, ratings_train: np.ndarray) -> None:
        """
        Fits the model to the given data, computes the SVD and reconstructs the full matrix.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
        """
        ratings_matrix, _ = create_ratings_matrix(
            self.num_users, self.num_movies, users_train, movies_train, ratings_train, impute_value=np.nan
        )

        ratings_matrix, col_mean, col_std = norm(ratings_matrix)
        ratings_matrix = impute(ratings_matrix)

        # Run SVD
        u_matrix, s_matrix, vt_matrix = self.compute_svd(ratings_matrix, self.num_movies, self.svd_rank)
        self.u_matrix = u_matrix
        self.s_matrix = s_matrix
        self.vt_matrix = vt_matrix

        # Reconstruct the full matrix
        self.reconstruct(col_mean, col_std)

    def reconstruct(self, col_mean: np.ndarray, col_std: np.ndarray) -> np.ndarray:
        """
        Reconstructs the full matrix from the SVD matrices and denormalizes it.

        Args:
            col_mean: The column mean.
            col_std: The column standard deviation.
        """
        if any([matrix is None for matrix in [self.u_matrix, self.s_matrix, self.vt_matrix]]):
            raise ValueError("Model has not been trained yet.")
        reconstructed_matrix = np.dot(self.u_matrix, np.dot(self.s_matrix, self.vt_matrix))
        reconstructed_matrix = denorm(reconstructed_matrix, col_mean, col_std).clip(self.min_rating, self.max_rating)
        self.reconstructed_matrix = reconstructed_matrix
        return self.reconstructed_matrix


class SVDppModel(BaseModel):
    def __init__(self, rank: int = 3, n_epochs: int = 70, verbose: bool = True) -> None:
        """
        Initializes the model.

        Args:
            rank: The rank of the SVD.
            n_epochs: The number of epochs.
            verbose: Whether to print the output.
        """
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)
        self.svd_rank = rank
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.model = SVDpp(n_factors=self.svd_rank, n_epochs=self.n_epochs, verbose=self.verbose)
        self.model_name = "SVDpp"
        self._params = ["svd_rank"]

    def set_params(self, params: list) -> None:
        """
        Sets the hyperparameters of the model.

        Args:
            params: List of hyperparameters.
        """
        assert len(params) == len(self._params), f"{self.model_name} takes {len(self._params)} hyperparameter"
        for param_name, param_value in zip(self._params, params):
            setattr(self, param_name, param_value)
        self.model = SVDpp(n_factors=self.svd_rank, n_epochs=self.n_epochs, verbose=self.verbose)
        print(f"n_factors = {self.model.n_factors}")

    def fit(self, users_train: np.ndarray, movies_train: np.ndarray, ratings_train: np.ndarray) -> None:
        """
        Fits the model to the given data.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
        """
        data = pd.DataFrame({"userID": users_train, "itemID": movies_train, "rating": ratings_train})
        reader = Reader(rating_scale=(1, 5))
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
        return self.model.predict(user_id, movie_id).est

    def predict(self, users_test: np.ndarray, movies_test: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings for the given data.

        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data
        """
        return np.array([self.predict_single(user, movie) for user, movie in zip(users_test, movies_test)])

    def reconstruct(self) -> np.ndarray:
        """
        Returns the reconstructed matrix.
        """
        self.reconstructed_matrix = np.zeros((self.num_users, self.num_movies))

        for user in range(self.num_users):
            for movie in range(self.num_movies):
                prediction = self.predict_single(user, movie)
                self.reconstructed_matrix[user, movie] = prediction[0]
        return self.reconstructed_matrix


class RSVDModel(BaseModel):
    def __init__(
        self,
        rank: int = 12,
        lambda1: float = 0.07,
        lambda2: float = 0.05,
        lr: float = 0.025,
        num_iterations: int = 15,
        decay_iterations: int = 5,
        decay_rate: float = 0.997,
    ) -> None:
        """
        Initializes the model.

        Args:
            rank: The rank of the SVD.
            lambda1: The regularization parameter for the user and movie features.
            lambda2: The regularization parameter for the user and movie biases.
            lr: The learning rate.
            num_iterations: The number of iterations.
            decay_iterations: The number of iterations after which to decay the learning rate.
            decay_rate: The decay rate of the learning rate.
        """
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)
        self.rank_svd = rank
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.num_iterations = num_iterations
        self.decay_iterations = decay_iterations
        self.decay_rate = decay_rate
        self.model_name = "RSVD"
        self._params = ["rank_svd", "lambda1", "lambda2"]
        self.u_matrix, self.v_matrix = None, None
        self.u_bias, self.v_bias = None, None

    def fit(self, users_train: np.ndarray, movies_train: np.ndarray, ratings_train: np.ndarray) -> None:
        """
        Fits the model to the given data.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
        """
        self.u_matrix = np.random.uniform(0, 0.01, (self.num_users, self.rank_svd))
        self.v_matrix = np.random.uniform(0, 0.01, (self.num_movies, self.rank_svd))

        self.u_bias = np.zeros(self.num_users)
        self.v_bias = np.zeros(self.num_movies)
        learning_rate = self.lr
        overall_mean = np.mean(ratings_train)
        total_epochs = self.num_iterations

        for epoch in range(1, total_epochs + 1):
            print(f"Epoch {epoch}")
            # Shuffle data for stochastic gradient descent
            shuffled_users, shuffled_movies, shuffled_ratings = shuffle(users_train, movies_train, ratings_train)

            for user, movie, rating in zip(shuffled_users, shuffled_movies, shuffled_ratings):
                # Retrieve user and movie features
                user_features = self.u_matrix[user, :]
                movie_features = self.v_matrix[movie, :]
                user_bias = self.u_bias[user]
                movie_bias = self.v_bias[movie]
                prediction = user_features.dot(movie_features)

                prediction += user_bias
                prediction += movie_bias
                error = rating - prediction

                # Perform gradient descent update
                self.u_matrix[user, :] += learning_rate * (error * movie_features - self.lambda1 * user_features)
                self.v_matrix[movie, :] += learning_rate * (error * user_features - self.lambda1 * movie_features)
                user_bias_update = learning_rate * (error - self.lambda2 * (user_bias - overall_mean + movie_bias))
                movie_bias_update = learning_rate * (error - self.lambda2 * (movie_bias - overall_mean + user_bias))
                self.u_bias[user] += user_bias_update
                self.v_bias[movie] += movie_bias_update

            # Adjust learning rate at specified intervals
            if epoch % self.decay_iterations == 0:
                learning_rate *= self.decay_rate

    def predict_single(self, user_id: int, movie_id: int) -> float:
        """
        Predicts the rating for a single user-movie pair.

        Args:
            user_id: The user id.
            movie_id: The movie id.
        """
        prediction = np.dot(self.u_matrix[user_id, :], self.v_matrix[movie_id, :])
        prediction += self.u_bias[user_id]
        prediction += self.v_bias[movie_id]
        prediction = np.clip(prediction, self.min_rating, self.max_rating)
        return prediction

    def predict(self, users_test: np.ndarray, movies_test: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings for the given data.

        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data
        """
        return np.array([self.predict_single(user, movie) for user, movie in zip(users_test, movies_test)])

    def reconstruct(self) -> np.ndarray:
        """
        Returns the reconstructed matrix.
        """
        self.reconstructed_matrix = np.zeros((self.num_users, self.num_movies))

        for user in range(self.num_users):
            for movie in range(self.num_movies):
                prediction = self.predict_single(user, movie)
                self.reconstructed_matrix[user, movie] = prediction[0]
        return self.reconstructed_matrix


class IterativeSVDModel(BaseModel):
    def __init__(self, num_iterations: int = 15, shrinkage: int = 32) -> None:
        """
        Initializes the model.

        Args:
            num_iterations: The number of iterations to perform.
            shrinkage: The shrinkage to apply to the singular values.
        """
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)
        self.num_iterations = num_iterations
        self.shrinkage = shrinkage  # cross-validated: best 35 Best params: (35,) with RMSE: 0.989327026716
        self.model_name = "IterSVD"
        self._params = ["shrinkage"]

    def iterate_svd(self, matrix: np.ndarray, matrix_mask: np.ndarray, shrinkage: int, iterations: int) -> np.ndarray:
        """
        Computes the SVD of the given matrix and applies shrinkage to the singular values.

        Args:
            matrix: The matrix to compute the SVD of.
            matrix_mask: The mask indicating the positions of the original ratings.
            shrinkage: The shrinkage to apply to the singular values.
            iterations: The number of iterations to perform
        """
        final_matrix = matrix.copy()
        for iteration in range(iterations):
            # Perform SVD on the current matrix
            u_matrix, s_matrix, vt_matrix = np.linalg.svd(final_matrix, full_matrices=False)
            # Apply shrinkage to the singular values
            s_shrinked = np.maximum(s_matrix - shrinkage, 0)
            # Reconstruct the matrix from the truncated SVD
            final_matrix = np.dot(u_matrix, np.dot(np.diag(s_shrinked), vt_matrix))
            # Restore the original ratings at the positions indicated by mask_A
            np.putmask(final_matrix, matrix_mask, matrix)
            print(f"{self.model_name}: iteration {iteration + 1}/{iterations} completed")

        # Clip the values of the reconstructed matrix to be within the allowed rating range
        final_matrix = np.clip(final_matrix, self.min_rating, self.max_rating)
        return final_matrix

    def fit(self, users_train: np.ndarray, movies_train: np.ndarray, ratings_train: np.ndarray) -> None:
        """
        Fits the model to the given data.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
        """
        # Create full matrix, calculate mean, std and impute missing values with 0
        ratings_matrix, mask = create_ratings_matrix(
            self.num_users, self.num_movies, users_train, movies_train, ratings_train, impute_value=np.nan
        )

        ratings_matrix = impute(A=ratings_matrix, impute_method="col_mean")
        # Perform SVD.
        ratings_matrix = self.iterate_svd(ratings_matrix, mask, self.shrinkage, self.num_iterations)

        self.reconstructed_matrix = ratings_matrix

    def reconstruct(self) -> np.ndarray:
        """
        Returns the reconstructed matrix.
        """
        if self.reconstructed_matrix is None:
            raise ValueError("Model has not been trained yet.")
        return self.reconstructed_matrix
