import numpy as np
from model_base import BaseModel
from utils import (MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS,
                   create_ratings_matrix, denorm, impute, norm)


class ALSModel(BaseModel):
    def __init__(self, rank_svd: int = 6, rank_als: int = 3, num_iterations: int = 15, lambda_als: float = 0.1) -> None:
        """
        Initializes the model.

        Args:
            rank_svd: The rank of the SVD model.
            rank_als: The rank of the ALS model.
            num_iterations: The number of iterations.
            lambda_als: The regularization parameter.
        """
        # best cross-validated: rank_svd = 6, rank_als = 3
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)
        self.rank_svd = rank_svd
        self.rank_als = rank_als
        self.num_iterations = num_iterations  # best 15
        self.lambda_als = lambda_als
        self.model_name = "ALS"
        self._params = ["rank_svd", "rank_als"]
        self.u_matrix, self.v_matrix = None, None

    def als_algorithm(
        self, matrix: np.ndarray, mask_matrix: np.ndarray, k: int = 3, n_itr: int = 20, lambda_: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:

        n_users, n_items = matrix.shape
        u_matrix, _, vt_matrix = self.compute_svd(matrix, self.num_movies, self.rank_svd)
        u_matrix = u_matrix[:, :k]
        vk_matrix = vt_matrix[:k, :]

        eye_k = np.eye(k)

        for iteration in range(n_itr):
            for user_idx in range(n_users):
                mask_user = mask_matrix[user_idx]
                masked_v = vk_matrix * mask_user
                user_a_matrix = np.dot(masked_v, vk_matrix.T) + lambda_ * eye_k
                user_b = np.dot(masked_v, matrix[user_idx].T)
                u_matrix[user_idx] = np.linalg.solve(user_a_matrix, user_b).T

            for item_idx in range(n_items):
                mask_item = mask_matrix[:, item_idx]
                masked_u = u_matrix.T * mask_item
                item_a_matrix = np.dot(masked_u, u_matrix) + lambda_ * eye_k
                item_b = np.dot(masked_u, matrix[:, item_idx])
                vk_matrix[:, item_idx] = np.linalg.solve(item_a_matrix, item_b)

            print(f"{self.model_name}: iteration {iteration + 1}/{n_itr} completed")

        return u_matrix, vk_matrix

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
        Fits the model to the given data, reconstructs the full matrix.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
        """
        # Create full matrix, calculate mean, std and impute missing values with 0
        ratings_matrix, mask = create_ratings_matrix(
            self.num_users, self.num_movies, users_train, movies_train, ratings_train, impute_value=np.nan
        )
        ratings_matrix, col_mean, col_std = norm(ratings_matrix)
        ratings_matrix = impute(ratings_matrix)

        # Run ALS.
        u_matrix, v_matrix = self.als_algorithm(
            ratings_matrix, mask, self.rank_als, self.num_iterations, self.lambda_als
        )
        self.u_matrix = u_matrix
        self.v_matrix = v_matrix

        self.reconstruct(col_mean, col_std)

    def reconstruct(self, col_mean: np.ndarray, col_std: np.ndarray) -> np.ndarray:
        """
        Reconstructs the full matrix from the SVD matrices and denormalizes it.

        Args:
            col_mean: The column mean.
            col_std: The column standard deviation.
        """
        if any([matrix is None for matrix in [self.u_matrix, self.v_matrix]]):
            raise ValueError("Model has not been trained yet.")
        reconstructed_matrix = np.dot(self.u_matrix, self.v_matrix)
        reconstructed_matrix = denorm(reconstructed_matrix, col_mean, col_std).clip(self.min_rating, self.max_rating)
        self.reconstructed_matrix = reconstructed_matrix
