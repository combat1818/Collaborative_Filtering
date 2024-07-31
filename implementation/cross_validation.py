import itertools

import numpy as np
from model_base import BaseModel
from sklearn.model_selection import KFold
from utils import get_data


def cross_validate(
    users: np.ndarray, movies: np.ndarray, ratings: np.ndarray, parameters: list, n_splits: int, model: BaseModel
) -> tuple[list, float]:
    """
    Cross validates the model with the given parameters.

    Args:
        users: The user ids.
        movies: The movie ids.
        ratings: The ratings.
        parameters: The parameters to cross validate.
        n_splits: The number of splits.
        model: The model to cross validate.
    """

    best_params = None
    best_rmse = float("inf")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    combinations = list(itertools.product(*parameters))

    for params in combinations:
        print(f"Evaluating params = {params}")
        # model.param = param
        model.set_params(params)

        rmse_scores = []

        for train_index, test_index in kf.split(users):
            users_train, users_test = users[train_index], users[test_index]
            movies_train, movies_test = movies[train_index], movies[test_index]
            ratings_train, ratings_test = ratings[train_index], ratings[test_index]

            model.fit(users_train, movies_train, ratings_train)

            rmse = model.evaluate(users_test, movies_test, ratings_test)
            rmse_scores.append(rmse)

        mean_rmse = np.mean(rmse_scores)
        print(f"params={params}, Average RMSE: {mean_rmse}")

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    print(f"Best params: {best_params} with RMSE: {best_rmse}")
    return best_params, best_rmse


# model = KNNMeansModel()
# train_users, train_movies, train_ratings, test_users, test_movies, test_ratings = get_data()

# svd_rank = [6, 7, 8, 9]
# als_rank = [1, 2, 3, 4, 5, 6]
# schrinkage = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

# rsvd_rank = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# svdpp_nfactors = [2, 4, 6, 8, 10]

# knn_means_neighbors = [10, 20, 30, 40, 50, 60, 70, 80]

# parameters = [knn_means_neighbors]


# n_splits = 5

# cross_validate(train_users, train_movies, train_ratings, parameters, n_splits, model)
