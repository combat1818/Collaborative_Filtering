import os
from typing import Any

import numpy as np
import pandas as pd
from model_als import ALSModel
from model_autoencoder import AutoencoderModel
from model_base import BaseModel
from model_knn_means import KNNMeansModel
from model_ncf import NCFModel
from model_svd import IterativeSVDModel, RSVDModel, SVDModel, SVDppModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from utils import (calculate_stats, extract_users_items_ratings,
                   generate_submission)
from xgboost import XGBRegressor


class Stacking:
    def __init__(
        self,
        train_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        train_for_submission: bool,
        meta_model: Any,
        base_models: list[BaseModel],
        retrain_base_models: list[bool],
        meta_model_name: str,
        add_stats_features: bool = False,
    ) -> None:
        """
        Initializes the stacking model.

        Args:
            train_df: The training data.
            predict_df: The test data.
            train_for_submission: Whether to train the model for submission or not.
            meta_model: The meta-model to use for blending.
            base_models: The base models to use for blending.
            retrain_base_models: Whether to retrain the base models or not, should be a list of the same length as base_models.
            meta_model_name: The name of the meta-model.
            add_stats_features: Whether to add statistics based features or not.
        """
        self.meta_model = meta_model
        self.base_models = base_models
        self.retrain_base_models = retrain_base_models
        self.training_type = "submission" if train_for_submission else "train"
        self.meta_model_name = meta_model_name
        self.add_stats_features = add_stats_features

        if train_for_submission:
            self.train_users, self.train_movies, self.train_ratings = extract_users_items_ratings(train_df)
            self.test_users, self.test_movies, self.test_ratings = extract_users_items_ratings(predict_df)
        else:
            # Split training data into training and validation sets for blending
            test_size = 0.2
            train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=42)

            self.train_users, self.train_movies, self.train_ratings = extract_users_items_ratings(train_df)
            self.test_users, self.test_movies, self.test_ratings = extract_users_items_ratings(test_df)

        print(f"Number of training samples: {len(self.train_users)}, number of test samples: {len(self.test_users)}")

    def fit(self) -> None:
        """
        Train base models on the OOF predictions and the meta-model on these predictions.
        The OOF predictions are generated using K-Fold cross-validation.
        """
        # Create arrays to hold out-of-fold predictions and test set predictions
        add_stats_features_number = 0 if not self.add_stats_features else 9 * 2
        oof_predictions = np.zeros((len(self.train_users), len(self.base_models) + add_stats_features_number))

        # Perform K-Fold cross-validation and generate out-of-fold predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, model in enumerate(self.base_models):
            model_name = model.model_name

            oof_filename = (
                f"./predictions/{self.meta_model_name}_{self.training_type}_stacking_oof_predictions_{model_name}.npy"
            )

            if self.retrain_base_models[i] or not os.path.exists(oof_filename):
                print(f"Training model: {model_name}")
                for train_idx, val_idx in kf.split(self.train_users):
                    users_train_fold, users_val_fold = self.train_users[train_idx], self.train_users[val_idx]
                    movies_train_fold, movies_val_fold = self.train_movies[train_idx], self.train_movies[val_idx]
                    ratings_train_fold, ratings_val_fold = self.train_ratings[train_idx], self.train_ratings[val_idx]

                    model.fit(users_train_fold, movies_train_fold, ratings_train_fold)
                    oof_predictions[val_idx, i] = model.predict(users_val_fold, movies_val_fold)

                # Save out-of-fold predictions to file
                np.save(oof_filename, oof_predictions[:, i])
            else:
                print(f"Loading precomputed predictions for model: {model_name}")
                oof_predictions[:, i] = np.load(oof_filename)
        if self.add_stats_features:
            print("Calculating stats")
            cached_user_stats = {}
            cached_movie_stats = {}
            for train_idx, val_idx in kf.split(self.train_users):
                users_train_fold, users_val_fold = self.train_users[train_idx], self.train_users[val_idx]
                movies_train_fold, movies_val_fold = self.train_movies[train_idx], self.train_movies[val_idx]
                ratings_train_fold, ratings_val_fold = self.train_ratings[train_idx], self.train_ratings[val_idx]
                for val_single_idx in val_idx:
                    user = self.train_users[val_single_idx]
                    movie = self.train_movies[val_single_idx]
                    if user not in cached_user_stats:
                        cached_user_stats[user] = calculate_stats(ratings_train_fold[users_train_fold == user])
                    if movie not in cached_movie_stats:
                        cached_movie_stats[movie] = calculate_stats(ratings_train_fold[movies_train_fold == movie])
                    user_stats, movie_stats = cached_user_stats[user], cached_movie_stats[movie]
                    oof_predictions[val_single_idx, len(self.base_models) : len(self.base_models) + len(user_stats)] = (
                        user_stats
                    )
                    oof_predictions[
                        val_single_idx,
                        len(self.base_models)
                        + len(user_stats) : len(self.base_models)
                        + len(user_stats)
                        + len(movie_stats),
                    ] = movie_stats

        # Train the meta-model
        self.meta_model.fit(oof_predictions, self.train_ratings)

    def predict(self) -> np.ndarray:
        """
        Train base models on the full training data and generate test set predictions and make final predictions with the previously trained blending model.
        """
        add_stats_features_number = 0 if not self.add_stats_features else 9 * 2
        test_predictions = np.zeros((len(self.test_users), len(self.base_models) + add_stats_features_number))

        for i, model in enumerate(self.base_models):
            model_name = model.model_name
            test_filename = (
                f"./predictions/{self.meta_model_name}_{self.training_type}_stacking_test_predictions_{model_name}.npy"
            )

            if self.retrain_base_models[i] or not os.path.exists(test_filename):
                print(f"Training model: {model_name} on full training data for test predictions")
                model.fit(self.train_users, self.train_movies, self.train_ratings)
                test_predictions[:, i] = model.predict(self.test_users, self.test_movies)

                # Save test predictions to file
                np.save(test_filename, test_predictions[:, i])
            else:
                print(f"Loading precomputed test predictions for model: {model_name}")
                test_predictions[:, i] = np.load(test_filename)

        if self.add_stats_features:
            cached_user_stats = {}
            cached_movie_stats = {}
            for test_single_idx in range(len(self.test_users)):
                user = self.test_users[test_single_idx]
                movie = self.test_movies[test_single_idx]
                if user not in cached_user_stats:
                    cached_user_stats[user] = calculate_stats(self.train_ratings[self.train_users == user])
                if movie not in cached_movie_stats:
                    cached_movie_stats[movie] = calculate_stats(self.train_ratings[self.train_movies == movie])
                user_stats, movie_stats = cached_user_stats[user], cached_movie_stats[movie]
                test_predictions[test_single_idx, len(self.base_models) : len(self.base_models) + len(user_stats)] = (
                    user_stats
                )
                test_predictions[
                    test_single_idx,
                    len(self.base_models)
                    + len(user_stats) : len(self.base_models)
                    + len(user_stats)
                    + len(movie_stats),
                ] = movie_stats

        # Generate final predictions using the meta-model
        final_predictions = self.meta_model.predict(test_predictions)
        return final_predictions

    def evaluate(self) -> float:
        """
        Evaluate the model on the test set.
        """
        final_predictions = self.predict()
        mse = mean_squared_error(self.test_ratings, final_predictions)
        print(f"Mean Squared Error: {mse}")
        return mse
