import os
from typing import Any

import numpy as np
import pandas as pd
from model_base import BaseModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import extract_users_items_ratings


class Blending:
    def __init__(
        self,
        train_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        train_for_submission: bool,
        meta_model: Any,
        base_models: list[BaseModel],
        retrain_base_models: list[bool],
        meta_model_name: str,
    ) -> None:
        """
        Initializes the blending model.

        Args:
            train_df: The training data.
            predict_df: The test data.
            train_for_submission: Whether to train the model for submission or not.
            meta_model: The meta-model to use for blending.
            base_models: The base models to use for blending.
            retrain_base_models: Whether to retrain the base models or not, should be a list of the same length as base_models.
            meta_model_name: The name of the meta-model.
        """
        self.meta_model = meta_model
        self.base_models = base_models
        self.retrain_base_models = retrain_base_models
        self.training_type = "submission" if train_for_submission else "train"
        self.meta_model_name = meta_model_name
        test_size = 0.2

        if train_for_submission:
            train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=42)

            self.train_users, self.train_movies, self.train_ratings = extract_users_items_ratings(train_df)
            self.val_users, self.val_movies, self.val_ratings = extract_users_items_ratings(val_df)

            self.test_users, self.test_movies, self.test_ratings = extract_users_items_ratings(predict_df)
        else:
            # Split training data into training and validation sets for blending
            train_df_full, test_df = train_test_split(train_df, test_size=test_size, random_state=42)
            train_df, val_df = train_test_split(train_df_full, test_size=test_size, random_state=42)

            self.train_users, self.train_movies, self.train_ratings = extract_users_items_ratings(train_df)
            self.test_users, self.test_movies, self.test_ratings = extract_users_items_ratings(test_df)
            self.val_users, self.val_movies, self.val_ratings = extract_users_items_ratings(val_df)

        print(
            f"Number of training samples: {len(self.train_users)}, number of test samples: {len(self.test_users)}, number of val samples: {len(self.val_users)}."
        )

    def fit(self, val_pred_path: str = "./predictions/") -> None:
        """
        Train base models on the training set and the meta-model on the validation set predictions.
        """
        val_predictions = []
        for i, model in enumerate(self.base_models):
            model_name = model.model_name
            val_pred_file = (
                val_pred_path + f"{self.meta_model_name}_{self.training_type}_blending_val_pred_{model_name}.npy"
            )

            if self.retrain_base_models[i] or not os.path.exists(val_pred_file):
                print(f"Training model: {model_name}")
                model.fit(self.train_users, self.train_movies, self.train_ratings)
                val_prediction = model.predict(self.val_users, self.val_movies)
                np.save(val_pred_file, val_prediction)
            else:
                print(f"Loading predictions for model: {model_name} from file")
                val_prediction = np.load(val_pred_file)

            val_predictions.append(val_prediction)

        print(f"Finished training base models")

        # Create a new dataset for the meta-model
        meta_X_train = np.column_stack(val_predictions)
        meta_y_train = self.val_ratings

        # Train the meta-model on the validation set predictions:
        self.meta_model.fit(meta_X_train, meta_y_train)

    def predict(self, test_pred_path: str = "./predictions/") -> np.ndarray:
        """
        Train the base models on the full training set and make predictions with the previously trained blending model.
        """
        # Make predictions with base models on the test data:
        test_predictions = []
        for i, model in enumerate(self.base_models):
            model_name = model.model_name
            test_pred_file = (
                test_pred_path + f"{self.meta_model_name}_{self.training_type}_blending_test_pred_{model_name}.npy"
            )

            if self.retrain_base_models[i] or not os.path.exists(test_pred_file):
                model.fit(self.train_users, self.train_movies, self.train_ratings)
                test_prediction = model.predict(self.test_users, self.test_movies)
                np.save(test_pred_file, test_prediction)
            else:
                print(f"Loading predictions for model: {model_name} from file")
                test_prediction = np.load(test_pred_file)

            test_predictions.append(test_prediction)

        # Generate meta-features for the test set:
        meta_X_test = np.column_stack(test_predictions)

        # Make final predictions with the meta-model:
        final_predictions = self.meta_model.predict(meta_X_test)
        return final_predictions

    def evaluate(self) -> float:
        """
        Evaluate the blending model on the test set.
        """
        final_predictions = self.predict()
        mse = mean_squared_error(self.test_ratings, final_predictions)
        print(f"Mean Squared Error: {mse}")
        return mse
