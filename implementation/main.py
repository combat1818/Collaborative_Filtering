import argparse

import pandas as pd
from blending import Blending
from model_als import ALSModel
from model_autoencoder import AutoencoderModel
from model_base import BaseModel
from model_dmf import DMFModel
from model_knn_means import KNNMeansModel
from model_ncf import NCFModel
from model_svd import IterativeSVDModel, RSVDModel, SVDModel, SVDppModel
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from stacking import Stacking
from utils import (generate_submission, generate_submission_from_model,
                   get_all_dfs, get_data)
from xgboost import XGBRegressor

# SVDpp 40 epochs, rank 3: RMSE score: 0.9966304419212764,  70 epochs rank 3: RMSE score: 0.9938960311887785


def train_evaluate_model(model: BaseModel) -> float:
    """
    Train and evaluate the model.
    """
    train_users, train_movies, train_ratings, test_users, test_movies, test_ratings = get_data()

    print("Training model: ")
    model.fit(train_users, train_movies, train_ratings, test_users, test_movies, test_ratings)
    # model.train(train_users, train_movies, train_ratings, test_users, test_movies, test_ratings)

    print("Evaluate the model: ")
    score = model.evaluate(test_users, test_movies, test_ratings)
    print(f"RMSE score: {score}")
    return score


def train_model_for_submission(
    model: BaseModel,
    data_file_path: str = "./data/data_train.csv",
    submission_file_path: str = "./data/sampleSubmission.csv",
    save_path: str = "./submissions/",
) -> None:
    """
    Train the model and generate submission.
    
    Args:
        model: The model to train.
        data_file_path: The path to the training data file.
        submission_file_path: The path to the submission file.
        save_path: The path to save the submission file.
    """
    train_users, train_movies, train_ratings, _, _, _ = get_data(data_file_path, train_size=1)

    print("Training model: ")
    model.fit(train_users, train_movies, train_ratings)

    generate_submission_from_model(model, submission_file_path, save_path)


def main(args: argparse.Namespace) -> None:
    """
    Main function to train and evaluate the model.
    """

    model = None
    meta_model = None
    base_models = []
    retrain_base_models = []
    train_df, predict_df = get_all_dfs()
    train_for_submission = True if args.submission else False
    meta_model_name = ""

    if args.model == "SVD":
        model = SVDModel()
    elif args.model == "ALS":
        model = ALSModel()
    elif args.model == "IterativeSVD":
        model = IterativeSVDModel()
    elif args.model == "RSVD":
        model = RSVDModel()
    elif args.model == "Autoencoder":
        model = AutoencoderModel()
    elif args.model == "NCF":
        model = NCFModel()
    elif args.model == "SVDpp":
        model = SVDppModel()
    elif args.model == "KNNMeans":
        model = KNNMeansModel()
    elif args.model == "DMF":
        model = DMFModel()

    if args.SVD:
        base_models.append(SVDModel())
        retrain_base_models.append(False)
    if args.ALS:
        base_models.append(ALSModel())
        retrain_base_models.append(False)
    if args.IterativeSVD:
        base_models.append(IterativeSVDModel())
        retrain_base_models.append(False)
    if args.RSVD:
        base_models.append(RSVDModel())
        retrain_base_models.append(False)
    if args.SVDpp:
        base_models.append(SVDppModel())
        retrain_base_models.append(False)
    if args.KNNMeans:
        base_models.append(KNNMeansModel())
        retrain_base_models.append(False)
    if args.Autoencoder:
        base_models.append(AutoencoderModel())
        retrain_base_models.append(False)
    if args.NCF:
        base_models.append(NCFModel())
        retrain_base_models.append(False)
    if args.DMF:
        base_models.append(DMFModel())
        retrain_base_models.append(False)

    if args.meta_model == "XGB":
        meta_model = XGBRegressor(n_estimators=100, random_state=42)
        meta_model_name = "XGB"
    elif args.meta_model == "GB":
        meta_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        meta_model_name = "GB"
    elif args.meta_model == "AdaBoost":
        meta_model = AdaBoostRegressor(n_estimators=100, random_state=42)
        meta_model_name = "AdaBoost"

    if args.stacking:
        print("Stacking")
        stacking = Stacking(
            train_df=train_df,
            predict_df=predict_df,
            train_for_submission=train_for_submission,
            meta_model=meta_model,
            base_models=base_models,
            retrain_base_models=retrain_base_models,
            meta_model_name=meta_model_name,
        )
        stacking.fit()

        if train_for_submission:
            predictions = stacking.predict()
            generate_submission(predictions=predictions, model_name=meta_model_name)
        else:
            stacking.evaluate()
    elif args.blending:
        print("Blending")
        blending = Blending(
            train_df=train_df,
            predict_df=predict_df,
            train_for_submission=train_for_submission,
            meta_model=meta_model,
            base_models=base_models,
            retrain_base_models=retrain_base_models,
            meta_model_name=meta_model_name,
        )
        blending.fit()

        if train_for_submission:
            predictions = blending.predict()
            generate_submission(predictions=predictions, model_name=meta_model_name)
        else:
            blending.evaluate()
    elif args.train:
        train_evaluate_model(model=model)
    elif args.submission:
        train_model_for_submission(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SVD", help="Model to select")
    parser.add_argument("--train", action="store_true", help="Train and evaluate the model")
    parser.add_argument("--submission", action="store_true", help="Train the model and create submission")

    parser.add_argument("--stacking", action="store_true", help="Stacking base models")
    parser.add_argument("--blending", action="store_true", help="Blending base models")

    parser.add_argument("--meta_model", type=str, default="XGB", help="Model to select")
    parser.add_argument("--SVD", action="store_true")
    parser.add_argument("--ALS", action="store_true")
    parser.add_argument("--IterativeSVD", action="store_true")
    parser.add_argument("--RSVD", action="store_true")
    parser.add_argument("--SVDpp", action="store_true")
    parser.add_argument("--KNNMeans", action="store_true")
    parser.add_argument("--Autoencoder", action="store_true")
    parser.add_argument("--NCF", action="store_true")
    parser.add_argument("--DMF", action="store_true")

    args = parser.parse_args()
    main(args)

    # Train a single model: python "main.py file path" --model "SVDpp" --train
    # or for training and evaluation: python "main.py file path" --model "SVDpp" --submission
    # train multiple models for stacking/blending: python "main.py file path" --stacking --meta_model "XGB" --SVD --ALS --Autoencoder --submisson
    # or for training and evaluation: python "main.py file path" --stacking --meta_model "XGB" --SVD --ALS --Autoencoder --train


# SVD: local test rmse: 1.0098990404931265, public test (kaggle) rmse: 1.00511, svd_rank = 8
# ALS: local test rmse: 0.9895262482280919, public test (kaggle) rmse: 0.98724, rank_svd = 6, rank_als = 3
# IterativeSVD: local test rmse: 0.9860179221983582, public test (kaggle) rmse: 0.98316 , shrinkage = 32
# RSVD: local test rmse: 0.99014187589719, public test (kaggle) rmse: 0.98897, rank = 12, lambda1 = 0.07, lambda2 = 0.05, epochs = 15, lr = 0.025
# Autoencoder: local test rmse: 0.9864811745816908, public test (kaggle) rmse: 0.98201, 200 epochs, describe the architecture
# SVDpp: local test rmse: 0.9956187638453866, public test (kaggle) rmse: 0.99412, rank 9, 70 epochs
# KNNBaseline: local test rmse: 0.9966396909757901, public test(kaggle) rmse: 0.99415, k_neighbors = 3000, it is KNNBaseline now, check in the surprise library
# NCF: local test rmse: 1.0155184695620993, public test (kaggle) rmse: 1.01314


# Blending:
# XGB: --SVD --ALS, --IterativeSVD --RSVD --SVDpp --KNNMeans: local test rmse: 0.98243329316131, public test (kaggle) rmse: 0.98252
# GB: --SVD --ALS, --IterativeSVD --RSVD --SVDpp --KNNMeans: local test rmse: 0.9693242678982958, public test (kaggle) rmse: 0.97717
# AdaBoost: --SVD --ALS, --IterativeSVD --RSVD --SVDpp: local test rmse: 1.019884057026279, public test (kaggle) rmse: 1.00482

# Stacking
# XGB: --SVD --ALS, --IterativeSVD --RSVD --SVDpp --KNNMeans: local test rmse: 0.9627373493077763, public test (kaggle) rmse: 0.97650
# GB: --SVD --ALS, --IterativeSVD --RSVD --SVDpp --KNNMeans: local test rmse: 0.9599510222212979, public test (kaggle) rmse: 0.97544
# AdaBoost: --SVD --ALS, --IterativeSVD --RSVD --SVDpp: local test rmse: 1.013252945371907, public test (kaggle) rmse: 1.00483
