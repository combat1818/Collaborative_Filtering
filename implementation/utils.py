import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

MIN_RATING = 1
MAX_RATING = 5
NUM_USERS = 10000
NUM_MOVIES = 1000


class CustomDataset(Dataset):
    def __init__(self, data, response):
        self.data = data
        self.response = response

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.response[idx]


class MSELoss(nn.Module):
    def __init__(self, lambda_reg=1e-4):
        super(MSELoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, predictions, targets, user_embeddings, item_embeddings):
        loss = ((predictions - targets) ** 2).mean()
        reg_loss = self.lambda_reg * (user_embeddings.norm(2).pow(2) + item_embeddings.norm(2).pow(2))
        return loss + reg_loss


class RatingDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item, rating = self.interactions[idx]
        return user, item, rating
    
class RatingsDataset(Dataset):
    def __init__(self, T, Y):
        self.T = T
        self.Y = Y

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        user_idx = self.T[idx, 0]
        movie_idx = self.T[idx, 1]
        return (
            self.Y[user_idx, :].astype(np.float32),
            self.Y[:, movie_idx].T.astype(np.float32),
            self.Y[user_idx, movie_idx].astype(np.float32),
        )


def extract_users_items_ratings(data_pd: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users, movies = [
        np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1)
    ]
    predictions = np.array(data_pd.Prediction.values)
    return users, movies, predictions


def norm(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_mean = np.nanmean(A, axis=0)
    col_std = np.nanstd(A, axis=0)
    A_n = (A - col_mean) / col_std
    return A_n, col_mean, col_std


def denorm(A: np.ndarray, col_mean: np.ndarray, col_std: np.ndarray) -> np.ndarray:
    return (A * col_std) + col_mean


def impute(A: np.ndarray, impute_method: str = "zero") -> np.ndarray:

    if impute_method == "zero":
        A_i = np.nan_to_num(A, nan=0.0)
        return A_i
    else:
        A_i = np.copy(A)
        item_mean = np.nanmean(A_i, axis=0)
        na_idx = np.where(np.isnan(A_i))
        A_i[na_idx] = np.take(item_mean, na_idx[1])
        return A_i


def extract_matrix_predictions(reconstructed_matrix: np.ndarray, users: np.ndarray, movies: np.ndarray) -> np.ndarray:
    # assert(len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(users))
    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]
    return predictions


def get_score(prediction: np.ndarray, target: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(prediction, target))


def create_ratings_matrix(
    num_users: int,
    num_movies: int,
    users_array: np.ndarray,
    movies_array: np.ndarray,
    ratings_array: np.ndarray,
    impute_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    M = np.full((num_users, num_movies), impute_value)
    mask = np.full((num_users, num_movies), 0)
    for i, (user, movie) in enumerate(zip(users_array, movies_array)):
        M[user][movie] = ratings_array[i]
        mask[user][movie] = 1
    return M, mask


def get_dataloader(
    users_train: np.ndarray,
    movies_train: np.ndarray,
    ratings_train: np.ndarray,
    batch_size: int = 1024,
    num_users: int = NUM_USERS,
    num_movies: int = NUM_MOVIES,
) -> tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert available rating records to a num_users*num_movies matrix, return its PyTorch dataloader,
    data torch tensor, mask tensor, and user ID tensor.
    """
    # Initialize data and mask matrices with zeros
    data_matrix = np.zeros((num_users, num_movies))
    mask_matrix = np.zeros((num_users, num_movies))
    # Fill in the matrices based on the training data
    for user, movie, rating in zip(users_train, movies_train, ratings_train):
        data_matrix[user, movie] = rating
        mask_matrix[user, movie] = 1

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert numpy arrays to PyTorch tensors
    data_tensor = torch.tensor(data_matrix, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask_matrix, dtype=torch.float32, device=device)
    user_ids_tensor = torch.arange(num_users, device=device)
    # Create a DataLoader for the dataset
    dataset = TensorDataset(data_tensor, mask_tensor, user_ids_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, data_tensor, mask_tensor, user_ids_tensor


def get_data(
    file_path: str = "./data/data_train.csv", train_size: float = 0.9
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    # Split the dataset into train and test
    if train_size == 1:
        train_df = df
        test_df = df
    else:
        train_df, test_df = train_test_split(df, train_size=train_size, random_state=42)
    train_users, train_movies, train_ratings = extract_users_items_ratings(train_df)
    test_users, test_movies, test_ratings = extract_users_items_ratings(test_df)
    print(f"Number of training sumples: {len(train_users)}")
    if train_size != 1:
        print(f"Number of testing sumples: {len(test_users)}")
    return train_users, train_movies, train_ratings, test_users, test_movies, test_ratings


def get_all_dfs(
    train_df_file_path: str = "./data/data_train.csv", predict_df_file_path: str = "./data/sampleSubmission.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load data
    train_df = pd.read_csv(train_df_file_path)
    predict_df = pd.read_csv(predict_df_file_path)
    return train_df, predict_df


def generate_submission_from_model(
    trained_model: "BaseModel", file_path: str = "./data/sampleSubmission.csv", save_path: str = "./submissions/"
) -> None:
    predict_df = pd.read_csv(file_path)
    print("Generating predictions")
    users_to_predict, movies_to_predict, _ = extract_users_items_ratings(predict_df)
    predictions = trained_model.predict(users_to_predict, movies_to_predict)
    generate_submission(predictions, trained_model.model_name, file_path, save_path)


def generate_submission(
    predictions: np.ndarray,
    model_name: str,
    file_path: str = "./data/sampleSubmission.csv",
    save_path: str = "./submissions/",
) -> None:
    predict_df = pd.read_csv(file_path)
    users_to_predict, movies_to_predict, _ = extract_users_items_ratings(predict_df)

    print("Writing to file")
    with open(save_path + f"{model_name}_submission.csv", "w") as f:
        f.write("Id,Prediction\n")
        for user, movie, pred in zip(users_to_predict, movies_to_predict, predictions):
            f.write("r{}_c{},{}\n".format(user + 1, movie + 1, pred))

def calculate_stats(values: np.ndarray) -> list[float]:
    """
    Calculate statistics of the given values.
    """
    if len(values) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    mean_val = values.mean()
    std_val = values.std()
    median_val = np.median(values)
    mode_val = stats.mode(values)[0]
    percent_of_5 = len(values[values == 5]) / len(values)
    percent_of_4 = len(values[values == 4]) / len(values)
    percent_of_3 = len(values[values == 3]) / len(values)
    percent_of_2 = len(values[values == 2]) / len(values)
    percent_of_1 = len(values[values == 1]) / len(values)
    return [mean_val, std_val, median_val, mode_val, percent_of_5, percent_of_4, percent_of_3, percent_of_2, percent_of_1]