import numpy as np
import torch
import torch.optim as optim
from model_base import BaseModel
from pytorch_models.NCF import NCF
from torch import nn
from utils import (MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS,
                   CustomDataset, DataLoader, get_score)


class NCFModel(BaseModel):
    def __init__(self, epochs: int=12, batch_size: int=1024, cleora_embeddings_path: str = None) -> None:
        """
        Initializes the NCF model.
        
        Args:
            epochs: The number of epochs to train the model.
            batch_size: The batch size to use during training.
        """
        super().__init__(NUM_USERS, NUM_MOVIES, MIN_RATING, MAX_RATING)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.add_cleora = cleora_embeddings_path is not None
        self.model = NCF(self.num_users, self.num_movies, cleora_embeddings_path=cleora_embeddings_path).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = "NCF"

    def fit(
        self,
        users_train: np.ndarray,
        movies_train: np.ndarray,
        ratings_train: np.ndarray,
        users_val: np.ndarray = None,
        movies_val: np.ndarray = None,
        ratings_val: np.ndarray = None,
    ) -> None:
        """
        Fits the model to the given data.

        Args:
            users_train: The user ids of the training data.
            movies_train: The movie ids of the training data.
            ratings_train: The ratings of the training data.
            users_val: The user ids of the validation data used for validation during training.
            movies_val: The movie ids of the validation data used for validation during training.
            ratings_val: The ratings of the validation data used for validation during training.
        """
        # reset model
        self.model = NCF(self.num_users, self.num_movies, add_cleora=self.add_cleora).to(self.device)
        train_data = np.column_stack((users_train, movies_train))
        train_labels = ratings_train.reshape(-1, 1)
        train_dataset = CustomDataset(train_data, train_labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        if not any(data is None for data in [users_val, movies_val, ratings_val]):
            test_data = np.column_stack((users_val, movies_val))
            test_labels = ratings_val.reshape(-1, 1)
            test_dataset = CustomDataset(test_data, test_labels)
            test_loader = DataLoader(dataset=test_dataset, batch_size=len(users_val), shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)
        criterion = nn.MSELoss()
        print("Training initialized.")
        # print(train_data)
        train_loss = []
        eval_loss = []
        for epoch in range(self.epochs):
            accumulated_loss = []
            print(f"{self.model_name}: epoch {epoch+1}/{self.epochs}")
            self.model.train()
            for _, (data_batch, label_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                data_batch = data_batch.to(self.device)
                label_batch = label_batch.to(self.device).float()
                predictions = self.model(data_batch[:, 0], data_batch[:, 1])
                loss = torch.sqrt(criterion(predictions, label_batch))
                loss.backward()
                optimizer.step()
                # print loss value
                accumulated_loss.append(loss.item())
            train_loss_val = sum(accumulated_loss) / len(accumulated_loss)
            train_loss.append(train_loss_val)
            print(f"Train loss: {train_loss_val}")

            if not any(data is None for data in [users_val, movies_val, ratings_val]):
                self.model.eval()
                with torch.no_grad():
                    score = self.evaluate(users_val, movies_val, ratings_val)
                    print(f"RMSE score: {score}")
                    eval_loss.append(score)
                    # for _, (data_batch, label_batch) in enumerate(test_loader):
                    #     data_batch = data_batch.to(self.device)
                    #     label_batch = label_batch.to(self.device).float()
                    #     predictions = self.model(data_batch[:, 0], data_batch[:, 1])
                    #     predictions = predictions.squeeze(1).cpu().numpy()
                    #     targets = label_batch.squeeze(1).cpu().numpy()
                    #     rmse = get_score(predictions, targets)
                    #     print(f"Reconstruction RMSE: {rmse}")
        # save losses to csv
        np.savetxt(f"./{self.model_name}2_cleora_train_loss.csv", train_loss, delimiter=",")
        if len(eval_loss) > 0:
            np.savetxt(f"./{self.model_name}2_cleora_eval_loss.csv", eval_loss, delimiter=",")
            

    
    def predict(self, users_test: np.ndarray, movies_test: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings for the given data.

        Args:
            users_test: The user ids of the test data.
            movies_test: The movie ids of the test data
        """
        test_data = np.column_stack((users_test, movies_test))
        test_data_tensor = torch.tensor(test_data)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_data_tensor[:, 0], test_data_tensor[:, 1])
            predictions = predictions.squeeze(1).cpu().numpy()
            predictions = np.clip(predictions, self.min_rating, self.max_rating)
        return predictions
    
    def reconstruct(self) -> np.ndarray:
        """
        Returns the reconstructed matrix.
        """
        reconstructed_matrix = np.zeros((self.num_users, self.num_movies))
        for user in range(self.num_users):
            for movie in range(self.num_movies):
                test_data = np.column_stack((user, movie))
                test_data_tensor = torch.tensor(test_data)
                prediction = self.predict(test_data_tensor)
                reconstructed_matrix[user, movie] = prediction[0]
        self.reconstructed_matrix = reconstructed_matrix
        return self.reconstructed_matrix
