import torch
import torch.nn as nn
import torch.nn.functional as F


class DMF(nn.Module):
    #hyperparams : num_factors1, num_factors2, depth of linear layers
    def __init__(self, num_users: int, num_movies: int, num_factors1: int=128, num_factors2: int=64) -> None:
        """
        Initializes the DMF model.
        
        Args:
            num_users: The number of users.
            num_movies: The number of movies.
            num_factors1: The number of factors for the first linear layer.
            num_factors2: The number of factors for the second linear layer.
        """
        super(DMF, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies

        self.user_embedding = nn.Sequential(
            nn.Linear(num_movies, num_factors1),
            nn.ReLU(),
            nn.Linear(num_factors1, num_factors2),
            nn.ReLU()
        )

        self.movie_embedding = nn.Sequential(
            nn.Linear(num_users, num_factors1),
            nn.ReLU(),
            nn.Linear(num_factors1, num_factors2),
            nn.ReLU()
        )

    def forward(self, user_vect: torch.Tensor, movie_vect: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            user_vect: The user vector.
            movie_vect: The movie vector.
        """
        user_embedded = self.user_embedding(user_vect)
        movie_embedded = self.movie_embedding(movie_vect)
        
        cosine_sim = F.cosine_similarity(user_embedded, movie_embedded)
        return cosine_sim
