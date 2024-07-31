import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class NCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_movies: int,
        users_latent_dim: int = 64,
        movies_latent_dim: int = 64,
        hidden_dim: int = 16,
        num_layers: int = 3,
        mlp_network_layers: int = 3,
        drop_prob: float = 0.4,
        cleora_embeddings_path: str = None,
    ) -> None:
        """
        Initializes the NCF model.
        
        Args:
            n_users: The number of users.
            n_movies: The number of movies.
            users_latent_dim: The latent dimension of the user embeddings.
            movies_latent_dim: The latent dimension of the movie embeddings.
            hidden_dim: The hidden dimension of the MLP layers.
            num_layers: The number of layers in the MLP.
            mlp_network_layers: The number of layers in the MLP network.
            add_cleora: Whether to add Cleora embeddings or not.
        """
        super().__init__()
        self.users_latent_dim = users_latent_dim
        self.movies_latent_dim = movies_latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mlp_network_layers = mlp_network_layers
        self.drop_prob = drop_prob
        self.add_cleora = cleora_embeddings_path is not None

        self.user_emb = nn.Embedding(n_users, self.users_latent_dim)
        self.movie_emb = nn.Embedding(n_movies, self.movies_latent_dim)
        # cleora embeddings
        if self.add_cleora:
            cleora_output = np.load(cleora_embeddings_path)
            self.cleora_movies_embeds = cleora_output['vectors']
            self.cleora_latent_dim = self.cleora_movies_embeds.shape[1]

        self.mlp_user = self.generate_layers(self.num_layers, self.users_latent_dim)
        self.user_norm = nn.BatchNorm1d(self.users_latent_dim)
        self.mlp_movie = self.generate_layers(self.num_layers, self.movies_latent_dim)
        self.movie_norm = nn.BatchNorm1d(self.movies_latent_dim)
        if self.add_cleora:
            self.mlp_movie_cleora = self.generate_layers(self.num_layers, self.cleora_latent_dim)
            self.cleora_norm = nn.BatchNorm1d(self.cleora_latent_dim)

        self.mf_user = self.generate_layers(self.num_layers, self.users_latent_dim)
        self.user_norm_mf = nn.BatchNorm1d(self.users_latent_dim)
        self.mf_movie = self.generate_layers(self.num_layers, self.movies_latent_dim)
        self.movie_norm_mf = nn.BatchNorm1d(self.movies_latent_dim)
        if self.add_cleora:
            self.mf_movie_cleora = self.generate_layers(self.num_layers, self.cleora_latent_dim)
            self.cleora_norm_mf = nn.BatchNorm1d(self.cleora_latent_dim)
        
        add_dim = 0 if not self.add_cleora else self.cleora_latent_dim

        self.mlp_network = self.generate_layers(self.mlp_network_layers, self.users_latent_dim + self.movies_latent_dim + add_dim)

        self.fc1 = nn.Linear(2*self.users_latent_dim + 2*add_dim + self.movies_latent_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            users: The user indices.
            movies: The movie indices.
        """
        users_emb = self.user_emb(users)
        movies_emb = self.movie_emb(movies)
        if self.add_cleora:
            cleora_emb = torch.tensor(self.cleora_movies_embeds[movies])

        users_mlp_tensor = self.user_norm(self.mlp_user(users_emb) + users_emb)
        movies_mlp_tensor = self.movie_norm(self.mlp_movie(movies_emb) + movies_emb)
        if self.add_cleora:
            cleora_mlp_tensor = self.cleora_norm(self.mlp_movie_cleora(cleora_emb) + cleora_emb)

        users_mf_tensor = self.user_norm_mf(self.mf_user(users_emb))
        movies_mf_tensor = self.movie_norm_mf(self.mf_movie(movies_emb))
        if self.add_cleora:
            cleora_mf_tensor = self.cleora_norm_mf(self.mf_movie_cleora(cleora_emb))

        elementwise_product = torch.mul(users_mf_tensor, movies_mf_tensor)
        if self.add_cleora:
            elementwise_user_cleora = torch.mul(users_mf_tensor, cleora_mf_tensor)
        if self.add_cleora:
            concatenation = torch.cat([users_mlp_tensor, movies_mlp_tensor, cleora_mlp_tensor], dim=1)
        else:
            concatenation = torch.cat([users_mlp_tensor, movies_mlp_tensor], dim=1)
        concatenation = self.mlp_network(concatenation)

        if self.add_cleora:
            final_concatenation = torch.cat([elementwise_product, elementwise_user_cleora, concatenation], dim=1)
        else:
            final_concatenation = torch.cat([elementwise_product, concatenation], dim=1)
        res = self.fc1(final_concatenation)
        res = F.relu(res)
        res = self.fc2(res)

        return res

    def generate_layers(self, num_layers: int, dim: int) -> nn.Sequential:
        """
        Generates a sequence of linear layers with ReLU activation and dropout.
        """
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            if i != num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.drop_prob))

        return nn.Sequential(*layers)
