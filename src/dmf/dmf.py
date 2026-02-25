import torch
from components.interactions import Interactions
from components.base import BaseModel
from .layers.embedding import build as build_embedding_layer
from .layers.representation import RepresentationLayer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class DeepMatrixFactorization(BaseModel):
    def __init__(
        self,
        interactions: Interactions, 
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        """
        Deep matrix factorization models for recommender systems (Xue et al., 2017)
        -----
        Implements the base structure of Deep Matrix Factorization (DMF),
        MF & history embedding based latent factor model.

        Args:
            interactions (Interactions): 
                user-item interaction matrix, masked evaluation datasets.
                (shape: [U+2, I+2])
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            projection_dim (int):
                dimensionality of user and item projection vectors.
            hidden_dim (list):
                layer dimensions for the MLP-based matching function. 
                (e.g., [128, 64, 32])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__(locals())

        self.pred_dim = hidden_dim[-1]

        # USER-ITEM INTERACTION MATRIX VIEWER ==========
        self.interactions = interactions

        # HISTORY EMBEDDING ==========
        self.embedding = build_embedding_layer(
            name="history",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # REPRESENTATION LEARNING ==========
        self.representation = RepresentationLayer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # BILINEAR MATCHING FUNCTION ==========
        self.matching = build_matching_layer(
            name="mf",
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # SEARCH USER-ITEM MAT. ==========
        user_vec, item_vec = self.interactions(user_idx, item_idx)
        # HISTORY EMBEDDING ==========
        user_emb, item_emb = self.embedding(user_vec, item_vec)
        # REPRESENTATION LEARNING ==========
        user_rep, item_rep = self.representation(user_emb, item_emb)
        # BILINEAR MATCHING FUNCTION ==========
        X_pred = self.matching(user_rep, item_rep)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        # INTERACTION MODELING ==========
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION ==========
        logit = self.prediction(X_pred)
        return logit