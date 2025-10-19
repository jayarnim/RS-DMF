import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        hidden: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        """
        Deep matrix factorization models for recommender systems (Xue et al., 2017)
        -----
        Implements the base structure of Deep Matrix Factorization (DMF),
        MF & history embedding based latent factor model.

        Args:
            n_users (int):
                total number of users in the dataset, U.
            n_items (int):
                total number of items in the dataset, I.
            n_factors (int):
                dimensionality of user and item latent representation vectors, K.
            hidden (list):
                layer dimensions for the MLP-based matching function. 
                (e.g., [64, 32, 16, 8])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
            interaction (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets.
                (shape: [U+1, I+1])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        return self.score(user_idx, item_idx)

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        logit = self.score(user_idx, item_idx)
        prob = torch.sigmoid(logit)
        return prob

    def score(self, user_idx, item_idx):
        user_embed_slice, item_embed_slice = self.rl(user_idx, item_idx)
        logit = F.cosine_similarity(user_embed_slice, item_embed_slice, dim=1).squeeze(-1)
        return logit

    def rl(self, user_idx, item_idx):
        user_embed_slice = self.user_hist_embed_generator(user_idx, item_idx)
        item_embed_slice = self.item_hist_embed_generator(user_idx, item_idx)
        return user_embed_slice, item_embed_slice

    def user_hist_embed_generator(self, user_idx, item_idx):
        # get user vector from interactions
        user_interaction_slice = self.interactions[user_idx, :-1].clone()
        
        # masking target items
        user_idx_batch = torch.arange(user_idx.size(0))
        user_interaction_slice[user_idx_batch, item_idx] = 0
        
        # projection
        user_proj_slice = self.proj_u(user_interaction_slice.float())
        
        # representation learning
        user_rep_slice = self.rep_u(user_proj_slice)

        return user_rep_slice

    def item_hist_embed_generator(self, user_idx, item_idx):
        # get item vector from interactions
        item_interaction_slice = self.interactions.T[item_idx, :-1].clone()
        
        # masking target users
        item_idx_batch = torch.arange(item_idx.size(0))
        item_interaction_slice[item_idx_batch, user_idx] = 0
        
        # projection
        item_proj_slice = self.proj_i(item_interaction_slice.float())
        
        # representation learning
        item_rep_slice = self.rep_i(item_proj_slice)

        return item_rep_slice

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.n_items,
            out_features=self.hidden[0],
            bias=False,
        )
        self.proj_u = nn.Linear(**kwargs)

        kwargs = dict(
            in_features=self.n_users,
            out_features=self.hidden[0],
            bias=False,
        )
        self.proj_i = nn.Linear(**kwargs)

        components = list(self._yield_linear_block(self.hidden))
        self.rep_u = nn.Sequential(*components)

        components = list(self._yield_linear_block(self.hidden))
        self.rep_i = nn.Sequential(*components)

    def _yield_linear_block(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Sequential(
                nn.Linear(hidden[idx-1], hidden[idx]),
                nn.LayerNorm(hidden[idx]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            idx += 1