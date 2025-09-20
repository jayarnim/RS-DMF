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
        super(Module, self).__init__()
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
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        rep_user, rep_item = self.rl(user_idx, item_idx)
        logit = F.cosine_similarity(rep_user, rep_item, dim=1).squeeze(-1)
        return logit

    def rl(self, user_idx, item_idx):
        rep_user = self.user(user_idx, item_idx)
        rep_item = self.item(user_idx, item_idx)
        return rep_user, rep_item

    def user(self, user_idx, item_idx):
        # get user vector from interactions
        user_slice = self.interactions[user_idx, :-1].clone()
        
        # masking target items
        user_batch = torch.arange(user_idx.size(0))
        user_slice[user_batch, item_idx] = 0
        
        # projection
        proj_user = self.proj_u(user_slice.float())
        
        # representation learning
        rep_user = self.mlp_u(proj_user)

        return rep_user

    def item(self, user_idx, item_idx):
        # get item vector from interactions
        item_slice = self.interactions.T[item_idx, :-1].clone()
        
        # masking target users
        item_batch = torch.arange(item_idx.size(0))
        item_slice[item_batch, user_idx] = 0
        
        # projection
        proj_item = self.proj_i(item_slice.float())
        
        # representation learning
        rep_item = self.mlp_i(proj_item)

        return rep_item

    def _init_layers(self):
        self.proj_u = nn.Linear(
            in_features=self.n_items,
            out_features=self.hidden[0],
            bias=False,
        )
        self.proj_i = nn.Linear(
            in_features=self.n_users,
            out_features=self.hidden[0],
            bias=False,
        )
        self.mlp_u = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )
        self.mlp_i = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )
        self.logit_layer = nn.Linear(
            in_features=self.hidden[-1],
            out_features=1,
        )

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1