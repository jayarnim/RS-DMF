import torch
import torch.nn as nn
from components.functions import fc_block


class RepresentationLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        super().__init__()

        kwargs = dict(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        components = list(fc_block(**kwargs))
        mlp_user = nn.Sequential(*components)

        components = list(fc_block(**kwargs))
        mlp_item = nn.Sequential(*components)

        components = dict(
            user=mlp_user,
            item=mlp_item,
        )
        self.mlp = nn.ModuleDict(components)

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_rep = self.mlp["user"](user_emb)
        item_rep = self.mlp["item"](item_emb)
        return user_rep, item_rep