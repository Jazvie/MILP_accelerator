from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

class BipartiteGNN(nn.Module):
    def __init__(self, var_dim:int=19, con_dim:int=5, emb_dim:int=64):
        super().__init__()
        self.var_dim = var_dim
        self.con_dim = con_dim
        self.emb_dim = emb_dim
        self.var_embedding = nn.Linear(var_dim, emb_dim)
        self.con_embedding = nn.Linear(con_dim, emb_dim)
        self.edge_weight = nn.Linear(1, emb_dim, bias=False)
        self.con_update = nn.Linear(2 * emb_dim, emb_dim)
        self.var_update = nn.Linear(2 * emb_dim, emb_dim)
        self.policy = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(
        self,
        var_features: torch.Tensor,
        con_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_vars = var_features.size(0)
        n_cons = con_features.size(0)

        var_emb = F.relu(self.var_embedding(var_features))
        con_emb = F.relu(self.con_embedding(con_features))

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_emb = self.edge_weight(edge_attr)
        con_idx = edge_index[0]
        var_idx = edge_index[1]
        con_agg = scatter(var_emb[var_idx] * edge_emb, con_idx, dim=0, dim_size=n_cons, reduce="add")
        con_emb = F.relu(self.con_update(torch.cat([con_emb, con_agg], dim=-1)))
        var_agg = scatter(con_emb[con_idx] * edge_emb, var_idx, dim=0, dim_size=n_vars, reduce="add")
        var_emb = F.relu(self.var_update(torch.cat([var_emb, var_agg], dim=-1)))
        scores = self.policy(var_emb).squeeze(-1)
        if action_mask is not None:
            scores = scores.masked_fill(~action_mask, float("-inf"))
        return scores
