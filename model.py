# model.py
"""
Defines the TempHypE-GNN architecture.
- Embeds node IDs
- hyperbolic GNN
- Dot-product edge scoring
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TempHypE_GNN(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, embed_dim)
        self.conv1 = GCNConv(embed_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True, normalize=True)

        # Optional: layer norm / dropout
        self.dropout = nn.Dropout(p=0.1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Edge scorer parameters (simple dot product; extend with relation embeddings if needed)
        # self.rel_embed = nn.Embedding(num_relations, embed_dim)  # if you add relations

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: torch_geometric.data.Data with fields:
                - x: [N, 1] node ids (long)
                - edge_index: [2, E]
        Returns:
            node embeddings X: [N, D]
        """
        x_ids = data.x.squeeze(-1).long()   # [N]
        x = self.embed(x_ids)               # [N, D]
        x = self.conv1(x, data.edge_index)
        x = self.ln1(F.relu(x))
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index)
        x = self.ln2(x)
        return x

    @staticmethod
    def score_edges(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Dot-product score for edges in edge_index.
        Args:
            x: [N, D] node embeddings
            edge_index: [2, E] pairs (src, dst)
        Returns:
            scores: [E]
        """
        h = x[edge_index[0]]  # [E, D]
        t = x[edge_index[1]]  # [E, D]
        scores = (h * t).sum(dim=-1)        # dot product
        return scores
