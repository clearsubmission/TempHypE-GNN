# train.py
"""
Minimal training loop for link prediction with negative sampling
on temporal snapshots. Uses BCEWithLogitsLoss over positive/negative edges.
"""
import os
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from .data_processing import TempHypEDataset
from .model import TempHypE_GNN
from .utils import set_seed, plot_loss

def sample_negative_edges(num_nodes: int, pos_edge_index: torch.Tensor, num_neg: int) -> torch.Tensor:
    """Uniformly sample negative (u, v') edges (tail corruption)."""
    E = pos_edge_index.size(1)
    i = torch.randint(0, E, (num_neg,), device=pos_edge_index.device)
    src = pos_edge_index[0, i]
    dst = torch.randint(0, num_nodes, (num_neg,), device=pos_edge_index.device)
    return torch.stack([src, dst], dim=0)

def train(
    csv_path: str = "data/demo_triples.csv",
    epochs: int = 50,
    lr: float = 1e-3,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    batch_size: int = 1,
    neg_ratio: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    ckpt_path: str = "models/checkpoint.pt",
):
    set_seed(seed)
    os.makedirs("models", exist_ok=True)

    dataset = TempHypEDataset(csv_path, sep=",", has_header=True, time_unit="day")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_nodes = len(dataset.ent2id)
    model = TempHypE_GNN(num_nodes=num_nodes, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    loss_hist = []
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for data in loader:
            data = data.to(device)
            x = model(data)                             # [N, D]
            pos = data.edge_index                       # [2, E+E_rev] if reverse edges added
            num_neg = pos.size(1) * neg_ratio
            neg = sample_negative_edges(data.num_nodes, pos, num_neg)

            pos_scores = model.score_edges(x, pos)
            neg_scores = model.score_edges(x, neg)

            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

            opt.zero_grad(set_to_none=True)
            loss = criterion(scores, labels)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        loss_hist.append(epoch_loss)
        print(f"[Epoch {epoch:03d}] loss = {epoch_loss:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    plot_loss(loss_hist, out_path="results/loss_plot.png")
    print(f"Saved checkpoint to {ckpt_path} and loss curve to results/loss_plot.png")

if __name__ == "__main__":
    # For a quick demo, you can generate a small CSV with data_processing.py main
    train()
