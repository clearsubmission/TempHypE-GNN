# evaluate.py
"""
Evaluation: ranking metrics (MRR, Hits@K).
For each positive edge, sample K negatives; compute the rank of the positive among them.
This is a lightweight approximation to full ranking and works for quick checks.
"""
from typing import Dict, Tuple
import torch
from torch_geometric.loader import DataLoader

from .data_processing import TempHypEDataset
from .model import TempHypE_GNN

@torch.no_grad()
def sampled_ranking_metrics(model, dataset, K: int = 50, device="cpu") -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval().to(device)

    mrr_sum, hits1, hits3, hits10, count = 0.0, 0, 0, 0, 0
    for data in loader:
        data = data.to(device)
        x = model(data)
        pos = data.edge_index  # consider directed edges
        num_nodes = data.num_nodes

        # For each positive edge, compute rank among K negatives (tail corruption)
        for i in range(pos.size(1)):
            src = pos[0, i]
            dst = pos[1, i]
            pos_score = model.score_edges(x, pos[:, i:i+1])  # [1]

            # sample negatives different from dst
            neg_dst = torch.randint(0, num_nodes, (K,), device=device)
            neg_edges = torch.stack([src.repeat(K), neg_dst], dim=0)  # [2, K]
            neg_scores = model.score_edges(x, neg_edges)              # [K]

            # Rank: 1 + number of negatives with higher score
            greater = (neg_scores > pos_score).sum().item()
            rank = 1 + greater
            mrr_sum += 1.0 / rank
            hits1 += int(rank <= 1)
            hits3 += int(rank <= 3)
            hits10 += int(rank <= 10)
            count += 1

    return {
        "MRR": mrr_sum / max(1, count),
        "Hits@1": hits1 / max(1, count),
        "Hits@3": hits3 / max(1, count),
        "Hits@10": hits10 / max(1, count),
    }

def evaluate(csv_path="data/triples.csv", ckpt_path="models/checkpoint.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = TempHypEDataset(csv_path, sep=",", has_header=True, time_unit="day")

    model = TempHypE_GNN(num_nodes=len(ds.ent2id))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    metrics = ranking_metrics(model, ds, K=50, device=device)
    print({k: round(v, 4) for k, v in metrics.items()})

if __name__ == "__main__":
    evaluate()
