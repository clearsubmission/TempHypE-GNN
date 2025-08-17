# tests/test_model.py
import torch
from src.model import TempHypE_GNN

def test_model_forward():
    num_nodes = 10
    model = TempHypE_GNN(num_nodes=num_nodes, embed_dim=16, hidden_dim=16)
    # toy graph
    edge_index = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long)
    x_ids = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1)
    data = type("Data", (), {"x": x_ids, "edge_index": edge_index})
    out = model(data)
    assert out.shape == (num_nodes, 16)
    # scoring
    scores = model.score_edges(out, edge_index)
    assert scores.shape == (edge_index.size(1),)
