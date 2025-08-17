# utils.py
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_loss(loss_history, out_path="results/loss_plot.png"):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
