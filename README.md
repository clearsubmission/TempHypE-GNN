# TempHypE-GNN

Temporal Hyperbolic Graph Neural Network for Dynamic Link Prediction.  

## Overview

**TempHypE-GNN** is a graph neural network for **temporal link prediction** on **hierarchical knowledge graphs**. It unifies:
- **Hyperbolic embeddings** (Poincaré ball) to preserve hierarchical structure,
- **Hyperbolic GNN message passing** to propagate information along curved geometry, and
- **Neural ODE–based continuous-time dynamics** to model smooth temporal evolution between events.

This combination lets TempHypE-GNN capture both **who-is-above-whom** (hierarchy) and **how things change over time** (continuity), delivering strong performance on standard TKG benchmarks.

**Key features**
- Curvature-aware representation learning (hyperbolic Exp/Log maps and gyrovector ops).
- Continuous-time state updates via Neural ODEs (adaptive solvers supported).
- Plug-and-play scoring functions for temporal link prediction.
- Reproducible training/eval pipelines for **ICEWS14**, **ICEWS18**, and **GDELT**.
- Ablations to isolate the impact of hyperbolic space, GNN layers, and ODE dynamics.

## Installation
### Dependencies
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7 (if using GPU)

### Steps
```bash
git clone https://github.com/clearsubmission/TempHypE-GNN.git
cd TempHypE-GNN
pip install -r requirements.txt
