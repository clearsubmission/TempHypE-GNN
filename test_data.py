# tests/test_data.py
import os
from pathlib import Path
import torch
from src.data_processing import TempHypEDataset

def _write_demo(tmp_csv="data/triples.csv"):
    p = Path(tmp_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("head,rel,tail,time\n"
                 "a,r1,b,2020-01-01\n"
                 "b,r2,c,2020-01-01\n"
                 "c,r1,d,2020-01-02\n"
                 "d,r2,a,2020-01-03\n", encoding="utf-8")
    return str(p)

def test_data_loading():
    csv_path = _write()
    ds = TempHypEDataset(csv_path, sep=",", has_header=True)
    assert len(ds) >= 1
    g = ds[0]
    assert hasattr(g, "edge_index") and g.edge_index.ndim == 2
    assert hasattr(g, "x") and g.x.ndim == 2
    assert hasattr(g, "t")
