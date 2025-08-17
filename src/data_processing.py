# src/data_processing.py
"""
Utilities and a PyTorch Geometric Dataset for temporal KG triples.

- Reads (head, rel, tail, time) from CSV/TSV
- Maps entities/relations to integer IDs
- Buckets triples by time (day/month/year or numeric timestamps)
- Builds PyG `Data` timestamp per time bucket
- Chronological train/val/test split
- One-call helper: `load_dataset_and_split(...)`

quick demo:
    python -m src.data_processing
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import Data, Dataset


# ---------------------------------------------------------------------
# IO & preprocessing
# ---------------------------------------------------------------------
def read_triples_csv(
    file_path: str,
    sep: str = ",",
    has_header: bool = True,
    src_col: str = "head",
    rel_col: str = "rel",
    dst_col: str = "tail",
    time_col: str = "time",
) -> List[Dict[str, str]]:
    """
    Read temporal triples from a CSV/TSV file.

    Parameters
    ----------
    file_path : str
        Path to CSV/TSV file.
    sep : str
        Column separator ("," or "\\t").
    has_header : bool
        Whether the file has a header row.
    src_col, rel_col, dst_col, time_col : str
        Column names if `has_header=True`. Ignored otherwise.

    Returns
    -------
    list of dict(head, rel, tail, time)
    """
    rows: List[Dict[str, str]] = []
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=sep)
            for r in reader:
                rows.append(
                    {
                        "head": str(r[src_col]),
                        "rel": str(r[rel_col]),
                        "tail": str(r[dst_col]),
                        "time": str(r[time_col]),
                    }
                )
        else:
            reader = csv.reader(f, delimiter=sep)
            for r in reader:
                # Expect order: head, rel, tail, time
                rows.append({"head": r[0], "rel": r[1], "tail": r[2], "time": r[3]})
    return rows


def build_id_mappings(
    triples: Iterable[Dict[str, str]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create entity/relation string → integer ID maps."""
    entities, relations = set(), set()
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
        relations.add(t["rel"])
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def quantize_time(value: str, unit: str = "day") -> int:
    """
    Map a timestamp string to an integer bucket.
    - Numeric strings are cast directly.
    - ISO-like 'YYYY-MM-DD' strings are mapped by unit (day/month/year).
    - Otherwise, a stable hash bucket is used.
    """
    # numeric?
    try:
        return int(float(value))
    except ValueError:
        pass

    parts = value.strip().split("-")
    if len(parts) >= 3:
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if unit == "day":
            return y * 10000 + m * 100 + d
        if unit == "month":
            return y * 100 + m
        if unit == "year":
            return y

    # fallback
    return abs(hash(value)) % (10**9)


def group_by_time(
    triples: Iterable[Dict[str, str]], unit: str = "day"
) -> Dict[int, List[Dict[str, str]]]:
    """Group triples into buckets keyed by quantized time."""
    buckets: Dict[int, List[Dict[str, str]]] = {}
    for t in triples:
        tb = quantize_time(t["time"], unit=unit)
        buckets.setdefault(tb, []).append(t)
    # sort by time bucket for chronological order
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def to_pyg_timestamp(
    triples: Iterable[Dict[str, str]],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
    unit: str = "day",
    add_reverse_edges: bool = True,
    include_edge_type: bool = True,
) -> List[Data]:
    """
    Convert triples to a list of PyG Data.

    Each timestamp Data contains:
      - edge_index: LongTensor [2, E]
      - edge_type:  LongTensor [E]  (if include_edge_type)
      - x:          LongTensor [N, 1] node IDs (use with nn.Embedding)
      - t:          LongTensor scalar time bucket
      - num_nodes:  int

    Reverse edges are optional; when added, their relation IDs are offset by |R|.
    """
    by_time = group_by_time(triples, unit=unit)
    timestamp: List[Data] = []
    num_nodes = len(ent2id)
    R = len(rel2id)

    for tb, rows in by_time.items():
        srcs, dsts, types = [], [], []

        for r in rows:
            s = ent2id[r["head"]]
            o = ent2id[r["tail"]]
            p = rel2id[r["rel"]]

            srcs.append(s)
            dsts.append(o)
            if include_edge_type:
                types.append(p)

            if add_reverse_edges:
                srcs.append(o)
                dsts.append(s)
                if include_edge_type:
                    types.append(p + R)

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        data = Data(edge_index=edge_index)
        data.num_nodes = num_nodes
        data.x = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1)  # node IDs
        data.t = torch.tensor(tb, dtype=torch.long)
        if include_edge_type:
            data.edge_type = torch.tensor(types, dtype=torch.long)

        snaps.append(data)

    return snaps


def chronological_split(
    timestamp: List[Data], ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Chronologically split timestamp into train/val/test by ratios."""
    assert math.isclose(sum(ratios), 1.0, rel_tol=1e-6), "ratios must sum to 1"
    n = len(timestamp)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    n_test = max(1, n - n_train - n_val)
    train = timestamp[:n_train]
    val = timestamp[n_train : n_train + n_val]
    test = timestamp[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


# ---------------------------------------------------------------------
# PyG Dataset
# ---------------------------------------------------------------------
class TempHypEDataset(Dataset):
    """
    PyTorch Geometric dataset for temporal KG triples.

    Parameters mirror `read_triples_csv`. The dataset holds a list of
    time-ordered timestamp accessible via `get(idx)` / `len()`.
    """

    def __init__(
        self,
        data_path: str,
        sep: str = ",",
        has_header: bool = True,
        src_col: str = "head",
        rel_col: str = "rel",
        dst_col: str = "tail",
        time_col: str = "time",
        time_unit: str = "day",
        add_reverse_edges: bool = True,
        include_edge_type: bool = True,
        transform=None,
        pre_transform=None,
    ):
        # root=None keeps it in-memory; PyG requires these args in the signature
        super().__init__(None, transform, pre_transform)

        rows = read_triples_csv(
            data_path,
            sep=sep,
            has_header=has_header,
            src_col=src_col,
            rel_col=rel_col,
            dst_col=dst_col,
            time_col=time_col,
        )
        self.ent2id, self.rel2id = build_id_mappings(rows)
        self.timestamp = to_pyg_timestamp(
            rows,
            self.ent2id,
            self.rel2id,
            unit=time_unit,
            add_reverse_edges=add_reverse_edges,
            include_edge_type=include_edge_type,
        )

    # PyG Dataset API
    def len(self) -> int:
        return len(self.timestamp)

    def get(self, idx: int) -> Data:
        return self.timestamp[idx]


# ---------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------
def load_dataset_and_split(
    csv_path: str,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    **kwargs,
):
    """
    Load triples → build dataset → chronological split.

    Returns
    -------
    (dataset, train_timestamp, val_timestamp, test_timestamp)
    """
    ds = TempHypEDataset(csv_path, **kwargs)
    train, val, test = chronological_split(ds.timestamp, ratios=ratios)
    return ds, train, val, test


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_csv = Path("data/demo_triples.csv")
    demo_csv.parent.mkdir(parents=True, exist_ok=True)
    demo_csv.write_text(
        "head,rel,tail,time\n"
        "a,r1,b,2020-01-01\n"
        "b,r2,c,2020-01-01\n"
        "c,r1,d,2020-01-02\n"
        "d,r2,a,2020-01-03\n",
        encoding="utf-8",
    )

    ds, train, val, test = load_dataset_and_split(
        str(demo_csv),
        ratios=(0.5, 0.25, 0.25),
        sep=",",
        has_header=True,
        time_unit="day",
        add_reverse_edges=True,
        include_edge_type=True,
    )

    print(f"Entities: {len(ds.ent2id)} | Relations: {len(ds.rel2id)} | Timestamps: {len(ds)}")
    print("Split sizes:", len(train), len(val), len(test))
