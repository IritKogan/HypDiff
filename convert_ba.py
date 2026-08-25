"""
convert_ba.py  —  Stage 0 helper for HypDiff (RingBDStack/HypDiff)
 
Problem this solves
-------------------
The repo ships data/BA.pkl, but the lptask loader for `--dataset ba`
(utils/data_utils.py -> load_synthetic_data) does NOT read that pickle.
It calls torch.load on three files that do not exist in the repo:
 
    data/ba/ba_edge_index.pt
    data/ba/ba_features.pt
    data/ba/ba_labels.pt
 
BA.pkl is actually a list of 500 networkx graphs (100 nodes each), and the
lptask pipeline processes ONE graph at a time. This script picks one graph
out of that list and writes the three tensors the loader expects, so the
baseline smoke test can run unmodified.
 
Usage
-----
    python convert_ba.py                # uses graph #0 from data/BA.pkl
    python convert_ba.py --index 7      # pick a different graph
    python convert_ba.py --pkl data/BA.pkl --out data/ba
 
After running, launch the baseline with:
    python main.py --taskselect lptask --type train --dataset ba
"""
 
import argparse
import os
import pickle
 
import networkx as nx
import torch
from torch_geometric.utils import to_scipy_sparse_matrix  # sanity-check only
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/BA.pkl", help="path to BA.pkl")
    ap.add_argument("--index", type=int, default=0, help="which graph in the list to use")
    ap.add_argument("--out", default="data/ba", help="output folder for the .pt files")
    args = ap.parse_args()
 
    # 1. Load the list of graphs and pick one.
    with open(args.pkl, "rb") as f:
        graphs = pickle.load(f)
    if not isinstance(graphs, (list, tuple)):
        graphs = [graphs]
    if args.index >= len(graphs):
        raise IndexError(f"index {args.index} out of range (file has {len(graphs)} graphs)")
    G = graphs[args.index]
 
    # 2. Relabel nodes 0..n-1 so tensor indices are contiguous (safe even if already clean).
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
 
    # 3. edge_index [2, 2E]: both directions, because the graph is undirected.
    edges = list(G.edges())
    if edges:
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
    else:
        src, dst = [], []
    edge_index = torch.tensor([src, dst], dtype=torch.long)
 
    # 4. Node features: identity matrix, matching the repo's own synthetic-data default
    #    (featureless graph -> one-hot node identity). Shape [n, n].
    features = torch.eye(n, dtype=torch.float)
 
    # 5. Labels: BA graphs have no community labels; zeros are fine for the
    #    link-prediction reconstruction task (labels are not used as a training target here).
    labels = torch.zeros(n, dtype=torch.long)
 
    # 6. Write the three files the loader reads.
    os.makedirs(args.out, exist_ok=True)
    torch.save(edge_index, os.path.join(args.out, "ba_edge_index.pt"))
    torch.save(features, os.path.join(args.out, "ba_features.pt"))
    torch.save(labels, os.path.join(args.out, "ba_labels.pt"))
 
    # 7. Sanity check: round-trip through the exact call the loader uses.
    adj = to_scipy_sparse_matrix(edge_index)
    print(f"Picked graph #{args.index}: {n} nodes, {G.number_of_edges()} undirected edges")
    print(f"  edge_index: {tuple(edge_index.shape)}  (2 x directed-edge-count)")
    print(f"  features:   {tuple(features.shape)}")
    print(f"  labels:     {tuple(labels.shape)}")
    print(f"  adj (loader view): {adj.shape}, nnz={adj.nnz}")
    print(f"Wrote -> {args.out}/ba_edge_index.pt, ba_features.pt, ba_labels.pt")
    print("Now run:  python main.py --taskselect lptask --type train --dataset ba")
 
 
if __name__ == "__main__":
    main()