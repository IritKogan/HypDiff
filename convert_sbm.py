"""
convert_sbm.py  —  build an SBM dataset for HypDiff (lptask path).
 
The loader for `--dataset sbm` (utils/data_utils.load_synthetic_data) reads:
    data/sbm/sbm_edge_index.pt
    data/sbm/sbm_features.pt
    data/sbm/sbm_labels.pt
 
Unlike BA, SBM has PLANTED community structure - which is exactly what this
method (community/hub-guided hyperbolic diffusion) is built to exploit. We
generate a 3-block SBM (matching the hardcoded n_clusters=3 in Community_cluster),
strong intra-block density and sparse inter-block links, and write the three
tensors in the same format convert_ba.py used.
 
Usage:
    python convert_sbm.py
    python convert_sbm.py --p_in 0.30 --p_out 0.01 --seed 7
 
Then:
    python main.py --taskselect lptask --type train --dataset sbm --cuda -1 --epochs 200 --diff_epoc 50
"""
 
import argparse
import os
 
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix  # sanity-check only
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[34, 33, 33],
                    help="nodes per community (sums to total nodes)")
    ap.add_argument("--p_in", type=float, default=0.25, help="intra-community edge prob")
    ap.add_argument("--p_out", type=float, default=0.01, help="inter-community edge prob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sbm", help="output folder")
    args = ap.parse_args()
 
    sizes = list(args.sizes)
    k = len(sizes)
    n = sum(sizes)
 
    # block probability matrix: p_in on the diagonal, p_out off-diagonal
    P = np.full((k, k), args.p_out, dtype=float)
    np.fill_diagonal(P, args.p_in)
 
    G = nx.stochastic_block_model(sizes, P, seed=args.seed)
    G = nx.convert_node_labels_to_integers(G)
 
    # true community label per node (0..k-1) - meaningful for SBM (unlike BA's zeros)
    labels = np.zeros(n, dtype=np.int64)
    node = 0
    for c, s in enumerate(sizes):
        labels[node:node + s] = c
        node += s
    labels = torch.tensor(labels, dtype=torch.long)
 
    # edge_index [2, 2E]: both directions (undirected)
    edges = list(G.edges())
    if edges:
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
    else:
        src, dst = [], []
    edge_index = torch.tensor([src, dst], dtype=torch.long)
 
    # identity node features (featureless graph -> one-hot identity), shape [n, n]
    features = torch.eye(n, dtype=torch.float)
 
    os.makedirs(args.out, exist_ok=True)
    torch.save(edge_index, os.path.join(args.out, "sbm_edge_index.pt"))
    torch.save(features, os.path.join(args.out, "sbm_features.pt"))
    torch.save(labels, os.path.join(args.out, "sbm_labels.pt"))
 
    adj = to_scipy_sparse_matrix(edge_index)
    # count intra vs inter edges as a structure sanity check
    ei = edge_index.numpy()
    intra = inter = 0
    seen = set()
    for u, v in zip(ei[0], ei[1]):
        u, v = int(u), int(v)
        if u >= v:
            continue
        if (u, v) in seen:
            continue
        seen.add((u, v))
        if labels[u] == labels[v]:
            intra += 1
        else:
            inter += 1
    print(f"SBM: {n} nodes, {k} communities {sizes}, seed={args.seed}")
    print(f"  edges: {intra} intra + {inter} inter = {intra + inter} undirected")
    print(f"  intra/inter ratio = {intra / max(inter,1):.1f}  (higher = stronger communities)")
    print(f"  edge_index {tuple(edge_index.shape)}  features {tuple(features.shape)}  labels {tuple(labels.shape)}")
    print(f"Wrote -> {args.out}/sbm_edge_index.pt, sbm_features.pt, sbm_labels.pt")
    print("Now run:  python main.py --taskselect lptask --type train --dataset sbm --cuda -1 --epochs 200 --diff_epoc 50")
 
 
if __name__ == "__main__":
    main()