"""
Stage 1 - Graph Perturbation Module
-----------------------------------
Produces N damaged copies of a graph at a damage level p, using two attack
methods: Random Edge Removal and Targeted (degree-based) Attack.
 
Only the adjacency changes. The node set and the node features are NEVER touched.
For every damaged copy we also record exactly which edges were removed, because
Stage 3 (scoring) needs that as ground truth ("did the edge come back?").
 
Input graph format matches Stage 0: edge_index is a [2, num_directed_edges]
LongTensor listing every edge in BOTH directions (u->v and v->u).
"""
 
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
 
 
# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------
@dataclass
class PerturbedGraph:
    edge_index: torch.Tensor   # [2, 2*len(kept)] both directions -> feeds the HGCN encoder
    removed_edges: list        # list of (u, v) undirected pairs that were removed (ground truth)
    kept_edges: list           # list of (u, v) undirected pairs that survived
    num_nodes: int             # unchanged from the original graph
    method: str                # "random" or "targeted"
    p: float                   # damage fraction, e.g. 0.20
    seed: int                  # the seed used for this copy
 
 
# ----------------------------------------------------------------------------
# Representation helpers
# ----------------------------------------------------------------------------
def edge_index_to_undirected_set(edge_index):
    """[2,E] tensor (both directions) -> sorted list of canonical (u,v) with u<v, no self-loops."""
    ei = edge_index.cpu().numpy()
    edges = set()
    for u, v in zip(ei[0], ei[1]):
        u, v = int(u), int(v)
        if u == v:
            continue
        edges.add((min(u, v), max(u, v)))
    return sorted(edges)
 
 
def undirected_set_to_edge_index(edges):
    """list of (u,v) -> [2, 2*len] LongTensor with both directions (matches Stage 0 format)."""
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]
    return torch.tensor([src, dst], dtype=torch.long)
 
 
def num_edges_to_remove(E, p):
    """Exact, deterministic count of edges to remove."""
    return int(round(p * E))
 
 
# ----------------------------------------------------------------------------
# Attack method A: Random Edge Removal
# ----------------------------------------------------------------------------
def random_edge_removal(edges, k, rng):
    idx = rng.choice(len(edges), size=k, replace=False)
    removed = {edges[i] for i in idx}
    kept = [e for e in edges if e not in removed]
    return kept, sorted(removed)
 
 
# ----------------------------------------------------------------------------
# Attack method B: Targeted (degree-based) Attack
# ----------------------------------------------------------------------------
def edge_scores(edges, num_nodes):
    """Importance of each edge = product of its two endpoint degrees (on the ORIGINAL graph)."""
    deg = np.zeros(num_nodes, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return np.array([deg[u] * deg[v] for (u, v) in edges], dtype=np.float64)
 
 
def targeted_attack(edges, num_nodes, k, rng, probabilistic=True):
    scores = edge_scores(edges, num_nodes)
    if not probabilistic:
        # deterministic: remove the k highest-scoring edges
        order = np.argsort(-scores, kind="stable")
        removed_idx = order[:k]
    else:
        # probabilistic: sample k edges WITHOUT replacement, weighted by score.
        # Still targets high-degree edges, but each seed gives an independent copy.
        w = scores / scores.sum()
        removed_idx = rng.choice(len(edges), size=k, replace=False, p=w)
    removed = {edges[i] for i in removed_idx}
    kept = [e for e in edges if e not in removed]
    return kept, sorted(removed)
 
 
# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def perturb(edge_index, num_nodes, p, method="random", N=10, base_seed=0,
            probabilistic_targeted=True):
    """
    Returns a list of N PerturbedGraph objects.
    method: "random" or "targeted".
    Each copy uses seed = base_seed + i, so the whole run is reproducible from base_seed.
    """
    edges = edge_index_to_undirected_set(edge_index)
    E = len(edges)
    k = num_edges_to_remove(E, p)
 
    out = []
    for i in range(N):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        if method == "random":
            kept, removed = random_edge_removal(edges, k, rng)
        elif method == "targeted":
            kept, removed = targeted_attack(edges, num_nodes, k, rng,
                                            probabilistic=probabilistic_targeted)
        else:
            raise ValueError(f"unknown method: {method}")
 
        out.append(PerturbedGraph(
            edge_index=undirected_set_to_edge_index(kept),
            removed_edges=removed,
            kept_edges=kept,
            num_nodes=num_nodes,
            method=method, p=p, seed=seed,
        ))
    return out
 
 
# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------
def save_batch(batch, out_dir="data/ba/perturbed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for g in batch:
        fname = f"{g.method}_p{int(round(g.p*100)):02d}_seed{g.seed}.pt"
        path = Path(out_dir) / fname
        torch.save({
            "edge_index": g.edge_index,
            "removed_edges": g.removed_edges,
            "kept_edges": g.kept_edges,
            "num_nodes": g.num_nodes,
            "method": g.method, "p": g.p, "seed": g.seed,
        }, path)
        paths.append(str(path))
    return paths
 
 
# ----------------------------------------------------------------------------
# CLI demo on the Stage 0 BA graph
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    edge_index = torch.load("data/ba/ba_edge_index.pt")
    num_nodes = int(edge_index.max()) + 1
    E = len(edge_index_to_undirected_set(edge_index))
    print(f"Loaded BA graph: {num_nodes} nodes, {E} undirected edges")
 
    for method in ("random", "targeted"):
        batch = perturb(edge_index, num_nodes, p=0.20, method=method, N=10, base_seed=0)
        removed_counts = {len(g.removed_edges) for g in batch}
        print(f"{method:8s}: removed {removed_counts} edges per copy (target = {num_edges_to_remove(E, 0.20)})")
        save_batch(batch)
    print("Saved batches to data/ba/perturbed/")