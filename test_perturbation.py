"""
Stage 1 unit tests. Run with:  python -m pytest test_perturbation.py -v
or plain:                        python test_perturbation.py
"""
import numpy as np
import torch
 
from perturbation import (
    perturb, edge_index_to_undirected_set, undirected_set_to_edge_index,
    num_edges_to_remove, edge_scores,
)
 
 
# ---- a tiny, hand-checkable mock graph ----
# 6 nodes. Node 0 is a hub connected to 1,2,3,4,5. Plus one edge 1-2.
# Degrees: node0=5, node1=2, node2=2, node3=1, node4=1, node5=1
# Edge (0,1) score = 5*2 = 10 ; edge (3,0)=5*1=5 ; edge (1,2)=2*2=4 ...
MOCK_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2)]  # E = 6
MOCK_NUM_NODES = 6
 
 
def mock_edge_index():
    return undirected_set_to_edge_index(MOCK_EDGES)
 
 
def original_set():
    return set(MOCK_EDGES)
 
 
# 1. Exact count removed, for every damage level
def test_exact_count():
    E = len(MOCK_EDGES)
    for p in (0.1, 0.2, 0.3, 0.4, 0.5):
        k = num_edges_to_remove(E, p)
        batch = perturb(mock_edge_index(), MOCK_NUM_NODES, p=p, method="random", N=5)
        for g in batch:
            assert len(g.removed_edges) == k, (p, len(g.removed_edges), k)
 
 
# 2. Node set stays intact (num_nodes never changes)
def test_node_set_intact():
    batch = perturb(mock_edge_index(), MOCK_NUM_NODES, p=0.5, method="random", N=5)
    for g in batch:
        assert g.num_nodes == MOCK_NUM_NODES
        # every node id in the output is within the original range
        if g.edge_index.numel() > 0:
            assert int(g.edge_index.max()) < MOCK_NUM_NODES
 
 
# 3. No invented edges: kept ∪ removed == original, kept ∩ removed == ∅
def test_no_invented_edges():
    for method in ("random", "targeted"):
        batch = perturb(mock_edge_index(), MOCK_NUM_NODES, p=0.3, method=method, N=5)
        for g in batch:
            kept, removed = set(g.kept_edges), set(g.removed_edges)
            assert kept | removed == original_set()
            assert kept & removed == set()
 
 
# 4. Output edge_index is symmetric and self-loop free
def test_symmetry():
    batch = perturb(mock_edge_index(), MOCK_NUM_NODES, p=0.3, method="random", N=3)
    for g in batch:
        ei = g.edge_index.numpy()
        pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
        for u, v in pairs:
            assert u != v                       # no self-loops
            assert (v, u) in pairs              # both directions present
 
 
# 5. Reproducibility: same seed -> identical removal; different seeds -> usually differ
def test_reproducibility():
    a = perturb(mock_edge_index(), MOCK_NUM_NODES, p=0.3, method="random", N=1, base_seed=42)[0]
    b = perturb(mock_edge_index(), MOCK_NUM_NODES, p=0.3, method="random", N=1, base_seed=42)[0]
    assert a.removed_edges == b.removed_edges
    c = perturb(mock_edge_index(), MOCK_NUM_NODES, p=0.3, method="random", N=1, base_seed=999)[0]
    # not guaranteed but overwhelmingly likely to differ
    assert not (a.removed_edges == c.removed_edges and a.seed != c.seed) or a.seed == c.seed
 
 
# 6a. Targeted DETERMINISTIC removes the highest degree-product edges first
def test_targeted_deterministic():
    scores = edge_scores(MOCK_EDGES, MOCK_NUM_NODES)
    # remove k=2 -> should be the two highest-score edges
    k = 2
    order = np.argsort(-scores, kind="stable")
    expected = {MOCK_EDGES[i] for i in order[:k]}
    g = perturb(mock_edge_index(), MOCK_NUM_NODES, p=k/len(MOCK_EDGES),
                method="targeted", N=1, probabilistic_targeted=False)[0]
    assert set(g.removed_edges) == expected
 
 
# 6b. Targeted PROBABILISTIC removes high-score edges far more often than low-score ones
def test_targeted_probabilistic_bias():
    trials = 300
    counts = {e: 0 for e in MOCK_EDGES}
    for s in range(trials):
        g = perturb(mock_edge_index(), MOCK_NUM_NODES, p=1/len(MOCK_EDGES),
                    method="targeted", N=1, base_seed=s, probabilistic_targeted=True)[0]
        for e in g.removed_edges:
            counts[e] += 1
    hub_edge = (0, 1)          # score 10
    leaf_edge = (0, 5)         # score 5 ... still hub, use (1,2)=4 vs a leaf
    assert counts[(0, 1)] > counts[(1, 2)]   # higher-score edge removed more often
 
 
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")