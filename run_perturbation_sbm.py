"""
Stage 1 sweep for SBM - reuses the (dataset-agnostic) perturbation module,
just points it at the SBM graph and saves under data/sbm/perturbed/.
 
Run from repo root:  python run_perturbation_sbm.py
"""
 
import torch
from perturbation import perturb, save_batch, edge_index_to_undirected_set
 
edge_index = torch.load("data/sbm/sbm_edge_index.pt")
num_nodes = int(edge_index.max()) + 1
E = len(edge_index_to_undirected_set(edge_index))
print(f"SBM graph: {num_nodes} nodes, {E} undirected edges")
 
for p in (0.1, 0.2, 0.3, 0.4, 0.5):
    for method in ("random", "targeted"):
        batch = perturb(edge_index, num_nodes, p=p, method=method, N=10, base_seed=0)
        save_batch(batch, out_dir="data/sbm/perturbed")
        print(f"saved {method} at {int(p*100)}% damage")
print("Done -> data/sbm/perturbed/")