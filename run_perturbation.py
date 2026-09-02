import torch
from perturbation import perturb, save_batch
 
edge_index = torch.load("data/ba/ba_edge_index.pt")
num_nodes = int(edge_index.max()) + 1
 
for p in (0.1, 0.2, 0.3, 0.4, 0.5):
    for method in ("random", "targeted"):
        batch = perturb(edge_index, num_nodes, p=p, method=method, N=10, base_seed=0)
        save_batch(batch)
        print(f"saved {method} at {int(p*100)}% damage")