import numpy as np, torch, networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
from t_star_sweep_score import sample_null
 
def louvain_labels(edges, n, seed=1432):
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    part = community_louvain.best_partition(G, random_state=seed)
    return np.array([part[i] for i in range(n)])
 
def same_comm(lab, E):
    E = np.asarray(E)
    return np.mean(lab[E[:, 0]] == lab[E[:, 1]])
 
for ds in ["ba", "sbm"]:
    ei = torch.load(f"data/{ds}/{ds}_edge_index.pt")
    n = int(ei.max()) + 1
    intact = louvain_labels(zip(ei[0].tolist(), ei[1].tolist()), n)   # what the pipeline uses
    true = torch.load(f"data/{ds}/{ds}_labels.pt").numpy()
    print(f"\n=== {ds.upper()} ===")
    for method in ["random", "targeted"]:
        for p in [30, 50]:
            r_int, n_int, r_dmg, n_dmg, aris = [], [], [], [], []
            for seed in range(10):
                d = torch.load(f"data/{ds}/perturbed/{method}_p{p}_seed{seed}.pt", weights_only=False)
                rem = np.array(d["removed_edges"])
                nul = sample_null(d["edge_index"], torch.tensor(rem), n, 200,
                                  np.random.default_rng(seed)).numpy()
                damaged = louvain_labels(d["kept_edges"], n)                 # honest labels
                r_int.append(same_comm(intact, rem));  n_int.append(same_comm(intact, nul))
                r_dmg.append(same_comm(damaged, rem)); n_dmg.append(same_comm(damaged, nul))
                if len(np.unique(true)) > 1:
                    aris.append(adjusted_rand_score(true, damaged))
            line = (f"{method:8s} p{p}:  intact labels: removed {np.mean(r_int):.2f} / null {np.mean(n_int):.2f}"
                    f"   |  damaged labels: removed {np.mean(r_dmg):.2f} / null {np.mean(n_dmg):.2f}")
            if aris:
                line += f"   |  ARI(damaged Louvain, true blocks) = {np.mean(aris):.2f}"
            print(line)