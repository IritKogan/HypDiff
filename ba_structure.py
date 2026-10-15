import torch, numpy as np, networkx as nx
import community as community_louvain   # python-louvain
 
ei = torch.load("data/ba/ba_edge_index.pt")
n = int(ei.max()) + 1
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
G.remove_edges_from(nx.selfloop_edges(G))
 
# 1) hubs: degree distribution
deg = np.array([G.degree(i) for i in range(n)])
print(f"nodes={n} edges={G.number_of_edges()}")
print(f"degree: min={deg.min()} median={np.median(deg)} mean={deg.mean():.2f} max={deg.max()}")
print("top-10 degrees:", sorted(deg)[-10:])
 
# 2) communities: Louvain modularity on BA
part = community_louvain.best_partition(G, random_state=0)
Q = community_louvain.modularity(part, G)
k = len(set(part.values()))
print(f"Louvain on BA: k={k} communities, modularity Q={Q:.3f}")
 
# 3) null: same degrees, edges wired at random (no communities by construction)
Qs = []
for s in range(10):
    H = nx.Graph(nx.configuration_model(deg.tolist(), seed=s))
    H.remove_edges_from(nx.selfloop_edges(H))
    Qs.append(community_louvain.modularity(community_louvain.best_partition(H, random_state=0), H))
print(f"random graphs with same degrees: Q = {np.mean(Qs):.3f} +- {np.std(Qs):.3f}")
 
# 4) stability: does Louvain give the same partition with different seeds?
from sklearn.metrics import adjusted_rand_score
labs = [np.array([community_louvain.best_partition(G, random_state=s)[i] for i in range(n)]) for s in range(5)]
aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(5) for j in range(i+1, 5)]
print(f"Louvain seed-to-seed agreement (ARI): mean={np.mean(aris):.2f}")