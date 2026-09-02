"""
Stage 2 diagnostic: is the recovery signal there, or is decode dead?
 
Compares Fermi-Dirac edge probabilities for three groups on the SAME damaged
graph + reconstruction:
    surviving edges  (still present in the damaged graph)
    removed edges    (Stage 1 ground truth - what recovery should bring back)
    random non-edges (pairs that were never connected)
 
Also decodes on the raw encoder embedding h0 (no diffusion) as a reference, so
we can see whether the diffusion step moved anything.
 
Run from repo root:  python diag_recover.py
"""
 
import os
import glob
 
import numpy as np
import torch
 
from recover import _load_args, load_centers, damaged_to_inputs
from models.base_models import LPModel
from models.direction_diffusion import diffusion
 
device = "cpu"
frozen = os.path.join("logs", "lptask", "ba", "frozen_original")
path = sorted(glob.glob(os.path.join("data", "ba", "perturbed", "*.pt")))[0]
 
d = torch.load(path)
edge_index = d["edge_index"]
removed = torch.as_tensor(d["removed_edges"], dtype=torch.long)
if removed.dim() == 1:
    removed = removed.view(-1, 2)
n = int(d.get("num_nodes", int(edge_index.max()) + 1))
print(f"file={os.path.basename(path)}  nodes={n}  removed={len(removed)}")
 
# rebuild frozen model + one reconstruction (t_star=100)
args = _load_args(frozen, device)
label, center = load_centers(frozen, device)
adj_norm, features = damaged_to_inputs(edge_index, n, args)
args.n_nodes, args.feat_dim = features.shape
args.nb_false_edges = 1
args.nb_edges = 1
 
model = LPModel(args).to(device)
model.load_state_dict(torch.load(os.path.join(frozen, "model.pth"), map_location=device))
model.eval()
with torch.no_grad():
    _, _, h0, _ = model.encode(features, adj_norm)
 
diff = diffusion(args, dim=h0.shape[1]).to(device)
diff.load_state_dict(torch.load(os.path.join(frozen, "sample.pt"), map_location=device))
diff.eval()
with torch.no_grad():
    sa = diff.p_sample_loop(h0.clone(), adj_norm, label, center, t_star=100)
 
# edge groups (undirected, u < v)
surv = edge_index.t()
surv = surv[surv[:, 0] < surv[:, 1]]
 
edgeset = set(map(tuple, surv.tolist())) | set(map(tuple, removed.tolist()))
rng = np.random.default_rng(0)
rand = []
while len(rand) < 200:
    a, b = sorted(rng.integers(0, n, 2))
    if a != b and (a, b) not in edgeset:
        rand.append((a, b))
rand = torch.tensor(rand, dtype=torch.long)
 
 
def show(tag, emb, idx):
    with torch.no_grad():
        p = model.decode(emb, idx).cpu().numpy()
    print(f"  {tag:22s} n={len(p):4d}  min={p.min():.3e}  mean={p.mean():.3e}  max={p.max():.3e}")
 
 
print("\n-- decode on RECOVERED embedding (sa, after diffusion) --")
show("surviving edges", sa, surv)
show("removed edges", sa, removed)
show("random non-edges", sa, rand)
 
print("\n-- decode on RAW encoder embedding (h0, no diffusion) --")
show("surviving edges", h0, surv)
show("removed edges", h0, removed)
show("random non-edges", h0, rand)
 
# also report the raw hyperbolic distances for removed edges (before Fermi-Dirac),
# to see whether it's the distance or the r,t decoder params squashing to 0
with torch.no_grad():
    din = sa[removed[:, 0]]
    dout = sa[removed[:, 1]]
    sq = model.manifold.sqdist(din, dout, model.c).cpu().numpy()
print(f"\nremoved-edge sqdist on sa:  min={sq.min():.3f}  mean={sq.mean():.3f}  max={sq.max():.3f}")
print(f"Fermi-Dirac params: r={args.r}  t={args.t}  (prob=0.5 at dist=r)")