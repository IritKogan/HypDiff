"""
Stage 2 / Stage 4 probe: how does recovery quality depend on t_star?
 
Now dataset-parameterized:
    python t_star_sweep.py          # defaults to ba
    python t_star_sweep.py sbm      # runs on SBM
 
For each t_star we run K=3 reconstructions and report the mean Fermi-Dirac
probability (decode at c=0.6) for surviving / removed / random edges.
Good recovery = removed clearly above random at a usable scale.
"""
 
import os
import sys
import glob
 
import numpy as np
import torch
 
from recover import _load_args, load_centers, damaged_to_inputs
from models.base_models import LPModel
from models.direction_diffusion import diffusion
 
ds = sys.argv[1] if len(sys.argv) > 1 else "ba"
device = "cpu"
frozen = os.path.join("logs", "lptask", ds, "frozen_original")
pert = os.path.join("data", ds, "perturbed", "*.pt")
files = sorted(glob.glob(pert))
assert files, f"no perturbed files in data/{ds}/perturbed/ - run Stage 1 for {ds} first."
path = files[0]
 
d = torch.load(path)
ei = d["edge_index"]
removed = torch.as_tensor(d["removed_edges"], dtype=torch.long)
if removed.dim() == 1:
    removed = removed.view(-1, 2)
n = int(d.get("num_nodes", int(ei.max()) + 1))
print(f"dataset={ds}  file={os.path.basename(path)}  nodes={n}  removed={len(removed)}")
 
args = _load_args(frozen, device)
label, center = load_centers(frozen, device)
adj_norm, features = damaged_to_inputs(ei, n, args)
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
 
surv = ei.t()
surv = surv[surv[:, 0] < surv[:, 1]]
edgeset = set(map(tuple, surv.tolist())) | set(map(tuple, removed.tolist()))
rng = np.random.default_rng(0)
rand = []
while len(rand) < 200:
    a, b = sorted(rng.integers(0, n, 2))
    if a != b and (a, b) not in edgeset:
        rand.append((a, b))
rand = torch.tensor(rand, dtype=torch.long)
 
 
def meanprob(emb, idx):
    with torch.no_grad():
        return float(model.decode(emb, idx).mean())
 
 
print(f"\nbaseline h0 (no diffusion):  surv={meanprob(h0, surv):.4f}  "
      f"removed={meanprob(h0, removed):.4f}  random={meanprob(h0, rand):.4f}")
print("\nt*     surv    removed  random   (removed-random)")
for t_star in [2, 5, 10, 20, 40, 80, 150, 199]:
    s, r, nn = [], [], []
    for _ in range(3):
        with torch.no_grad():
            sa = diff.p_sample_loop(h0.clone(), adj_norm, label, center, t_star=t_star)
        s.append(meanprob(sa, surv))
        r.append(meanprob(sa, removed))
        nn.append(meanprob(sa, rand))
    s, r, nn = np.mean(s), np.mean(r), np.mean(nn)
    print(f"{t_star:<4d}  {s:.4f}  {r:.4f}  {nn:.4f}   {r - nn:+.4f}")
 
print("\nGood sign: some t* where removed clearly exceeds random at usable scale.")