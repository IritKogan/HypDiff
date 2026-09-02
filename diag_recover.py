"""

Stage 2 diagnostic v2: is it a SCALE problem or a SIGNAL problem?
 
Decodes the recovered embedding at BOTH curvatures:

    c = model.c (0.6)  - the default decode path

    c = 1.0            - the curvature the diffusion expmaps onto
 
For three edge groups: surviving / removed / random non-edges.

If removed >> random at some curvature -> signal exists, it's a scale/threshold

matter (Stage 4). If removed ~= random everywhere -> BA has no recoverable

signal for this method (try SBM).
 
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
 
 
def probs_at(emb, idx, c):

    with torch.no_grad():

        ein = emb[idx[:, 0]]

        eout = emb[idx[:, 1]]

        sq = model.manifold.sqdist(ein, eout, torch.tensor([float(c)]))

        p = model.dc.forward(sq).cpu().numpy()

        sq = sq.cpu().numpy()

    return p, sq
 
 
def block(title, emb, c):

    print(f"\n-- {title}  (Fermi-Dirac r={args.r}, t={args.t}) --")

    for tag, idx in [("surviving", surv), ("removed", removed), ("random", rand)]:

        p, sq = probs_at(emb, idx, c)

        print(f"  {tag:10s} n={len(p):4d}  prob[mean={p.mean():.3e} max={p.max():.3e}]  "

              f"sqdist[mean={sq.mean():.2f} min={sq.min():.2f}]")
 
 
block("RECOVERED sa, decode at c=0.6 (default)", sa, 0.6)

block("RECOVERED sa, decode at c=1.0 (matches expmap)", sa, 1.0)

block("RAW encoder h0, decode at c=0.6", h0, 0.6)
 
print("\nRanking is what matters for a stability score: at whichever curvature,")

print("does removed's mean prob clearly exceed random's? If yes -> Stage 4 calibration.")
 