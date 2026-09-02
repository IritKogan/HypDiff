"""
Stage 2 - Guided Recovery Engine (lptask path).
 
Takes ONE damaged graph (a Stage 1 perturbation output) and returns K
reconstructions, by:
  (a) encoding the damaged graph with the FROZEN, already-trained HGCN encoder,
  (b) noising its latent only up to step t_star (partial start), and
  (c) running the reverse loop K times, reusing the ORIGINAL graph's community
      centers (frozen_original/center.npy, label.npy).
 
Nothing is trained here. Encoder (model.pth) and diffusion (sample.pt) are loaded
read-only from the frozen_original folder produced in Stage 2 step 4.
 
Run from the repo root:  python recover.py
"""
 
import os
import json
import glob
import argparse
 
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.utils as tu
 
from utils.data_utils import process
from models.base_models import LPModel
from models.direction_diffusion import diffusion
 
 
# --------------------------------------------------------------------------
# frozen assets (all computed once on the ORIGINAL, undamaged graph)
# --------------------------------------------------------------------------
 
def _load_args(frozen_dir, device):
    """Rebuild the exact training args from the frozen config.json."""
    with open(os.path.join(frozen_dir, "config.json")) as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    args.cuda = 0 if device != "cpu" else -1
    args.device = device
    return args
 
 
def load_centers(frozen_dir, device):
    """Original-graph community centers + per-node labels (reused for every copy)."""
    label = np.load(os.path.join(frozen_dir, "label.npy"))
    center = torch.tensor(np.load(os.path.join(frozen_dir, "center.npy"))).to(device)
    return label, center
 
 
# --------------------------------------------------------------------------
# (a) encode the damaged graph
# --------------------------------------------------------------------------
 
def damaged_to_inputs(edge_index, num_nodes, args):
    """
    Damaged edge_index -> (normalized sparse adj, identity features), using the
    SAME preprocessing the encoder was trained with (process: row-normalize A+I,
    row-normalize identity feats) -- but WITHOUT mask_edges, so we encode the
    damaged graph's real surviving structure, not a further-split version.
    """
    adj = tu.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj = sp.csr_matrix(adj)
    feats = sp.eye(num_nodes)
    adj_norm, features = process(adj, feats, args.normalize_adj, args.normalize_feats)
    return adj_norm, features
 
 
# --------------------------------------------------------------------------
# main entry: one damaged graph -> K reconstructions (+ removed-edge probs)
# --------------------------------------------------------------------------
 
def recover_one(edge_index, removed_edges, num_nodes, frozen_dir,
                t_star, K, device="cpu"):
    """
    Parameters
    ----------
    edge_index   : LongTensor [2, E] - the DAMAGED graph (Stage 1 output).
    removed_edges: LongTensor/list [E_rm, 2] - edges Stage 1 removed (ground truth
                   for Stage 3). Pass None to skip probability decoding.
    num_nodes    : int - node count (unchanged from the original graph).
    frozen_dir   : path to logs/lptask/<ds>/frozen_original.
    t_star       : int in [1, num_timesteps-1] - truncation step (partial start).
    K            : int - reconstructions per damaged graph.
 
    Returns
    -------
    recons : list of K tensors, each [num_nodes, dim] - recovered embeddings.
    probs  : list of K tensors [E_rm] - Fermi-Dirac recovery probability for each
             removed edge (for Stage 3). None if removed_edges is None.
    """
    args = _load_args(frozen_dir, device)
    label, center = load_centers(frozen_dir, device)
 
    # (a) encode the DAMAGED graph with the FROZEN encoder
    adj_norm, features = damaged_to_inputs(edge_index, num_nodes, args)
    args.n_nodes, args.feat_dim = features.shape
    args.nb_false_edges = 1          # only used by compute_metrics (unused here)
    args.nb_edges = 1
 
    model = LPModel(args).to(device)
    model.load_state_dict(torch.load(os.path.join(frozen_dir, "model.pth"),
                                     map_location=device))
    model.eval()
 
    adj_norm = adj_norm.to(device)
    features = features.to(device)
    with torch.no_grad():
        _, _, h0, _ = model.encode(features, adj_norm)   # h0: [num_nodes, dim]
 
    # frozen diffusion (trained on the ORIGINAL graph)
    diff = diffusion(args, dim=h0.shape[1]).to(device)
    diff.load_state_dict(torch.load(os.path.join(frozen_dir, "sample.pt"),
                                    map_location=device))
    diff.eval()
 
    # prepare removed-edge index once
    idx = None
    if removed_edges is not None:
        idx = torch.as_tensor(removed_edges, dtype=torch.long, device=device)
        if idx.dim() == 1:
            idx = idx.view(-1, 2)
 
    # (b)+(c) K partial-start reconstructions, reusing ORIGINAL centers
    recons, probs = [], ([] if idx is not None else None)
    for _ in range(K):
        with torch.no_grad():
            # clone h0 every call: p_sample_loop mutates its input in place
            sa_k = diff.p_sample_loop(h0.clone().to(device), adj_norm,
                                      label, center, t_star=t_star)
        recons.append(sa_k)
        if idx is not None:
            with torch.no_grad():
                probs.append(model.decode(sa_k, idx).cpu())   # Fermi-Dirac probs
    return recons, probs
 
 
# --------------------------------------------------------------------------
# Stage 2 step 6: smoke test on one Stage 1 damaged copy
# --------------------------------------------------------------------------
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen = os.path.join("logs", "lptask", "ba", "frozen_original")
    assert os.path.isdir(frozen), f"missing {frozen} - run Stage 2 step 4 first."
 
    files = sorted(glob.glob(os.path.join("data", "ba", "perturbed", "*.pt")))
    assert files, "No perturbed files in data/ba/perturbed/ - run Stage 1 first."
    path = files[0]
    d = torch.load(path)
    edge_index = d["edge_index"]
    removed = d["removed_edges"]
    n = int(d.get("num_nodes", int(edge_index.max()) + 1))
 
    t_star, K = 100, 5
    print(f"[recover] device={device}  file={os.path.basename(path)}")
    print(f"[recover] nodes={n}  removed_edges={len(removed)}  t_star={t_star}  K={K}")
 
    recons, probs = recover_one(edge_index, removed, n, frozen,
                                t_star=t_star, K=K, device=device)
 
    # --- checks (this is the Stage 2 deliverable proof) ---
    print(f"got {len(recons)} reconstructions, each shape {tuple(recons[0].shape)}")
 
    diff01 = (recons[0] - recons[1]).abs().mean().item()
    print(f"mean|recon0 - recon1| = {diff01:.6f}   (should be > 0: K copies must differ)")
 
    p0 = probs[0]
    inrange = bool((p0 >= 0).all() and (p0 <= 1).all())
    print(f"removed-edge recovery prob: min={p0.min():.3f} mean={p0.mean():.3f} max={p0.max():.3f}")
    print(f"all probs in [0,1]: {inrange}")
    print("[recover] OK" if (len(recons) == K and diff01 > 0 and inrange)
          else "[recover] CHECK FAILED - inspect output above")