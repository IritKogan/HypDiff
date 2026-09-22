"""
Stage 4 fix (Phase 1) — retrain the diffusion with CORRECT community guidance.
 
Diagnosis this addresses
------------------------
The frozen SBM diffusion was trained AND sampled with a degenerate community
labeling: the unsupervised HyperbolicKMeans in diff.Community_cluster collapses
the 3 planted SBM blocks into one cluster (label.npy = {0: 99, 2: 1}, ARI vs
truth ~ 0). The paper's anisotropic guidance (tran_direction uses per-community
sign vectors; the reverse loop reuses per-community centers) is therefore OFF.
That is why raising t* homogenizes then diverges the latent instead of healing.
 
The encoder embedding DOES separate the true communities cleanly
(nearest-true-centroid accuracy = 1.000, silhouette 0.44), so the fix is to feed
the correct communities and retrain the diffusion so it actually learns
community-guided denoising.
 
What this script does (faithful reproduction of diff.hyperdiff, one variable changed)
------------------------------------------------------------------------------------
  * loads args from the frozen config.json,
  * reuses the SAME adj_train_norm (via load_data) and the SAME frozen
    embeddings.npy that diff.hyperdiff trained on,
  * REPLACES the collapsed label/center with the ground-truth SBM blocks
    (data/sbm/sbm_labels.pt) and per-community Fréchet-mean centers
    (logmap0 -> mean -> expmap0, exactly like diffusion.cal_u0),
  * retrains the diffusion (restrict=True, Adam lr=0.001, diff_epoc epochs) —
    identical to hyperdiff except Community_cluster is skipped,
  * writes a NEW frozen dir (default logs/lptask/sbm/frozen_truecomm),
    leaving frozen_original untouched for comparison.
 
Then re-run the t* sweep against the new dir:
    python t_star_sweep_score.py --p-values 30 --t-stars 5 15 30 60 100 150 199 \
        --frozen logs/lptask/sbm/frozen_truecomm
 
Run from the HypDiff repo root.
"""
 
from __future__ import annotations
 
import os
import json
import shutil
import argparse
 
import numpy as np
import torch
import torch.optim as optim
 
# same data path convention as main.py
os.environ.setdefault("DATAPATH", "data/")
os.environ.setdefault("LOG_DIR", "logs/lptask/")
 
from utils.data_utils import load_data
from models.direction_diffusion import diffusion
 
 
def build_args(frozen_dir, device):
    with open(os.path.join(frozen_dir, "config.json")) as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    args.cuda = 0 if device != "cpu" else -1
    args.device = device
    return args
 
 
def clamp_to_ball(h0, c=1.0, eps=1e-4):
    """Scale any node with ||x|| >= 1/sqrt(c) to just inside the Poincare ball.
 
    The diffusion's manifold ops are all hardcoded to c=1.0, but the c=0.6 encoder
    can emit points with ||x|| >= 1 (here 2/100 nodes, max 1.06). logmap0 at the
    origin then hits artanh(||x||) -> NaN, which is exactly what poisoned the first
    retrain. Nudging only those boundary nodes inside the c=1.0 ball fixes it and
    leaves the other 98 untouched.
    """
    bound = (1.0 / (c ** 0.5)) * (1.0 - eps)
    nrm = h0.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    scale = torch.clamp(bound / nrm, max=1.0)
    n_clamped = int((scale < 1.0).sum())
    return h0 * scale, n_clamped
 
 
def community_centers(h0, labels, manifold, k, c=1.0):
    """Per-community Fréchet mean: logmap0 -> tangent mean -> expmap0 (== cal_u0).
 
    Assumes h0 is already inside the c-ball (see clamp_to_ball)."""
    centers = []
    for cc in range(k):
        idx = np.where(labels == cc)[0]
        assert len(idx) > 0, f"community {cc} is empty; check the label source."
        pts = h0[idx]
        tang = manifold.logmap0(pts, c=c)
        u = tang.mean(dim=0)
        u = manifold.expmap0(u, c=c)
        centers.append(u)
    C = torch.stack(centers)
    assert torch.isfinite(C).all(), "non-finite community centers; check norms/curvature."
    return C
 
 
def main():
    ap = argparse.ArgumentParser(description="Retrain the diffusion with true SBM community guidance.")
    ap.add_argument("--dataset", default="sbm")
    ap.add_argument("--frozen", default="logs/lptask/sbm/frozen_original",
                    help="source frozen dir (encoder + embeddings + config)")
    ap.add_argument("--out", default="logs/lptask/sbm/frozen_truecomm",
                    help="new frozen dir to write (kept separate from --frozen)")
    ap.add_argument("--labels", default="data/sbm/sbm_labels.pt",
                    help="ground-truth community labels (torch .pt of shape [n])")
    ap.add_argument("--epochs", type=int, default=None,
                    help="diffusion epochs (default: diff_epoc from config, =50)")
    ap.add_argument("--lr", type=float, default=0.001, help="matches hyperdiff (Adam 0.001)")
    ap.add_argument("--seed", type=int, default=1432, help="matches config seed")
    ap.add_argument("--device", default="cpu")
    args_cli = ap.parse_args()
 
    device = args_cli.device
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
 
    args = build_args(args_cli.frozen, device)
    epochs = args_cli.epochs if args_cli.epochs is not None else int(getattr(args, "diff_epoc", 50))
 
    # --- reuse the SAME adj_train_norm the diffusion originally trained on ---
    data = load_data(args, os.path.join(os.environ["DATAPATH"], args.dataset))
    adj = data["adj_train_norm"].to(device)
 
    # --- reuse the SAME frozen encoder embeddings (h0) ---
    emb = np.load(os.path.join(args_cli.frozen, "embeddings.npy"))
    h0 = torch.from_numpy(emb).to(device)            # float64, as in hyperdiff
    n, dim = h0.shape
    # keep every node inside the c=1.0 ball the diffusion assumes (fixes 2 boundary nodes)
    h0, n_clamped = clamp_to_ball(h0, c=1.0)
    if n_clamped:
        print(f"[fix] clamped {n_clamped}/{n} boundary nodes (||x||>=1) inside the c=1.0 ball")
 
    # --- CORRECT community guidance from ground-truth blocks ---
    true = torch.load(args_cli.labels)
    labels = (true.numpy() if hasattr(true, "numpy") else np.asarray(true)).astype(np.int64)
    assert labels.shape[0] == n, f"labels {labels.shape} vs {n} nodes"
    k = int(labels.max()) + 1
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"[fix] true communities: k={k}  distribution={dict(zip(uniq.tolist(), cnt.tolist()))}")
    old = np.load(os.path.join(args_cli.frozen, "label.npy"))
    ou, oc = np.unique(old, return_counts=True)
    print(f"[fix] (replacing collapsed labeling {dict(zip(ou.tolist(), oc.tolist()))})")
 
    diff = diffusion(args, dim=dim).to(device)
    center = community_centers(h0, labels, diff.manifold, k).to(device)
    # sanity: centers must be distinct (collapsed guidance had ~identical centers)
    with torch.no_grad():
        cd = torch.cdist(center.float(), center.float())
        print(f"[fix] between-center distances:\n{cd.cpu().numpy().round(3)}")
 
    # --- retrain (identical to hyperdiff except the guidance source) ---
    optimizer = optim.Adam(diff.parameters(), lr=args_cli.lr)
    diff.train()
    print(f"[fix] retraining diffusion: {epochs} epochs, lr={args_cli.lr}, restrict=True, dim={dim}")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = diff(h0, adj, labels, center, restrict=True, graphset=False)
        if not torch.isfinite(loss):
            raise SystemExit(
                f"[fix] loss is {loss.item()} at epoch {epoch + 1} — aborting so no broken "
                "model is written. Centers were finite, so this is in the training pass; "
                "rerun and report this line."
            )
        loss.backward()
        optimizer.step()
        if epoch < 3 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1:4d}/{epochs}  loss={loss.item():.6f}")
 
    # --- write the new frozen dir (leave frozen_original untouched) ---
    os.makedirs(args_cli.out, exist_ok=True)
    for fn in ("model.pth", "config.json", "embeddings.npy"):
        shutil.copy(os.path.join(args_cli.frozen, fn), os.path.join(args_cli.out, fn))
    np.save(os.path.join(args_cli.out, "label.npy"), labels)
    np.save(os.path.join(args_cli.out, "center.npy"), center.detach().cpu().numpy())
    diff.eval()
    torch.save(diff.state_dict(), os.path.join(args_cli.out, "sample.pt"))
    print(f"[fix] wrote new frozen dir -> {args_cli.out}")
    print(f"[fix] next: python t_star_sweep_score.py --p-values 30 "
          f"--t-stars 5 15 30 60 100 150 199 --frozen {args_cli.out}")
 
 
if __name__ == "__main__":
    main()