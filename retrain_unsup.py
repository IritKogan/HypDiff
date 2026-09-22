"""
Stage 4, Phase 2 — retrain the diffusion with UNSUPERVISED community detection.
 
Phase 1 (`retrain_truecomm.py`) proved the fix using the ground-truth SBM blocks.
This makes it stand on its own: recover the communities without labels, so the fix
generalizes to the real-world datasets (Stage 5) where no ground truth exists.
 
The broken component is diff.Community_cluster (HyperbolicKMeans n_clusters=3,
max_epochs=5), which collapses the 3 planted SBM blocks to one cluster (99/1,
ARI~0). We replace it with a robust detector and rebuild label.npy / center.npy,
then retrain exactly as Phase 1 (restrict=True, Adam 1e-3, same epochs) into a new
frozen dir. The ONLY change vs Phase 1 is the label source (unsupervised, not
ground truth), so comparing the two isolates "does an unsupervised clustering work
as well as the true blocks?".
 
Methods:
  louvain  (default) — modularity community detection on the GRAPH (python-louvain).
                       Auto-detects the number of communities; robust; the natural
                       choice for generalizing to arbitrary graphs.
  kmeans             — sklearn KMeans on the encoder embedding (repairs the paper's
                       own "cluster the hyperbolic embedding" idea; needs --n-clusters).
  spectral           — SpectralClustering on the adjacency.
 
It prints the detected cluster distribution and the ARI vs the true SBM blocks
(sanity only — the true labels are NOT used for training).
 
Run from the repo root:
    python retrain_unsup.py --method louvain
    python retrain_unsup.py --method kmeans --n-clusters 3
Then re-run the sweep against the new dir, e.g.:
    python t_star_sweep_score.py --t-stars 60 --tau 0.825 \
        --frozen logs/lptask/sbm/frozen_louvain --out stage4_louvain_damage
"""
 
from __future__ import annotations
 
import os
import json
import shutil
import argparse
 
import numpy as np
import torch
import torch.optim as optim
 
os.environ.setdefault("DATAPATH", "data/")
os.environ.setdefault("LOG_DIR", "logs/lptask/")
 
from utils.data_utils import load_data
from models.direction_diffusion import diffusion
# reuse the exact, validated center math + clamp from Phase 1
from retrain_truecomm import build_args, clamp_to_ball, community_centers
 
 
def detect_communities(method, edge_index, emb, n, n_clusters, seed):
    """Return integer labels [n] from an unsupervised method."""
    if method == "louvain":
        import networkx as nx
        try:
            import community as community_louvain          # python-louvain
        except ImportError as e:
            raise SystemExit("louvain needs python-louvain: pip install python-louvain") from e
        G = nx.Graph()
        G.add_nodes_from(range(n))
        ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
        G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        part = community_louvain.best_partition(G, random_state=seed)
        labels = np.array([part.get(i, 0) for i in range(n)], dtype=np.int64)
        # relabel to 0..k-1 contiguous
        _, labels = np.unique(labels, return_inverse=True)
        return labels.astype(np.int64)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(emb)
        return km.labels_.astype(np.int64)
    if method == "spectral":
        from sklearn.cluster import SpectralClustering
        import scipy.sparse as sp
        ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(float)
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                                assign_labels="discretize", random_state=seed)
        return sc.fit_predict(A).astype(np.int64)
    raise SystemExit(f"unknown --method {method}")
 
 
def main():
    ap = argparse.ArgumentParser(description="Retrain diffusion with unsupervised community detection.")
    ap.add_argument("--dataset", default="sbm")
    ap.add_argument("--frozen", default="logs/lptask/sbm/frozen_original",
                    help="source frozen dir (encoder + embeddings + config)")
    ap.add_argument("--method", default="louvain", choices=["louvain", "kmeans", "spectral"])
    ap.add_argument("--n-clusters", dest="n_clusters", type=int, default=3,
                    help="k for kmeans/spectral (louvain auto-detects)")
    ap.add_argument("--out", default=None, help="new frozen dir (default: logs/lptask/<ds>/frozen_<method>)")
    ap.add_argument("--true-labels", default="data/sbm/sbm_labels.pt",
                    help="ground-truth labels for the ARI sanity check ONLY (not used to train)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=1432)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
 
    device = a.device
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = a.out or f"logs/lptask/{a.dataset}/frozen_{a.method}"
    args = build_args(a.frozen, device)
    epochs = a.epochs if a.epochs is not None else int(getattr(args, "diff_epoc", 50))
 
    data = load_data(args, os.path.join(os.environ["DATAPATH"], args.dataset))
    adj = data["adj_train_norm"].to(device)
    edge_index = torch.load(os.path.join("data", a.dataset, f"{a.dataset}_edge_index.pt"))
 
    emb = np.load(os.path.join(a.frozen, "embeddings.npy"))
    h0 = torch.from_numpy(emb).to(device)
    n, dim = h0.shape
    h0, n_clamped = clamp_to_ball(h0, c=1.0)
    if n_clamped:
        print(f"[unsup] clamped {n_clamped}/{n} boundary nodes inside the c=1.0 ball")
 
    # ---- UNSUPERVISED community detection ---- #
    labels = detect_communities(a.method, edge_index, emb, n, a.n_clusters, a.seed)
    k = int(labels.max()) + 1
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"[unsup] method={a.method}  detected k={k}  distribution={dict(zip(uniq.tolist(), cnt.tolist()))}")
 
    # sanity: how well did it recover the true blocks? (labels NOT used for training)
    if os.path.exists(a.true_labels):
        try:
            from sklearn.metrics import adjusted_rand_score
            tl = torch.load(a.true_labels)
            tl = tl.numpy() if hasattr(tl, "numpy") else np.asarray(tl)
            print(f"[unsup] ARI vs true SBM blocks = {adjusted_rand_score(tl, labels):.4f}  "
                  f"(1.0 = perfect recovery; the old k-means was ~0)")
        except Exception as e:
            print(f"[unsup] ARI check skipped: {e}")
 
    diff = diffusion(args, dim=dim).to(device)
    center = community_centers(h0, labels, diff.manifold, k).to(device)
    with torch.no_grad():
        cd = torch.cdist(center.float(), center.float())
        print(f"[unsup] between-center distances:\n{cd.cpu().numpy().round(3)}")
 
    optimizer = optim.Adam(diff.parameters(), lr=a.lr)
    diff.train()
    print(f"[unsup] retraining diffusion: {epochs} epochs, lr={a.lr}, restrict=True, k={k}")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = diff(h0, adj, labels, center, restrict=True, graphset=False)
        if not torch.isfinite(loss):
            raise SystemExit(f"[unsup] non-finite loss at epoch {epoch+1}; aborting (no model written).")
        loss.backward(); optimizer.step()
        if epoch < 3 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={loss.item():.6f}")
 
    os.makedirs(out, exist_ok=True)
    for fn in ("model.pth", "config.json", "embeddings.npy"):
        shutil.copy(os.path.join(a.frozen, fn), os.path.join(out, fn))
    np.save(os.path.join(out, "label.npy"), labels)
    np.save(os.path.join(out, "center.npy"), center.detach().cpu().numpy())
    diff.eval()
    torch.save(diff.state_dict(), os.path.join(out, "sample.pt"))
    print(f"[unsup] wrote new frozen dir -> {out}")
    print(f"[unsup] next: python t_star_sweep_score.py --t-stars 60 --tau 0.825 "
          f"--frozen {out} --out stage4_{a.method}_damage")
 
 
if __name__ == "__main__":
    main()