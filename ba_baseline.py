import os, numpy as np, torch
os.environ.setdefault("DATAPATH", "data/")
from sklearn.metrics import roc_auc_score
from recover import _load_args, damaged_to_inputs
from models.base_models import LPModel
from t_star_sweep_score import sample_null
 
frozen = "logs/lptask/ba/frozen_original"
args = _load_args(frozen, "cpu")
args.n_nodes, args.feat_dim, args.nb_false_edges, args.nb_edges = 100, 100, 1, 1
model = LPModel(args)
model.load_state_dict(torch.load(os.path.join(frozen, "model.pth"), map_location="cpu"))
model.eval()
n = 100
 
for method in ["random", "targeted"]:
    for p in [10,20, 30,40, 50]:
        enc_auc, pa_auc = [], []
        for seed in range(10):
            d = torch.load(f"data/ba/perturbed/{method}_p{p}_seed{seed}.pt", weights_only=False)
            removed = torch.tensor(d["removed_edges"])
            null = sample_null(d["edge_index"], removed, n, 200, np.random.default_rng(seed))
            y = np.r_[np.ones(len(removed)), np.zeros(len(null))]
 
            # 1) encoder only (no diffusion)
            adj, feats = damaged_to_inputs(d["edge_index"], n, args)
            with torch.no_grad():
                _, _, h, _ = model.encode(feats, adj)
                s = np.r_[model.decode(h, removed).numpy(), model.decode(h, null).numpy()]
            enc_auc.append(roc_auc_score(y, s))
 
            # 2) preferential-attachment rule on the DAMAGED graph's degrees
            deg = torch.bincount(d["edge_index"][0], minlength=n).numpy()
            pa = lambda E: deg[E[:, 0]] * deg[E[:, 1]]
            pa_auc.append(roc_auc_score(y, np.r_[pa(removed.numpy()), pa(null.numpy())]))
        print(f"{method:8s} p{p}:  encoder AUC = {np.mean(enc_auc):.3f}   degree-rule AUC = {np.mean(pa_auc):.3f}")