"""
Stage 4, step 2 — calibrate tau on the fixed (true-community) model.
 
Why: the full damage sweep at t*=60 on frozen_truecomm showed the stability score
saturated (~0.92-0.94 flat from 10% to 50% damage) at the placeholder tau=0.5,
because removed-edge probs (~0.80) clear 0.5 regardless of damage. This tool finds
a principled tau by running recovery ONCE, caching the removed + null decode probs,
then sweeping tau instantly. For each tau it reports, per (method, p):
 
  score(tau)   = stability score = fraction of removed edges recovered (TPR)
  nullrate(tau)= fraction of random non-edges that would pass tau (FPR)
  J(tau)       = score - nullrate  (Youden's J; separation of removed from null)
 
and, across damage, the sensitivity  score(p10) - score(p50)  (we WANT this
positive and sizable — a declining degradation curve).
 
It recommends tau by (a) max mean Youden's J and (b) max damage sensitivity, and
plots score-vs-damage at several tau plus J-vs-tau.
 
Run from the repo root:
 
    # first time: runs recovery at t*=60 over the full grid, caches, sweeps tau
    python tau_calibrate.py --frozen logs/lptask/sbm/frozen_truecomm --t-star 60
 
    # re-sweep tau instantly from the cache (no recovery re-run)
    python tau_calibrate.py --frozen logs/lptask/sbm/frozen_truecomm --t-star 60 --from-cache
"""
 
from __future__ import annotations
 
import os
import csv
import json
import argparse
 
import numpy as np
import torch
 
from stability_score import stability_score
from run_stability_score import discover, _safe_torch_load
from recover import _load_args, load_centers, damaged_to_inputs
from models.base_models import LPModel
from models.direction_diffusion import diffusion
# reuse the exact helpers the sweep uses, so recovery is identical
from t_star_sweep_score import clamp_to_ball, sample_null
 
 
def build_parser():
    ap = argparse.ArgumentParser(description="Calibrate tau on cached recovery probs.")
    ap.add_argument("--dataset", default="sbm")
    ap.add_argument("--frozen", default="logs/lptask/sbm/frozen_truecomm")
    ap.add_argument("--perturbed-dir", default=None)
    ap.add_argument("--t-star", dest="t_star", type=int, default=60)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--methods", nargs="*", default=["random", "targeted"])
    ap.add_argument("--p-values", dest="p_values", nargs="*", type=int,
                    default=[10, 20, 30, 40, 50])
    ap.add_argument("--taus", nargs="*", type=float, default=None,
                    help="tau grid (default: 0.30..0.85 step 0.025)")
    ap.add_argument("--ddof", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-clamp", dest="no_clamp", action="store_true")
    ap.add_argument("--cache", default=None, help="probs cache path (default: tau_cache_<frozen>_t<t*>.pt)")
    ap.add_argument("--from-cache", dest="from_cache", action="store_true")
    ap.add_argument("--out", default="stage4_tau_calib")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap
 
 
def build_cache(args):
    """Run recovery once at t*, return {cond: {'removed': [ [K,E] per graph ],
    'null': [ [K,E] per graph ]}} as numpy arrays."""
    device = args.device
    perturbed_dir = args.perturbed_dir or os.path.join("data", args.dataset, "perturbed")
    groups = discover(perturbed_dir, set(args.methods), set(args.p_values))
    if not groups:
        raise SystemExit(f"no perturbed files matched in {perturbed_dir}")
 
    targs = _load_args(args.frozen, device)
    label, center = load_centers(args.frozen, device)
    any_path = next(iter(groups.values()))[0]
    d0 = _safe_torch_load(any_path)
    n0 = int(d0.get("num_nodes", int(d0["edge_index"].max()) + 1))
    _, feat0 = damaged_to_inputs(d0["edge_index"], n0, targs)
    targs.n_nodes, targs.feat_dim = feat0.shape
    targs.nb_false_edges = 1
    targs.nb_edges = 1
    model = LPModel(targs).to(device)
    model.load_state_dict(torch.load(os.path.join(args.frozen, "model.pth"), map_location=device))
    model.eval()
    diff = None
 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[tau] running recovery once at t*={args.t_star}, K={args.K} to cache probs...")
    cache = {}
    done = 0
    for (method, pp), files in groups.items():
        rem_graphs, null_graphs = [], []
        for path in files:
            d = _safe_torch_load(path)
            ei = d["edge_index"]
            n = int(d.get("num_nodes", int(ei.max()) + 1))
            adj_norm, features = damaged_to_inputs(ei, n, targs)
            adj_norm = adj_norm.to(device); features = features.to(device)
            with torch.no_grad():
                _, _, h0, _ = model.encode(features, adj_norm)
            if not args.no_clamp:
                h0, _ = clamp_to_ball(h0, c=1.0)
            if diff is None:
                diff = diffusion(targs, dim=h0.shape[1]).to(device)
                diff.load_state_dict(torch.load(os.path.join(args.frozen, "sample.pt"),
                                                map_location=device))
                diff.eval()
            idx = torch.as_tensor(d["removed_edges"], dtype=torch.long, device=device)
            if idx.dim() == 1:
                idx = idx.view(-1, 2)
            rng = np.random.default_rng(args.seed + done)
            null_idx = sample_null(ei, idx, n, 200, rng).to(device)
            rem_k, null_k = [], []
            for _ in range(args.K):
                with torch.no_grad():
                    sa = diff.p_sample_loop(h0.clone().to(device), adj_norm, label, center,
                                            t_star=args.t_star)
                    rem_k.append(model.decode(sa, idx).cpu().numpy())
                    null_k.append(model.decode(sa, null_idx).cpu().numpy())
            rem_graphs.append(np.vstack(rem_k))     # [K, E_rm]
            null_graphs.append(np.vstack(null_k))   # [K, E_null]
            done += 1
        cache[f"{method}_p{pp}"] = {"method": method, "p": pp,
                                    "removed": rem_graphs, "null": null_graphs}
        print(f"  cached {method:8s} p{pp:02d}  ({done} files)")
    return cache
 
 
def score_removed(rem_graphs, tau, ddof):
    """== stability_score on the cached removed probs (fraction recovered)."""
    graph_probs = [[g[k] for k in range(g.shape[0])] for g in rem_graphs]
    return stability_score(graph_probs, tau=tau, ddof=ddof).score
 
 
def null_pass_rate(null_graphs, tau):
    """FPR: fraction of random non-edges that pass tau (mean over K, then graphs)."""
    rates = []
    for g in null_graphs:                 # g: [K, E_null]
        if g.size == 0:
            continue
        rates.append(float((g >= tau).mean()))
    return float(np.mean(rates)) if rates else float("nan")
 
 
def main():
    args = build_parser().parse_args()
    cache_path = args.cache or f"tau_cache_{os.path.basename(args.frozen)}_t{args.t_star}.pt"
 
    if args.from_cache and os.path.exists(cache_path):
        print(f"[tau] loading cached probs from {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
    else:
        cache = build_cache(args)
        torch.save(cache, cache_path)
        print(f"[tau] wrote probs cache -> {cache_path}")
 
    taus = args.taus or list(np.round(np.arange(0.30, 0.851, 0.025), 3))
    rows = []
    for tau in taus:
        for cond, blob in cache.items():
            sc = score_removed(blob["removed"], tau, args.ddof)
            fpr = null_pass_rate(blob["null"], tau)
            rows.append({"tau": float(tau), "method": blob["method"], "p_percent": blob["p"],
                         "score": sc, "null_pass_rate": fpr, "youden_J": sc - fpr})
 
    # ---- summarise: mean J per tau, and damage sensitivity per tau -------- #
    def by_tau(tau):
        return [r for r in rows if abs(r["tau"] - tau) < 1e-9]
 
    print("\n tau  | meanJ  | mean score | mean FPR | sens(random p10-p50) | sens(targeted p10-p50)")
    best = None
    for tau in taus:
        b = by_tau(tau)
        meanJ = np.mean([r["youden_J"] for r in b])
        msc = np.mean([r["score"] for r in b])
        mfpr = np.mean([r["null_pass_rate"] for r in b])
        def sens(method):
            s10 = next((r["score"] for r in b if r["method"] == method and r["p_percent"] == 10), np.nan)
            s50 = next((r["score"] for r in b if r["method"] == method and r["p_percent"] == 50), np.nan)
            return s10 - s50
        sr, st = sens("random"), sens("targeted")
        print(f" {tau:.3f} | {meanJ:.3f} | {msc:.3f}      | {mfpr:.3f}    | {sr:+.3f}                | {st:+.3f}")
        if best is None or meanJ > best[1]:
            best = (tau, meanJ, msc, mfpr)
 
    print(f"\n[tau] max mean Youden's J at tau={best[0]:.3f} (J={best[1]:.3f}, "
          f"score={best[2]:.3f}, FPR={best[3]:.3f})")
 
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(args.out + ".json", "w") as f:
        json.dump({"settings": {k: v for k, v in vars(args).items()},
                   "recommended_tau_maxJ": best[0], "rows": rows}, f, indent=2)
    print(f"[tau] wrote {args.out}.csv / .json")
 
    if args.plot:
        try:
            _plot(rows, taus, args.out + ".png")
            print(f"[tau] wrote {args.out}.png")
        except Exception as ex:
            print(f"[tau] plot skipped: {ex}")
 
 
def _plot(rows, taus, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = sorted({r["p_percent"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: score vs damage at a few taus
    pick = [taus[0], taus[len(taus)//3], taus[2*len(taus)//3], taus[-1]]
    for tau in pick:
        for method in ("random", "targeted"):
            ys = [next((r["score"] for r in rows
                        if abs(r["tau"]-tau) < 1e-9 and r["method"] == method and r["p_percent"] == p), np.nan)
                  for p in ps]
            ls = "-" if method == "random" else "--"
            ax1.plot(ps, ys, ls, marker="o", label=f"tau={tau:.2f} {method}")
    ax1.set_xlabel("damage %"); ax1.set_ylabel("stability score")
    ax1.set_title("Damage curve at different tau (want it to DECLINE)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7, ncol=2)
    # right: mean J vs tau
    mJ = [np.mean([r["youden_J"] for r in rows if abs(r["tau"]-tau) < 1e-9]) for tau in taus]
    mS = [np.mean([r["score"] for r in rows if abs(r["tau"]-tau) < 1e-9]) for tau in taus]
    mF = [np.mean([r["null_pass_rate"] for r in rows if abs(r["tau"]-tau) < 1e-9]) for tau in taus]
    ax2.plot(taus, mJ, "o-", label="Youden J (score-FPR)")
    ax2.plot(taus, mS, "o--", alpha=0.6, label="mean score (TPR)")
    ax2.plot(taus, mF, "x--", alpha=0.6, label="mean FPR (null)")
    ax2.axvline(taus[int(np.argmax(mJ))], color="green", ls=":", label=f"max J @ {taus[int(np.argmax(mJ))]:.2f}")
    ax2.set_xlabel("tau"); ax2.set_ylabel("rate"); ax2.set_title("Separation vs tau")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=140)
 
 
if __name__ == "__main__":
    main()