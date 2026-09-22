"""
Stage 4, step 3 — diffusion-diversity experiment (noise-scale sweep).
 
Problem: on frozen_truecomm the collapse is fixed and the score is calibrated, but
the K reconstructions are near-identical (thr_diff_sd = 0, raw_diffusion_std ~ 0),
so the "distribution over reconstructions" is a single point. The strong community
guidance that fixed the collapse also contracts the K samples together.
 
Lever: the SDEdit-style injected noise. p_sample_loop seeds the reverse loop with
q_sample(..., noise=None) at t*, which draws fresh Gaussian noise scaled by
sqrt(1-alphabar_{t*}). That single draw is the ONLY source of per-reconstruction
randomness (verified: the per-step reverse noise and the training noise are passed
explicitly; only the injection uses noise=None). We scale THAT draw by lambda and
sweep lambda, WITHOUT touching the model file: a runtime monkeypatch on q_sample
multiplies the noise only when it is None.
 
For each lambda we report, at fixed t*=60 and tau=0.825 (the locked settings):
  raw_diffusion_std : spread of the raw recovery signal across the K recons (want >0)
  thr_diffusion_std : the score's diffusion-variance term (want >0)
  raw_separation    : removed - null (must STAY high; if it collapses we injected too much)
  stability_score   : at tau=0.825
Goal: find the lambda where diffusion variance becomes real while separation holds.
 
Run from the repo root:
    python noise_scale_sweep.py --frozen logs/lptask/sbm/frozen_truecomm \
        --t-star 60 --tau 0.825 --lambdas 1 2 4 8 16
"""
 
from __future__ import annotations
 
import os
import csv
import json
import argparse
 
import numpy as np
import torch
 
from stability_score import stability_score, score_graph
from run_stability_score import discover, _safe_torch_load
from recover import _load_args, load_centers, damaged_to_inputs
from models.base_models import LPModel
from models.direction_diffusion import diffusion
from t_star_sweep_score import clamp_to_ball, sample_null, raw_metrics
 
 
def build_parser():
    ap = argparse.ArgumentParser(description="Sweep injected-noise scale to induce reconstruction diversity.")
    ap.add_argument("--dataset", default="sbm")
    ap.add_argument("--frozen", default="logs/lptask/sbm/frozen_truecomm")
    ap.add_argument("--perturbed-dir", default=None)
    ap.add_argument("--t-star", dest="t_star", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.825)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--methods", nargs="*", default=["random", "targeted"])
    ap.add_argument("--p-values", dest="p_values", nargs="*", type=int, default=[30],
                    help="damage levels to test (default: p30 only, for speed)")
    ap.add_argument("--lambdas", nargs="*", type=float, default=[1, 2, 4, 8, 16],
                    help="injected-noise scale factors to sweep")
    ap.add_argument("--ddof", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-clamp", dest="no_clamp", action="store_true")
    ap.add_argument("--out", default="stage4_noise_scale")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap
 
 
def install_noise_scale(diff, scale_holder):
    """Monkeypatch diff.q_sample so a None-noise draw is multiplied by scale_holder[0].
 
    Only the t* injection passes noise=None, so only it is scaled; the reverse-step
    and training calls pass explicit noise and are untouched."""
    orig = diff.q_sample
 
    def scaled_q_sample(x_start, t, direction, noise, labels, u0, restrict=False):
        if noise is None:
            noise = torch.randn_like(x_start) * float(scale_holder[0])
        return orig(x_start, t, direction, noise, labels, u0, restrict=restrict)
 
    diff.q_sample = scaled_q_sample
    return orig
 
 
def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
 
    # encode all files once (t*/lambda-independent)
    enc = {}
    diff = None
    print(f"[noise] frozen={args.frozen}  t*={args.t_star}  tau={args.tau}  K={args.K}  "
          f"lambdas={args.lambdas}")
    done = 0
    for (method, pp), files in groups.items():
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
            enc[path] = {"h0": h0, "adj_norm": adj_norm, "idx": idx, "null_idx": null_idx,
                         "method": method, "p": pp}
            done += 1
    print(f"[noise] encoded {done} files")
 
    scale_holder = [1.0]
    install_noise_scale(diff, scale_holder)
 
    rows = []
    for lam in args.lambdas:
        scale_holder[0] = lam
        torch.manual_seed(args.seed)   # same noise sequence across lambda, only scaled
        for (method, pp), files in groups.items():
            graph_probs, graph_null = [], []
            for path in files:
                e = enc[path]
                probs_k, null_k = [], []
                for _ in range(args.K):
                    with torch.no_grad():
                        sa = diff.p_sample_loop(e["h0"].clone().to(device), e["adj_norm"],
                                                label, center, t_star=args.t_star)
                        probs_k.append(model.decode(sa, e["idx"]).cpu())
                        if e["null_idx"].shape[0] > 0:
                            null_k.append(model.decode(sa, e["null_idx"]).cpu())
                graph_probs.append(probs_k); graph_null.append(null_k)
            res = stability_score(graph_probs, tau=args.tau, ddof=args.ddof)
            rr, rn, rdsd = raw_metrics(graph_probs, graph_null, args.ddof)
            rows.append({
                "lambda": lam, "method": method, "p_percent": pp,
                "stability_score": res.score, "thr_diffusion_std": res.diffusion_std,
                "structural_std": res.structural_std,
                "raw_removed_mean": rr, "raw_null_mean": rn, "raw_separation": rr - rn,
                "raw_diffusion_std": rdsd,
            })
        blk = [r for r in rows if r["lambda"] == lam]
        print(f"lambda={lam:<5g} score={np.mean([r['stability_score'] for r in blk]):.4f}  "
              f"thr_diff_sd={np.mean([r['thr_diffusion_std'] for r in blk]):.4f}  "
              f"raw_diff_sd={np.mean([r['raw_diffusion_std'] for r in blk]):.4f}  "
              f"sep={np.mean([r['raw_separation'] for r in blk]):+.4f}")
 
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(args.out + ".json", "w") as f:
        json.dump({"settings": {k: v for k, v in vars(args).items()}, "rows": rows}, f, indent=2)
    print(f"\n[noise] wrote {args.out}.csv / .json")
 
    if args.plot:
        try:
            _plot(rows, args.lambdas, args.out + ".png", args.tau)
            print(f"[noise] wrote {args.out}.png")
        except Exception as ex:
            print(f"[noise] plot skipped: {ex}")
 
 
def _plot(rows, lambdas, png, tau):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def agg(key):
        return [np.mean([r[key] for r in rows if r["lambda"] == lam]) for lam in lambdas]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambdas, agg("raw_diffusion_std"), "o-", color="#1a7a3a", label="raw diffusion sd (diversity)")
    ax1.plot(lambdas, agg("thr_diffusion_std"), "s-", color="#2c6fbb", label=f"thr diffusion sd (tau={tau})")
    ax1.set_xlabel("injected-noise scale  λ"); ax1.set_ylabel("diffusion sd")
    ax1.set_xscale("log", base=2); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(lambdas, agg("raw_separation"), "^--", color="#c0392b", label="separation (removed-null)")
    ax2.plot(lambdas, agg("stability_score"), "v--", color="#8e44ad", alpha=0.6, label="stability score")
    ax2.set_ylabel("separation / score")
    ax1.set_title("Diffusion diversity vs injected-noise scale (want diffusion sd ↑ while separation holds)")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(png, dpi=140)
 
 
if __name__ == "__main__":
    main()