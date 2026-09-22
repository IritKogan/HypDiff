"""
Stage 4 probe: stability score AND diffusion variance vs the truncation step t*.
 
This is the direct test of the Stage 3 finding that at t*=15 the diffusion
variance is ~0 (the K reconstructions are near-identical, so recovery is carried
by the frozen encoder, not the generative process). The mechanism, confirmed in
models/direction_diffusion.py:
 
    p_sample_loop(t_star=int) starts the reverse loop from step t* after
    q_sample(..., noise=None), which draws FRESH Gaussian noise per call scaled
    by sqrt(1 - alphas_cumprod[t*]). At small t*, alphas_cumprod ~ 1 so that
    factor ~ 0: the K starts are almost identical -> diffusion variance ~ 0.
    As t* -> num_timesteps-1 (=199), more noise is injected -> diffusion
    variance should rise. Whether the *score* stays meaningful as it does is the
    question Stage 4 must answer honestly.
 
Design (extends the Stage 2 t_star_sweep.py load-once pattern; reuses the tested
Stage 3 scorer and grouping instead of re-deriving them):
  * Load the frozen encoder + diffusion ONCE (num_timesteps = 200).
  * Encode every perturbed file ONCE (h0 is t*-independent) and cache it.
  * For each t* and each (method, p) group of N=10 seeds, run K reconstructions,
    decode the removed edges, and hand the per-graph prob lists to the tested
    stability_score() -> score, structural_var, diffusion_var, total_var.
  * Write a tidy CSV/JSON (one row per t*, method, p) and a 2-panel figure:
    diffusion_std vs t* (is the generator alive?) and score vs t* (does recovery
    stay meaningful?).
 
Numerically identical to running run_stability_score.py --t-star X per t* (same
weights, same scorer); it just avoids reloading the model per file. As a built-in
check, running with the default grid reproduces the Stage 3 numbers at t*=15.
 
Run from the HypDiff repo root, e.g.:
 
    # fast decisive probe: the diffusion-variance curve on one damage level
    python t_star_sweep_score.py --p-values 30 --t-stars 5 15 30 60 100 150 199
 
    # full grid (both attacks, all damage levels)
    python t_star_sweep_score.py
 
    # smoke: 2 seeds per condition, two t*, to check it runs end-to-end
    python t_star_sweep_score.py --limit-seeds 2 --t-stars 15 100
"""
 
from __future__ import annotations
 
import os
import csv
import json
import time
import argparse
 
import numpy as np
import torch
 
# tested Stage 3 pieces - imported, not reimplemented
from stability_score import stability_score
from run_stability_score import discover, _safe_torch_load
 
# tested Stage 2 recovery internals
from recover import _load_args, load_centers, damaged_to_inputs
from models.base_models import LPModel
from models.direction_diffusion import diffusion
 
 
def build_parser():
    ap = argparse.ArgumentParser(description="Stage 4 t* sweep: score + diffusion variance vs t*.")
    ap.add_argument("--dataset", default="sbm")
    ap.add_argument("--perturbed-dir", default=None, help="default: data/<dataset>/perturbed")
    ap.add_argument("--frozen", default=None, help="default: logs/lptask/<dataset>/frozen_original")
    ap.add_argument("--t-stars", dest="t_stars", nargs="*", type=int,
                    default=[5, 15, 30, 60, 100, 150, 199],
                    help="truncation steps to sweep (each in [1, num_timesteps-1])")
    ap.add_argument("--methods", nargs="*", default=["random", "targeted"])
    ap.add_argument("--p-values", dest="p_values", nargs="*", type=int,
                    default=[10, 20, 30, 40, 50], help="damage levels as integer %%")
    ap.add_argument("--K", type=int, default=5, help="reconstructions per damaged graph")
    ap.add_argument("--null-size", dest="null_size", type=int, default=200,
                    help="random non-edges per graph for the baseline (0 disables)")
    ap.add_argument("--tau", type=float, default=0.5, help="reappearance threshold (calibrated separately)")
    ap.add_argument("--ddof", type=int, default=1,
                    help="1 = sample variance (default here, for uncertainty), 0 = population")
    ap.add_argument("--limit-seeds", dest="limit_seeds", type=int, default=0,
                    help="if >0, use only the first N seeds per (method,p) - fast probe")
    ap.add_argument("--seed", type=int, default=0, help="global RNG seed (reproducible sweep)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="stage4_tstar_sweep", help="output basename (.csv/.json/.png)")
    ap.add_argument("--no-plot", dest="plot", action="store_false", help="skip the PNG figure")
    ap.add_argument("--no-clamp", dest="no_clamp", action="store_true",
                    help="do NOT clamp encoded embeddings inside the c=1.0 ball "
                         "(clamp is on by default; it prevents artanh NaN during recovery)")
    return ap
 
 
def sample_null(edge_index, removed_idx, n, m, rng):
    """Sample up to m random non-edges (excluding surviving + removed edges).
 
    Gives a null baseline for the decode probabilities: recovery is only
    meaningful if removed-edge probs sit clearly ABOVE this null. Needed for tau
    calibration and to read the raw signal honestly. t*-independent, so sampled
    once per file during encoding."""
    surv = edge_index.t()
    surv = surv[surv[:, 0] < surv[:, 1]]
    edgeset = set(map(tuple, surv.tolist()))
    edgeset |= set(map(tuple, removed_idx.tolist()))
    out = []
    tries = 0
    while len(out) < m and tries < m * 50:
        a, b = sorted(int(x) for x in rng.integers(0, n, 2))
        tries += 1
        if a != b and (a, b) not in edgeset:
            out.append((a, b))
    return torch.tensor(out, dtype=torch.long) if out else torch.zeros((0, 2), dtype=torch.long)
 
 
def clamp_to_ball(h0, c=1.0, eps=1e-4):
    """Scale any node with ||x|| >= 1/sqrt(c) to just inside the Poincare ball.
 
    The diffusion's manifold ops are hardcoded to c=1.0 but the c=0.6 encoder can
    emit points with ||x|| >= 1. logmap0 / Mobius logmap at a community center then
    hit artanh(>=1) -> NaN during recovery. The retrainer clamps the same way, so
    training and sampling stay on the same (valid) ball. Returns (h0_clamped, n)."""
    bound = (1.0 / (c ** 0.5)) * (1.0 - eps)
    nrm = h0.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    scale = torch.clamp(bound / nrm, max=1.0)
    return h0 * scale, int((scale < 1.0).sum())
 
 
def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else float("nan")
 
 
def raw_metrics(graph_probs, graph_null, ddof):
    """Continuous (pre-threshold) analogues of the score's variance split.
 
    For graph g, reconstruction k: rate_raw[g,k] = mean removed-edge decode prob.
      raw_removed_mean = mean over (g,k) of rate_raw          (recovery strength)
      raw_null_mean    = mean over (g,k) of mean null-edge prob (baseline)
      raw_diffusion_std= sqrt( mean_g Var_k(rate_raw[g,k]) )  (generator's spread,
                          independent of tau: 0 only if the generator injects no
                          variance, NOT when the whole distribution drops below tau)
 
    NaN-robust: any non-finite decode prob (e.g. an off-manifold recovery) is
    dropped via nan-aware reductions rather than poisoning the whole condition.
    """
    per_g_rate_means, per_g_diff_var, null_means = [], [], []
    for probs_k, null_k in zip(graph_probs, graph_null):
        rates = np.array([_nanmean(np.asarray(p, dtype=float)) for p in probs_k])  # [K]
        per_g_rate_means.append(_nanmean(rates))
        finite = rates[np.isfinite(rates)]
        if finite.size - ddof > 0:
            per_g_diff_var.append(float(np.var(finite, ddof=ddof)))
        else:
            per_g_diff_var.append(np.nan)
        if null_k is not None and len(null_k) and np.asarray(null_k[0]).size:
            null_means.append(_nanmean([_nanmean(np.asarray(q, dtype=float)) for q in null_k]))
    raw_removed_mean = _nanmean(per_g_rate_means)
    raw_null_mean = _nanmean(null_means) if null_means else float("nan")
    raw_diffusion_std = float(np.sqrt(_nanmean(per_g_diff_var))) if per_g_diff_var else float("nan")
    return raw_removed_mean, raw_null_mean, raw_diffusion_std
 
 
def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
 
    device = args.device
    perturbed_dir = args.perturbed_dir or os.path.join("data", args.dataset, "perturbed")
    frozen = args.frozen or os.path.join("logs", "lptask", args.dataset, "frozen_original")
    assert os.path.isdir(frozen), f"missing frozen dir {frozen}"
 
    groups = discover(perturbed_dir, set(args.methods), set(args.p_values))
    if not groups:
        raise SystemExit(f"no perturbed files matched in {perturbed_dir}")
    if args.limit_seeds > 0:
        groups = {k: v[:args.limit_seeds] for k, v in groups.items()}
 
    # ---- load frozen model + diffusion ONCE ------------------------------- #
    targs = _load_args(frozen, device)
    label, center = load_centers(frozen, device)
 
    # peek one file to size the encoder (identity features -> feat_dim = n_nodes)
    any_path = next(iter(groups.values()))[0]
    d0 = _safe_torch_load(any_path)
    n0 = int(d0.get("num_nodes", int(d0["edge_index"].max()) + 1))
    adj0, feat0 = damaged_to_inputs(d0["edge_index"], n0, targs)
    targs.n_nodes, targs.feat_dim = feat0.shape
    targs.nb_false_edges = 1
    targs.nb_edges = 1
 
    model = LPModel(targs).to(device)
    model.load_state_dict(torch.load(os.path.join(frozen, "model.pth"), map_location=device))
    model.eval()
 
    # diffusion() needs the latent dim, which we read from the first encode below.
    diff = None
 
    # ---- encode every file ONCE (t*-independent) -------------------------- #
    print(f"[sweep] dataset={args.dataset}  t*={args.t_stars}  K={args.K}  tau={args.tau}  "
          f"ddof={args.ddof}  device={device}")
    print(f"[sweep] num_timesteps=200 (t* valid in [1,199]); encoding files once...")
    enc = {}   # path -> dict(h0, adj_norm, idx, meta)
    n_files = sum(len(v) for v in groups.values())
    done = 0
    total_clamped = 0
    for (method, pp), files in groups.items():
        for path in files:
            d = _safe_torch_load(path)
            ei = d["edge_index"]
            n = int(d.get("num_nodes", int(ei.max()) + 1))
            assert n == n0, (f"file {os.path.basename(path)} has {n} nodes but the encoder "
                             f"was built for {n0}; SBM copies should share node count.")
            adj_norm, features = damaged_to_inputs(ei, n, targs)
            adj_norm = adj_norm.to(device)
            features = features.to(device)
            with torch.no_grad():
                _, _, h0, _ = model.encode(features, adj_norm)
            if not args.no_clamp:
                h0, nclamp = clamp_to_ball(h0, c=1.0)
                total_clamped += nclamp
            if diff is None:
                diff = diffusion(targs, dim=h0.shape[1]).to(device)
                diff.load_state_dict(torch.load(os.path.join(frozen, "sample.pt"),
                                                map_location=device))
                diff.eval()
            idx = torch.as_tensor(d["removed_edges"], dtype=torch.long, device=device)
            if idx.dim() == 1:
                idx = idx.view(-1, 2)
            null_idx = None
            if args.null_size > 0:
                rng = np.random.default_rng(args.seed + done)
                null_idx = sample_null(ei, idx, n, args.null_size, rng).to(device)
            enc[path] = {
                "h0": h0, "adj_norm": adj_norm, "idx": idx, "null_idx": null_idx,
                "meta": {"method": method, "p": pp, "seed": d.get("seed"),
                         "num_removed_edges": int(idx.shape[0])},
            }
            done += 1
        print(f"  encoded {method:8s} p{pp:02d}  ({done}/{n_files} files)")
    if not args.no_clamp:
        print(f"[sweep] clamped {total_clamped} boundary node-embeddings inside the c=1.0 ball")
 
    # ---- sweep t* --------------------------------------------------------- #
    rows = []
    for t_star in args.t_stars:
        assert 1 <= t_star <= 199, f"t* must be in [1,199], got {t_star}"
        t0 = time.time()
        nan_recon = nan_prob_edges = tot_prob_edges = 0
        for (method, pp), files in groups.items():
            graph_probs = []
            graph_null = []
            removed_counts = []
            for path in files:
                e = enc[path]
                h0, adj_norm, idx, null_idx = e["h0"], e["adj_norm"], e["idx"], e["null_idx"]
                probs_k, null_k = [], []
                for _ in range(args.K):
                    with torch.no_grad():
                        # clone h0: p_sample_loop mutates its input in place
                        sa = diff.p_sample_loop(h0.clone().to(device), adj_norm,
                                                label, center, t_star=t_star)
                        nan_recon += int((~torch.isfinite(sa)).any().item())
                        pr = model.decode(sa, idx).cpu()
                        nan_prob_edges += int((~torch.isfinite(pr)).sum().item())
                        tot_prob_edges += int(pr.numel())
                        probs_k.append(pr)
                        if null_idx is not None and null_idx.shape[0] > 0:
                            null_k.append(model.decode(sa, null_idx).cpu())
                graph_probs.append(probs_k)
                graph_null.append(null_k)
                removed_counts.append(int(idx.shape[0]))
            res = stability_score(graph_probs, tau=args.tau, ddof=args.ddof)
            raw_removed_mean, raw_null_mean, raw_diffusion_std = raw_metrics(
                graph_probs, graph_null, args.ddof)
            rows.append({
                "t_star": t_star, "method": method, "p_percent": pp,
                "n_graphs": res.n_graphs, "K": args.K, "tau": args.tau, "ddof": args.ddof,
                "stability_score": res.score,
                "structural_var": res.structural_var, "structural_std": res.structural_std,
                "diffusion_var": res.diffusion_var, "diffusion_std": res.diffusion_std,
                "total_var": res.total_var,
                "raw_removed_mean": raw_removed_mean, "raw_null_mean": raw_null_mean,
                "raw_separation": raw_removed_mean - raw_null_mean,
                "raw_diffusion_std": raw_diffusion_std,
                "mean_removed_edges": float(sum(removed_counts) / len(removed_counts)),
            })
        dt = time.time() - t0
        # compact per-t* summary line (averaged over the conditions just scored)
        block = [r for r in rows if r["t_star"] == t_star]
        mscore = np.mean([r["stability_score"] for r in block])
        mdiff = np.mean([r["diffusion_std"] for r in block])
        mrawrem = np.mean([r["raw_removed_mean"] for r in block])
        mrawnull = np.mean([r["raw_null_mean"] for r in block])
        mrawdiff = np.mean([r["raw_diffusion_std"] for r in block])
        nanflag = ""
        if nan_prob_edges:
            nanflag = f"  [!NaN: {nan_recon} recon w/ NaN, {nan_prob_edges}/{tot_prob_edges} edge-probs NaN]"
        print(f"t*={t_star:<4d}  score={mscore:.4f}  thr_diff_sd={mdiff:.4f}  |  "
              f"raw removed={mrawrem:.4f} null={mrawnull:.4f} sep={mrawrem - mrawnull:+.4f} "
              f"raw_diff_sd={mrawdiff:.4f}  ({dt:.1f}s){nanflag}")
 
    # ---- write outputs ---------------------------------------------------- #
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out + ".json", "w") as f:
        json.dump({"settings": {k: v for k, v in vars(args).items()}, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}.csv and {args.out}.json  ({len(rows)} rows)")
 
    if args.plot:
        try:
            _plot(rows, args.out + ".png", args)
            print(f"wrote {args.out}.png")
        except Exception as ex:
            print(f"[plot skipped] {ex}")
 
 
def _plot(rows, png_path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    conds = sorted({(r["method"], r["p_percent"]) for r in rows})
    t_stars = sorted({r["t_star"] for r in rows})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
 
    for method, pp in conds:
        xs = [t for t in t_stars]
        d = {r["t_star"]: r for r in rows if r["method"] == method and r["p_percent"] == pp}
        ls = "-" if method == "random" else "--"
        ax1.plot(xs, [d[t]["raw_diffusion_std"] for t in xs], ls, marker="o", label=f"{method} p{pp}")
        ax2.plot(xs, [d[t]["raw_removed_mean"] for t in xs], ls, marker="o", label=f"{method} p{pp} removed")
        ax2.plot(xs, [d[t]["raw_null_mean"] for t in xs], ls, marker="x", alpha=0.5,
                 label=f"{method} p{pp} null")
        ax3.plot(xs, [d[t]["stability_score"] for t in xs], ls, marker="o", label=f"{method} p{pp}")
 
    ax1.set_ylabel("raw diffusion sd")
    ax1.set_title("Is the generator injecting variance? raw (pre-threshold) diffusion sd vs t*")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)
    ax2.set_ylabel("mean decode prob")
    ax2.set_title("Does recovery beat the null? removed-edge vs random-non-edge prob")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=6, ncol=2)
    ax3.set_ylabel(f"stability score (tau={rows[0]['tau']})")
    ax3.set_xlabel("t*  (truncation step; num_timesteps=200)")
    ax3.set_title("Does the thresholded score stay meaningful? score vs t*")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
 
 
if __name__ == "__main__":
    main()