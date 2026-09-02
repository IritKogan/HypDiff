"""Stage 3 runner - wire the scoring module to Stage 2's recover.py.

Runs the guided recovery engine over the SBM perturbed sweep, then scores it
with ``stability_score``.  Grouping: the N=10 seeds of one (method, p) condition
are the N damaged copies, so one stability score is produced per condition,
giving a stability-vs-damage curve for each attack type.

Because the diffusion recovery is the expensive step and tau / t_star are only
calibrated in Stage 4, this runner CACHES the raw Fermi-Dirac recovery
probabilities per file.  Re-scoring at a different tau then never re-runs the
model - use ``--from-cache``.

Run from the HypDiff repo root, e.g.:

    # smoke test: one file end-to-end (Stage 3 step "wire on one file")
    python run_stability_score.py --limit 1

    # full sweep (recover + cache + score)
    python run_stability_score.py

    # re-score everything at a new tau, no recovery re-run (Stage 4)
    python run_stability_score.py --from-cache --tau 0.6
"""

from __future__ import annotations

import os
import re
import csv
import json
import glob
import argparse
from collections import defaultdict

import torch

from stability_score import stability_score

# recover_one is only needed when we actually run recovery (not --from-cache),
# so import it lazily inside main() to keep --from-cache torch-only.

FNAME_RE = re.compile(r"^(?P<method>random|targeted)_p(?P<pp>\d{2})_seed(?P<seed>\d+)\.pt$")


def _safe_torch_load(path):
    """torch.load that works across torch versions.

    torch >= 2.6 defaults weights_only=True (rejects the python lists our .pt
    files carry); older torch has no weights_only kwarg at all.  Try the safe
    explicit call, fall back to the plain one.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------- #
def parse_name(path):
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return m.group("method"), int(m.group("pp")), int(m.group("seed"))


def discover(perturbed_dir, methods, p_values):
    """Return {(method, pp): [file paths sorted by seed]} for the requested subset."""
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(perturbed_dir, "*.pt")):
        parsed = parse_name(path)
        if parsed is None:
            continue
        method, pp, seed = parsed
        if methods and method not in methods:
            continue
        if p_values and pp not in p_values:
            continue
        groups[(method, pp)].append((seed, path))
    return {k: [p for _, p in sorted(v)] for k, v in sorted(groups.items())}


# --------------------------------------------------------------------------- #
def recover_probs_for_file(path, frozen_dir, t_star, K, device, recover_one):
    """Run recovery on one perturbed file -> (probs list of K [E_rm] tensors, meta)."""
    d = _safe_torch_load(path)
    edge_index = d["edge_index"]
    removed = d["removed_edges"]
    n = int(d.get("num_nodes", int(edge_index.max()) + 1))
    _, probs = recover_one(edge_index, removed, n, frozen_dir,
                           t_star=t_star, K=K, device=device)
    meta = {"method": d.get("method"), "p": d.get("p"), "seed": d.get("seed"),
            "num_nodes": n, "num_removed_edges": len(removed)}
    return probs, removed, meta


def cache_path(cache_dir, path):
    base = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(cache_dir, base + ".probs.pt")


def load_or_recover(path, args, recover_one):
    """Return (probs, removed, meta), using the on-disk probs cache when possible."""
    cp = cache_path(args.cache_dir, path)
    if args.from_cache or (args.use_cache and os.path.exists(cp)):
        if not os.path.exists(cp):
            raise FileNotFoundError(
                f"--from-cache but no cache for {os.path.basename(path)} ({cp}). "
                "Run once without --from-cache to build the cache."
            )
        blob = _safe_torch_load(cp)
        # sanity: cache must match the recovery settings we claim to score
        if blob.get("t_star") != args.t_star or blob.get("K") != args.K:
            raise ValueError(
                f"cache {os.path.basename(cp)} was built with "
                f"t_star={blob.get('t_star')}, K={blob.get('K')} but you asked for "
                f"t_star={args.t_star}, K={args.K}. tau is fine to change; "
                "t_star/K are not (they change the recovery). Rebuild the cache."
            )
        return blob["probs"], blob["removed_edges"], blob["meta"]

    if recover_one is None:
        raise RuntimeError("recover_one unavailable but recovery is required.")
    probs, removed, meta = recover_probs_for_file(
        path, args.frozen, args.t_star, args.K, args.device, recover_one)
    if args.use_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save({"probs": [p.cpu() for p in probs], "removed_edges": removed,
                    "meta": meta, "t_star": args.t_star, "K": args.K},
                   cache_path(args.cache_dir, path))
    return probs, removed, meta


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Stage 3 stability scoring runner.")
    ap.add_argument("--dataset", default="sbm")
    ap.add_argument("--perturbed-dir", default=None,
                    help="default: data/<dataset>/perturbed")
    ap.add_argument("--frozen", default=None,
                    help="default: logs/lptask/<dataset>/frozen_original")
    ap.add_argument("--t-star", dest="t_star", type=int, default=15)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--ddof", type=int, default=0,
                    help="0 = population variance (default), 1 = sample variance")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", nargs="*", default=["random", "targeted"])
    ap.add_argument("--p-values", dest="p_values", nargs="*", type=int,
                    default=[10, 20, 30, 40, 50], help="damage levels as integer %%")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, score only the first N files of the first group "
                         "(smoke test); prints per-file detail")
    ap.add_argument("--out", default="stage3_scores",
                    help="output basename; writes <out>.json and <out>.csv")
    ap.add_argument("--cache-dir", dest="cache_dir", default="stage3_probs_cache")
    ap.add_argument("--no-cache", dest="use_cache", action="store_false",
                    help="do not read or write the probs cache")
    ap.add_argument("--from-cache", dest="from_cache", action="store_true",
                    help="score only from cached probs; never run recovery")
    args = ap.parse_args()

    args.perturbed_dir = args.perturbed_dir or os.path.join("data", args.dataset, "perturbed")
    args.frozen = args.frozen or os.path.join("logs", "lptask", args.dataset, "frozen_original")

    recover_one = None
    if not args.from_cache:
        from recover import recover_one as _recover_one   # needs repo-root imports
        recover_one = _recover_one
        assert os.path.isdir(args.frozen), f"missing frozen dir {args.frozen}"

    groups = discover(args.perturbed_dir, set(args.methods), set(args.p_values))
    if not groups:
        raise SystemExit(f"no perturbed files matched in {args.perturbed_dir}")

    # ---- smoke mode: one (or N) file(s) from the first group ---------------- #
    if args.limit > 0:
        (method, pp), files = next(iter(groups.items()))
        files = files[:args.limit]
        print(f"[smoke] {method} p{pp}  x{len(files)} file(s)  "
              f"t*={args.t_star} K={args.K} tau={args.tau} dev={args.device}")
        graph_probs = []
        for path in files:
            probs, removed, meta = load_or_recover(path, args, recover_one)
            g = _score_one_debug(probs, removed, meta, args)
            graph_probs.append(probs)
        res = stability_score(graph_probs, tau=args.tau, ddof=args.ddof)
        print("[smoke] " + res.summary())
        print("[smoke] OK")
        return

    # ---- full sweep --------------------------------------------------------- #
    rows, detail = [], {}
    for (method, pp), files in groups.items():
        graph_probs = []
        removed_counts = []
        for path in files:
            probs, removed, meta = load_or_recover(path, args, recover_one)
            graph_probs.append(probs)
            removed_counts.append(len(removed))
        res = stability_score(graph_probs, tau=args.tau, ddof=args.ddof)
        row = {
            "method": method, "p_percent": pp, "n_graphs": res.n_graphs, "K": args.K,
            "t_star": args.t_star, "tau": args.tau,
            "stability_score": res.score,
            "structural_var": res.structural_var, "structural_std": res.structural_std,
            "diffusion_var": res.diffusion_var, "diffusion_std": res.diffusion_std,
            "total_var": res.total_var,
            "mean_removed_edges": float(sum(removed_counts) / len(removed_counts)),
        }
        rows.append(row)
        detail[f"{method}_p{pp}"] = {
            **row,
            "per_graph_scores": [float(s) for s in res.per_graph_scores],
        }
        print(f"{method:8s} p{pp:02d}  score={res.score:.4f}  "
              f"struct_sd={res.structural_std:.4f}  diff_sd={res.diffusion_std:.4f}  "
              f"(N={res.n_graphs}, mean removed={row['mean_removed_edges']:.1f})")

    with open(args.out + ".json", "w") as f:
        json.dump({"settings": vars(args), "results": detail}, f, indent=2)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}.json and {args.out}.csv  ({len(rows)} conditions)")


def _score_one_debug(probs, removed, meta, args):
    """Print per-file diagnostics in smoke mode; returns nothing important."""
    from stability_score import score_graph
    g = score_graph(probs, tau=args.tau, ddof=args.ddof)
    stacked = torch.stack([p.float() for p in probs])      # [K, E_rm]
    print(f"  {meta['method']} p={meta['p']} seed={meta['seed']}: "
          f"E_removed={g.num_removed_edges} K={g.num_reconstructions}  "
          f"prob[min={stacked.min():.3f} mean={stacked.mean():.3f} max={stacked.max():.3f}]  "
          f"S_g={g.score:.4f} diff_var={g.diffusion_var:.4f}")
    return g


if __name__ == "__main__":
    main()
