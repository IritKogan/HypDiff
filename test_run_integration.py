"""Integration test for run_stability_score.py WITHOUT the real model.

Validates the wiring around recover.py: filename parsing, (method, p) grouping,
the probs cache write, the --from-cache round-trip and settings guard, and
end-to-end integration with stability_score.  Uses fabricated perturbed .pt
files and a deterministic fake recover_one, so it runs anywhere torch is present.
"""

import os
import sys
import types
import tempfile

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                                  # tests at repo root
sys.path.insert(0, os.path.dirname(_here))                 # tests in tests/ subdir
import run_stability_score as R          # noqa: E402
from stability_score import stability_score  # noqa: E402


def _make_perturbed(dirpath, method, pp, seed, e_removed):
    """Write a perturbed .pt with the exact keys perturbation.save_batch uses."""
    removed = [(i, i + 1) for i in range(e_removed)]
    d = {
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "removed_edges": removed,
        "kept_edges": [],
        "num_nodes": 40,
        "method": method, "p": pp / 100.0, "seed": seed,
    }
    path = os.path.join(dirpath, f"{method}_p{pp:02d}_seed{seed}.pt")
    torch.save(d, path)
    return path


def make_fake_recover(prob_value):
    """recover_one stand-in: every removed edge gets `prob_value` in all K recons."""
    def _recover_one(edge_index, removed_edges, num_nodes, frozen_dir,
                     t_star, K, device="cpu"):
        e = len(removed_edges)
        probs = [torch.full((e,), float(prob_value)) for _ in range(K)]
        recons = [torch.zeros(num_nodes, 8) for _ in range(K)]
        return recons, probs
    return _recover_one


def _args(perturbed_dir, cache_dir, **over):
    ns = types.SimpleNamespace(
        dataset="sbm", perturbed_dir=perturbed_dir,
        frozen="unused", t_star=15, K=5, tau=0.5, ddof=0, device="cpu",
        methods=["random", "targeted"], p_values=[10, 20],
        limit=0, out=os.path.join(cache_dir, "scores"),
        cache_dir=cache_dir, use_cache=True, from_cache=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_parse_and_discover():
    with tempfile.TemporaryDirectory() as d:
        for method in ("random", "targeted"):
            for pp in (10, 20):
                for seed in range(3):
                    _make_perturbed(d, method, pp, seed, e_removed=4)
        # a decoy file that must be ignored
        torch.save({}, os.path.join(d, "notes_backup.pt"))

        groups = R.discover(d, {"random", "targeted"}, {10, 20})
        assert set(groups.keys()) == {("random", 10), ("random", 20),
                                      ("targeted", 10), ("targeted", 20)}
        for files in groups.values():
            assert len(files) == 3
            # sorted by seed
            seeds = [R.parse_name(f)[2] for f in files]
            assert seeds == [0, 1, 2]

        # subset filtering
        only = R.discover(d, {"random"}, {10})
        assert list(only.keys()) == [("random", 10)]


def test_recover_cache_and_from_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "cache")
        path = _make_perturbed(d, "random", 10, 0, e_removed=6)
        args = _args(d, cache)

        fake = make_fake_recover(0.9)
        probs, removed, meta = R.load_or_recover(path, args, fake)
        assert len(probs) == args.K
        assert probs[0].shape[0] == 6
        assert meta["method"] == "random" and meta["num_removed_edges"] == 6
        # cache file was written
        cp = R.cache_path(cache, path)
        assert os.path.exists(cp)

        # from-cache: no recover_one needed, identical probs
        args_fc = _args(d, cache, from_cache=True)
        probs2, removed2, meta2 = R.load_or_recover(path, args_fc, None)
        assert torch.allclose(torch.stack(probs), torch.stack(probs2))
        assert removed2 == removed

        # settings guard: changing K must be refused against an old cache
        args_badK = _args(d, cache, from_cache=True, K=7)
        try:
            R.load_or_recover(path, args_badK, None)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on K mismatch vs cache")


def test_end_to_end_scores_all_reappear():
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "cache")
        for seed in range(10):                       # N = 10 damaged copies
            _make_perturbed(d, "random", 10, seed, e_removed=5)
        args = _args(d, cache, p_values=[10], methods=["random"])

        fake = make_fake_recover(0.9)                # >= tau -> everything reappears
        groups = R.discover(d, {"random"}, {10})
        (key, files) = next(iter(groups.items()))
        graph_probs = [R.load_or_recover(p, args, fake)[0] for p in files]
        res = stability_score(graph_probs, tau=args.tau, ddof=args.ddof)
        assert res.n_graphs == 10
        assert abs(res.score - 1.0) < 1e-12          # all removed edges recovered
        assert abs(res.structural_var) < 1e-12       # identical across seeds
        assert abs(res.diffusion_var) < 1e-12        # identical across K


def test_end_to_end_scores_none_reappear():
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "cache")
        for seed in range(10):
            _make_perturbed(d, "targeted", 20, seed, e_removed=5)
        args = _args(d, cache, p_values=[20], methods=["targeted"])

        fake = make_fake_recover(0.1)                # < tau -> nothing reappears
        groups = R.discover(d, {"targeted"}, {20})
        (_, files) = next(iter(groups.items()))
        graph_probs = [R.load_or_recover(p, args, fake)[0] for p in files]
        res = stability_score(graph_probs, tau=args.tau, ddof=args.ddof)
        assert abs(res.score - 0.0) < 1e-12


def test_from_cache_tau_change_is_allowed():
    # Build cache at tau=0.5, then re-score at tau=0.95 from cache with NO recovery.
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "cache")
        files = [_make_perturbed(d, "random", 30, s, e_removed=4) for s in range(5)]
        args = _args(d, cache, p_values=[30])
        fake = make_fake_recover(0.9)
        for p in files:
            R.load_or_recover(p, args, fake)         # populate cache

        # re-score from cache only; tau changed, recover_one is None (never called)
        # cached prob is 0.9; at tau=0.95, 0.9 >= 0.95 is False -> nothing reappears.
        args_fc = _args(d, cache, p_values=[30], from_cache=True, tau=0.95)
        gp = [R.load_or_recover(p, args_fc, None)[0] for p in files]
        res_hi = stability_score(gp, tau=args_fc.tau)
        assert abs(res_hi.score - 0.0) < 1e-12
        # and at a tau below 0.9, the same cached probs now all reappear
        res_lo = stability_score(gp, tau=0.5)
        assert abs(res_lo.score - 1.0) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and isinstance(v, types.FunctionType)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:               # noqa: BLE001
            fails += 1
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
