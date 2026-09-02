"""Unit tests for the Stage 3 scoring module.

Everything here is checked against a mock computed by hand (see the module
docstring / the block comments below) so the arithmetic is independent of the
model.  Run with:  pytest -q     or:    python tests/test_stability_score.py
"""

import math
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                                  # tests at repo root
sys.path.insert(0, os.path.dirname(_here))                 # tests in tests/ subdir
from stability_score import (            # noqa: E402
    score_graph,
    stability_score,
    GraphScore,
    StabilityScore,
)

HIT = 0.8      # any value >= tau (0.5)
MISS = 0.2     # any value <  tau
TAU = 0.5


# --------------------------------------------------------------------------- #
# The hand-checked mock
# --------------------------------------------------------------------------- #
# Graph A, E=2, K=3.  hit pattern:
#   recon0 [1,1]  recon1 [1,0]  recon2 [0,0]
#   recon_rate = [1.0, 0.5, 0.0]   -> S_A = 0.5
#   edge_reappearance = [2/3, 1/3]
#   diffusion_var_A (pop) = Var([1,0.5,0]) = 1/6
GRAPH_A = [
    [HIT,  HIT],
    [HIT,  MISS],
    [MISS, MISS],
]

# Graph B, E=2, K=3.  hit pattern:
#   recon0 [1,1]  recon1 [1,1]  recon2 [1,0]
#   recon_rate = [1.0, 1.0, 0.5]   -> S_B = 5/6
#   edge_reappearance = [1, 2/3]
#   diffusion_var_B (pop) = Var([1,1,0.5]) = 1/18
GRAPH_B = [
    [HIT, HIT],
    [HIT, HIT],
    [HIT, MISS],
]

# Aggregate over [A, B] (equal K, ddof=0):
#   score          = mean(0.5, 5/6)       = 2/3
#   structural_var = Var([0.5, 5/6])       = 1/36
#   diffusion_var  = mean(1/6, 1/18)       = 1/9
#   total_var      = Var(all 6 rates)      = 5/36 = 1/36 + 4/36  (law of total variance)


def approx(a, b, tol=1e-12):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# per-graph
# --------------------------------------------------------------------------- #
def test_graph_a_by_hand():
    g = score_graph(GRAPH_A, tau=TAU)
    assert isinstance(g, GraphScore)
    assert g.num_removed_edges == 2
    assert g.num_reconstructions == 3
    assert np.allclose(g.edge_reappearance, [2 / 3, 1 / 3])
    assert np.allclose(g.recon_rate, [1.0, 0.5, 0.0])
    assert approx(g.score, 0.5)
    assert approx(g.diffusion_var, 1 / 6)
    assert not g.empty


def test_graph_b_by_hand():
    g = score_graph(GRAPH_B, tau=TAU)
    assert np.allclose(g.edge_reappearance, [1.0, 2 / 3])
    assert np.allclose(g.recon_rate, [1.0, 1.0, 0.5])
    assert approx(g.score, 5 / 6)
    assert approx(g.diffusion_var, 1 / 18)


def test_score_equals_two_ways():
    # S_g == mean(edge_reappearance) == mean(recon_rate)
    for graph in (GRAPH_A, GRAPH_B):
        g = score_graph(graph, tau=TAU)
        assert approx(g.score, float(g.edge_reappearance.mean()))
        assert approx(g.score, float(g.recon_rate.mean()))


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def test_aggregate_by_hand():
    res = stability_score([GRAPH_A, GRAPH_B], tau=TAU)
    assert isinstance(res, StabilityScore)
    assert res.n_graphs == 2
    assert res.n_empty_dropped == 0
    assert approx(res.score, 2 / 3)
    assert approx(res.structural_var, 1 / 36)
    assert approx(res.diffusion_var, 1 / 9)
    assert approx(res.total_var, 5 / 36)


def test_law_of_total_variance_identity():
    # With equal K per graph and ddof=0, between + within == total exactly.
    res = stability_score([GRAPH_A, GRAPH_B], tau=TAU, ddof=0)
    assert approx(res.structural_var + res.diffusion_var, res.total_var)


def test_std_properties():
    res = stability_score([GRAPH_A, GRAPH_B], tau=TAU)
    assert approx(res.structural_std, math.sqrt(1 / 36))
    assert approx(res.diffusion_std, math.sqrt(1 / 9))


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_tau_boundary_is_inclusive():
    # prob exactly == tau counts as a reappearance (>= tau).
    g = score_graph([[TAU, TAU]], tau=TAU)
    assert np.allclose(g.edge_reappearance, [1.0, 1.0])
    # just below tau does not
    eps = 1e-9
    g2 = score_graph([[TAU - eps, TAU - eps]], tau=TAU)
    assert np.allclose(g2.edge_reappearance, [0.0, 0.0])


def test_single_reconstruction_zero_diffusion_var():
    g = score_graph([[HIT, MISS, HIT]], tau=TAU)
    assert g.num_reconstructions == 1
    assert approx(g.diffusion_var, 0.0)      # only one recon -> no spread
    assert approx(g.score, 2 / 3)


def test_empty_removed_edge_set_is_dropped():
    # A damaged copy with no removed edges is flagged empty and excluded.
    empty_graph = [[], [], []]               # K=3 reconstructions, E=0
    g = score_graph(empty_graph, tau=TAU)
    assert g.empty
    assert math.isnan(g.score)

    res = stability_score([GRAPH_A, empty_graph, GRAPH_B], tau=TAU)
    assert res.n_graphs == 2                 # empty one dropped
    assert res.n_empty_dropped == 1
    assert approx(res.score, 2 / 3)          # same as the two-graph aggregate


def test_all_empty_raises():
    try:
        stability_score([[[], []], [[], []]], tau=TAU)
    except ValueError:
        return
    raise AssertionError("expected ValueError when every graph is empty")


def test_perfect_and_zero_recovery():
    perfect = score_graph([[HIT, HIT], [HIT, HIT]], tau=TAU)
    assert approx(perfect.score, 1.0)
    assert approx(perfect.diffusion_var, 0.0)
    none = score_graph([[MISS, MISS], [MISS, MISS]], tau=TAU)
    assert approx(none.score, 0.0)
    assert approx(none.diffusion_var, 0.0)


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
class _FakeTensor:
    """Mimics the torch.Tensor interface the scorer duck-types on."""
    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


def test_accepts_numpy_and_tensor_like_inputs():
    as_list = score_graph(GRAPH_A, tau=TAU)
    as_np = score_graph([np.asarray(r) for r in GRAPH_A], tau=TAU)
    as_tensor = score_graph([_FakeTensor(r) for r in GRAPH_A], tau=TAU)
    for g in (as_np, as_tensor):
        assert np.allclose(g.edge_reappearance, as_list.edge_reappearance)
        assert approx(g.score, as_list.score)


def test_mismatched_edge_count_raises():
    try:
        score_graph([[HIT, HIT], [HIT]], tau=TAU)   # E=2 then E=1
    except ValueError:
        return
    raise AssertionError("expected ValueError on inconsistent removed-edge count")


def test_zero_reconstructions_raises():
    try:
        score_graph([], tau=TAU)
    except ValueError:
        return
    raise AssertionError("expected ValueError on 0 reconstructions")


def test_ddof_sample_variance():
    # ddof=1 sample variance of S_g over the two graphs:
    #   values [0.5, 5/6], sample var = ((0.5-2/3)^2 + (5/6-2/3)^2) / (2-1)
    #                                 = (1/36 + 1/36) / 1 = 1/18
    res = stability_score([GRAPH_A, GRAPH_B], tau=TAU, ddof=1)
    assert approx(res.structural_var, 1 / 18)


if __name__ == "__main__":
    import types
    mod = sys.modules[__name__]
    tests = [getattr(mod, n) for n in dir(mod)
             if n.startswith("test_") and isinstance(getattr(mod, n), types.FunctionType)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:            # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
