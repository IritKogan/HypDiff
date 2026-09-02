"""Stage 3 - Probabilistic Stability Scoring.

Model-independent scoring module for the Network Stability project.

Consumes the raw output of Stage 2's ``recover.py``:

    recons, probs = recover_one(edge_index, removed_edges, num_nodes, ...)

where ``probs`` is a list of K tensors, each of shape ``[E_removed]`` giving the
Fermi-Dirac recovery probability of every removed edge in one reconstruction.
Stage 3 does NOT run the model - it takes those probabilities (for each of the N
damaged copies of a graph) and turns them into a single stability score.

The four-step score (project book / work plan)
------------------------------------------------
1. For each *removed edge*, how frequently it reappears across the K
   reconstructions.  "reappears" == recovery probability >= tau.
2. Average that per damaged graph (over its removed edges)  -> S_g
3. Average across all N damaged copies                      -> final score S
4. Keep structural-damage variance (across the N damaged copies) separate from
   diffusion variance (across the K reconstructions).

Variance separation
--------------------
Define the per-reconstruction reappearance rate for damaged graph g and
reconstruction k:

    r[g,k] = fraction of g's removed edges that reappear in reconstruction k

Then, with equal weight per graph:

    S_g          = mean_k  r[g,k]                 (per-graph score, step 2)
    score        = mean_g  S_g                    (final score, step 3)
    structural_var = Var_g( S_g )                 (between-graph: damage realization)
    diffusion_var  = mean_g Var_k( r[g,k] )       (within-graph: generative noise)

When every graph has the same number of reconstructions K, the law of total
variance makes these exact:

    structural_var + diffusion_var == Var over all (g,k) of r[g,k]

i.e. the total spread in the reappearance rate decomposes cleanly into the part
caused by *which* damaged copy we scored (structure) and the part caused by the
diffusion model's stochastic reconstruction (diffusion).  ``total_var`` below
exposes that pooled variance so the identity can be checked.

All variances are population variances (ddof=0) by default so a small example is
hand-checkable; pass ``ddof=1`` for the usual unbiased/sample estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_1d_array(x) -> np.ndarray:
    """Convert one reconstruction's probability vector to a 1-D float array.

    Accepts a python list, a numpy array, or any object exposing the usual
    torch-tensor interface (``.detach().cpu().numpy()``).  Kept dependency-light
    on purpose: the scorer never needs torch itself.
    """
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def _population_or_sample_var(values: np.ndarray, ddof: int) -> float:
    """Variance with the requested ddof; 0.0 when it is undefined (n <= ddof)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0 or n - ddof <= 0:
        return 0.0
    return float(np.var(values, ddof=ddof))


# --------------------------------------------------------------------------- #
# per-graph result
# --------------------------------------------------------------------------- #
@dataclass
class GraphScore:
    """Scoring result for ONE damaged graph (its K reconstructions)."""
    num_removed_edges: int
    num_reconstructions: int
    # per removed edge: fraction of the K reconstructions in which it reappears
    edge_reappearance: np.ndarray            # shape [E_removed]
    # per reconstruction: fraction of removed edges that reappear  (r[g, .])
    recon_rate: np.ndarray                   # shape [K]
    score: float                             # S_g == mean(edge_reappearance) == mean(recon_rate)
    diffusion_var: float                     # Var_k(recon_rate) for this graph
    empty: bool = False                      # True when the graph had no removed edges


def score_graph(probs: Sequence,
                tau: float = 0.5,
                ddof: int = 0) -> GraphScore:
    """Score one damaged graph from its per-reconstruction recovery probabilities.

    Parameters
    ----------
    probs : sequence of length K
        ``probs[k]`` is the ``[E_removed]`` vector of Fermi-Dirac recovery
        probabilities for reconstruction k.  Every entry must have the same
        length E_removed (the removed-edge set is fixed for a given damaged
        graph).  ``recover_one`` returns exactly this list.
    tau : float
        Reappearance threshold; an edge reappears in a reconstruction when its
        recovery probability is >= tau.
    ddof : int
        Delta-degrees-of-freedom for this graph's diffusion variance.

    Returns
    -------
    GraphScore
    """
    K = len(probs)
    if K == 0:
        raise ValueError("score_graph: got 0 reconstructions for a graph.")

    mats = [_to_1d_array(p) for p in probs]
    E = mats[0].size
    for i, m in enumerate(mats):
        if m.size != E:
            raise ValueError(
                f"score_graph: reconstruction {i} has {m.size} removed-edge "
                f"probabilities but reconstruction 0 has {E}; the removed-edge "
                "set must be identical across reconstructions of one graph."
            )

    if E == 0:
        # Damaged graph with no removed edges: score is undefined. Report it as
        # empty so the aggregator can drop it instead of poisoning the mean.
        return GraphScore(
            num_removed_edges=0,
            num_reconstructions=K,
            edge_reappearance=np.zeros(0),
            recon_rate=np.full(K, np.nan),
            score=float("nan"),
            diffusion_var=float("nan"),
            empty=True,
        )

    # reappearance indicator matrix: [K, E], 1.0 where prob >= tau
    hits = np.vstack([(m >= tau).astype(float) for m in mats])   # [K, E]

    edge_reappearance = hits.mean(axis=0)     # [E]  step 1 (per removed edge)
    recon_rate = hits.mean(axis=1)            # [K]  fraction reappearing per recon
    score = float(edge_reappearance.mean())   # S_g  (step 2)  == recon_rate.mean()
    diffusion_var = _population_or_sample_var(recon_rate, ddof)

    return GraphScore(
        num_removed_edges=E,
        num_reconstructions=K,
        edge_reappearance=edge_reappearance,
        recon_rate=recon_rate,
        score=score,
        diffusion_var=diffusion_var,
    )


# --------------------------------------------------------------------------- #
# aggregate (N damaged copies) result
# --------------------------------------------------------------------------- #
@dataclass
class StabilityScore:
    """Final Stage 3 result across the N damaged copies of a graph."""
    score: float                             # S = mean_g S_g   (step 3)
    structural_var: float                    # Var_g(S_g): sensitivity to damage realization
    diffusion_var: float                     # mean_g Var_k(r[g,k]): generative noise
    total_var: float                         # pooled Var over all (g,k) of r[g,k]
    n_graphs: int                            # N graphs actually scored (empties dropped)
    n_empty_dropped: int                     # damaged copies with no removed edges
    tau: float
    ddof: int
    per_graph: List[GraphScore] = field(default_factory=list)
    per_graph_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def structural_std(self) -> float:
        return float(np.sqrt(self.structural_var))

    @property
    def diffusion_std(self) -> float:
        return float(np.sqrt(self.diffusion_var))

    def summary(self) -> str:
        return (
            f"Stability score = {self.score:.4f}  "
            f"(structural sd {self.structural_std:.4f} over N={self.n_graphs}, "
            f"diffusion sd {self.diffusion_std:.4f}, tau={self.tau})"
        )


def stability_score(graph_probs: Sequence[Sequence],
                    tau: float = 0.5,
                    ddof: int = 0) -> StabilityScore:
    """Aggregate the four-step stability score over N damaged copies.

    Parameters
    ----------
    graph_probs : sequence of length N
        ``graph_probs[g]`` is the ``probs`` list (length K) returned by
        ``recover_one`` for damaged copy g.  So the full structure is
        N x K x [E_g], and E_g may differ from graph to graph (different damage
        levels remove different numbers of edges).
    tau : float
        Reappearance threshold (config default 0.5; Stage 4 calibrates it).
    ddof : int
        Delta-degrees-of-freedom for the reported variances.  0 (population) by
        default so small examples are exactly hand-checkable; 1 for the sample
        estimator when reporting uncertainty.

    Returns
    -------
    StabilityScore
    """
    if len(graph_probs) == 0:
        raise ValueError("stability_score: got 0 damaged graphs.")

    per_graph: List[GraphScore] = [score_graph(gp, tau=tau, ddof=ddof)
                                   for gp in graph_probs]

    scored = [g for g in per_graph if not g.empty]
    n_empty = len(per_graph) - len(scored)
    if len(scored) == 0:
        raise ValueError(
            "stability_score: every damaged graph had zero removed edges; "
            "nothing to score."
        )

    s_g = np.array([g.score for g in scored], dtype=float)          # S_g per graph
    score = float(s_g.mean())                                       # step 3
    structural_var = _population_or_sample_var(s_g, ddof)           # between-graph

    # within-graph (diffusion): average each graph's across-reconstruction var.
    diffusion_var = float(np.mean([g.diffusion_var for g in scored]))

    # pooled variance over every (g, k) reappearance rate; equals
    # structural_var + diffusion_var exactly when all graphs share the same K
    # and ddof == 0 (law of total variance).
    all_rates = np.concatenate([g.recon_rate for g in scored])
    total_var = _population_or_sample_var(all_rates, ddof)

    return StabilityScore(
        score=score,
        structural_var=structural_var,
        diffusion_var=diffusion_var,
        total_var=total_var,
        n_graphs=len(scored),
        n_empty_dropped=n_empty,
        tau=tau,
        ddof=ddof,
        per_graph=per_graph,
        per_graph_scores=s_g,
    )
