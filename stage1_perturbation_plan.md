# Stage 1 — Graph Perturbation Module (step-by-step)

**Goal:** build the first _new_ piece of the pipeline (it does not exist in HypDiff).
Take the original graph and produce **N** damaged copies at a chosen damage level **p**
(e.g. 20%), using two attack methods — **Random Edge Removal** and **Targeted
(degree-based) Attack**. Only the adjacency changes; the node set and node features are
never touched. For every damaged copy, record exactly which edges were removed, because
Stage 3 (scoring) needs that as ground truth ("did the edge come back?").

**Status:** module written and unit-tested (7/7 tests pass). Ready to run on the BA graph.

Work in the same folder where Stage 0 ran — the HypDiff repo root
(`C:\Users\...\Phase B Code\HypDiff`), where `main.py`, `convert_ba.py` and `data/ba/`
already exist.

---

## What feeds this stage and what consumes it

- **Input:** the graph Stage 0 built — `data/ba/ba_edge_index.pt` (edge*index in \_both*
  directions). Stage 1 reads only this file and leaves `ba_features.pt` / `ba_labels.pt`
  completely untouched.
- **Output for Stage 2:** each damaged copy as an `edge_index` in the _same
  both-directions format_ Stage 0 used, so the HGCN encoder ingests it with no changes.
- **Output for Stage 3:** for each damaged copy, the exact list of _removed_ edges. If
  you don't save this now, you cannot score later.

## The two files (delivered)

- `perturbation.py` — the self-contained module.
- `test_perturbation.py` — the unit tests (the Stage 1 deliverable requirement).

Both go in the repo root, next to `main.py`.

## How the module works internally

1. **Fix the edge representation.** The stored `edge_index` lists each edge twice
   (u->v and v->u). The module first collapses it to an undirected set of canonical
   `(min, max)` pairs and drops self-loops. `E` = that count. All percentages are against `E`.
2. **Exact damage count:** `k = round(p * E)`. Deterministic — the tests assert the count
   is _exactly_ k, not "about 20%".
3. **Random Edge Removal:** seed a NumPy generator per copy, sample `k` edges uniformly
   without replacement, rebuild a symmetric `edge_index` from what survives.
4. **Targeted Attack:** score every edge by the product of its endpoint degrees
   (`deg(u) x deg(v)`) computed on the _original_ graph, then remove high-scoring edges.
   Default is **probabilistic** (sample k edges weighted by score) so the N copies are
   genuinely different and give real variance; a deterministic top-k mode also exists.
5. **Batch + record:** loops N times with `seed = base_seed + i`, returning each copy's
   perturbed `edge_index` (for Stage 2) and its `removed_edges` (for Stage 3).

---

## Step-by-step

**Step 1 — Put the two files in the repo.** Place `perturbation.py` and
`test_perturbation.py` in the HypDiff repo root (same folder as `main.py`).

**Step 2 — Confirm the Stage 0 data is present.** Check `data/ba/ba_edge_index.pt`
exists (created by Stage 0's `convert_ba.py`).

**Step 3 — Run the unit tests first.** This proves correctness before touching real data:

```
python test_perturbation.py
```

(If `pytest` is installed you can also use `python -m pytest test_perturbation.py -v`,
but it is NOT required — the file has a built-in runner. `pip install pytest` if you want
it for later stages.) Expected: 7 `PASS` lines, ending with `All 7 tests passed.`
The tests guarantee:

1. removes _exactly_ `round(p*E)` edges at every level 10-50%
2. node count never changes
3. only removes edges, never invents them (`kept U removed == original`)
4. output stays a proper both-directions, self-loop-free `edge_index`
5. same seed -> identical damage; different seed -> different damage
6. targeted attack really hits the highest degree-product edges

**Step 4 — Run it on the real BA graph.**

```
python perturbation.py
```

Loads `data/ba/ba_edge_index.pt`, prints the node/edge count, makes 10 random + 10
targeted copies at 20% damage, prints the removed-edge count per copy (should equal
`round(0.20*E)`), and saves them to `data/ba/perturbed/`.

**Step 5 — Inspect one saved file.** Each `.pt` in `data/ba/perturbed/` is a dict with:

- `edge_index` — the damaged graph, in Stage 0 format -> **feeds Stage 2's encoder**
- `removed_edges` — the deleted `(u,v)` pairs -> **ground truth for Stage 3**
- plus `kept_edges`, `num_nodes`, `method`, `p`, `seed`.

**Step 6 — Generate the full sweep you'll use later.** Create a file
`run_perturbation.py` in the repo root:

```python
import torch
from perturbation import perturb, save_batch

edge_index = torch.load("data/ba/ba_edge_index.pt")
num_nodes = int(edge_index.max()) + 1

for p in (0.1, 0.2, 0.3, 0.4, 0.5):
    for method in ("random", "targeted"):
        batch = perturb(edge_index, num_nodes, p=p, method=method, N=10, base_seed=0)
        save_batch(batch)
        print(f"saved {method} at {int(p*100)}% damage")
```

Run it:

```
python run_perturbation.py
```

That produces the 10%-50% damage sweep for both attack methods, N=10 copies each, all
saved under `data/ba/perturbed/`. The whole run is reproducible from `base_seed`.

---

## Decision to flag to the advisor

For the **targeted** attack the module defaults to `probabilistic_targeted=True`: it
samples the removed edges weighted toward high degree-product edges, so the N copies
differ and carry real variance (which the project book's "statistically independent
batches" language needs). Passing `probabilistic_targeted=False` makes it pure top-k —
deterministic, so all N copies are identical and N=1 is all you get. Recommended: keep the
probabilistic default. This is the one modeling choice in Stage 1 worth a sentence to
Dr. Avros / Prof. Volkovich.

## Things to expect

- **Isolated nodes are OK.** At high damage some nodes end up with no edges. The spec says
  the node set is never touched, so the module keeps them rather than dropping them. It
  matters for how Stage 2's encoder behaves under heavy damage — note it, don't "fix" it.
- **Determinism** of an entire run traces back to `base_seed` alone.

## Deliverables

- `perturbation.py` — the module (both methods, batching, edge-record output).
- `test_perturbation.py` — unit tests, all passing.
- `run_perturbation.py` — the sweep generator.
- Saved batches in `data/ba/perturbed/` to hand to Stage 2.
