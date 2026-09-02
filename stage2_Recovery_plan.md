# Stage 2 — Adapt HypDiff into a "Guided Recovery" Engine (step-by-step)

**Goal:** turn HypDiff's _generator_ into a _repairer_. Take one damaged graph from
Stage 1 and return **K** reconstructed versions of it, by (a) encoding the damaged graph
with the already-trained HGCN encoder, (b) noising its latent only **partway** — up to a
truncation step **t\*** — instead of to full noise, and (c) running the existing reverse
loop **K** times using community centers computed **once from the original, undamaged
graph**. This is the core adaptation and the most delicate stage: nothing new is invented,
but three existing pieces of HypDiff are re-wired.

**Status:** CODE COMPLETE (all 7 steps built, tested, committed on branch
`stage2-guided-recovery`). Baseline provably preserved (`t_star=None` reproduces Stage 0
ROC/AP). Recovery engine returns K distinct reconstructions with valid decode
probabilities. **Empirical validation on BA surfaced a method-level finding — see
"Validation findings on BA" at the end — pointing to SBM as the correct next validation
graph.**

Work in the same folder Stage 0 / Stage 1 ran in — the repo root
(`C:\Users\iritk\Desktop\Phase B Code\HypDiff`), where `main.py`, `diff.py`,
`perturbation.py` and `logs/lptask/ba/` already exist.

We stay entirely on the **lptask** pipeline (the hyperbolic one). `graphtask` /
`diff_graphset.py` / `direction_diffusionset.py` are the VAE path — we do **not** touch
them. Everything below is about `diff.py` + `models/direction_diffusion.py` +
`models/base_models.py`.

---

## First, the real flow (confirmed by reading the code)

`main.py --taskselect lptask --type train` runs two steps in order:

1. `lp_train.train(args)` — the **encoder**. `LPModel.encode(features, adj_train_norm)`
   produces the embeddings `h0`, saved to `logs/lptask/<ds>/embeddings.npy`; the encoder
   weights (including the Fermi-Dirac decoder `dc` with `r,t`) are saved to `model.pth`.
2. `diff.hyperdiff(args)` — the **diffusion**. It calls `Community_cluster(args)` (writes
   `center.npy` + `label.npy`), loads `embeddings.npy`, trains the `diffusion` model,
   samples with `sa = diff.p_sample_loop(h0, adj, label, center)`, then scores link
   reconstruction with `LPModel.compute_metrics(...)` (ROC/AP) and saves `result.pt`.

Two frozen assets come out of Stage 0, both trained on the **original** graph, and Stage 2
**reuses them without retraining**: the encoder `model.pth`, and the diffusion weights
`sample.pt`. Stage 2 never trains — it only _samples_ (`.eval()` + `torch.no_grad()`).

**One correction to the work-plan wording, found in the code.** The plan's Stage 2(c)
says we "swap `generate()`'s fixed 0.5 for τ." On the lptask path, `generate()` (diff.py,
the `0.5` at line ~348) is **never called** — it belongs to the graphset/graphtask path.
On our path the per-edge recovery probability comes from `LPModel.decode(sa, idx)`
(base_models.py) which applies Fermi-Dirac to the hyperbolic distance between the two
endpoints. So **τ is the threshold on those decode-probabilities**, and it is applied in
**Stage 3 scoring**, not inside `generate()`. Stage 2's real deliverable is the K
recovered embeddings (plus, optionally, the decode-probabilities for the removed edges) —
not a thresholded graph. Worth a sentence to the advisor so the project book and code
agree.

## What feeds this stage and what consumes it

- **Input (from Stage 1):** each damaged copy's `edge_index` (both-directions format) and
  its `removed_edges`, saved under `data/ba/perturbed/`. Plus the original graph's
  `features` (identity) and node count — unchanged, since Stage 1 never touches the node
  set.
- **Input (from Stage 0):** the frozen encoder `model.pth`, frozen diffusion `sample.pt`,
  and — this is the key reuse — the **original** graph's `center.npy` + `label.npy`.
- **Output (for Stage 3):** for each damaged copy, **K** reconstructed embeddings
  (and/or the K Fermi-Dirac recovery probabilities for exactly that copy's `removed_edges`).
  Stage 3 turns those into the stability score.

---

## The three modifications, mapped to exact code

### (a) Encode the damaged graph — **reuse `LPModel.encode`, no change**

`LPModel.encode(x, adj)` (base_models.py, ~line 78) already does exactly what we need. The
only change is _what we feed it_: build `adj_train_norm` and `features` from the **damaged**
`edge_index` (Stage 1 output) instead of the full graph, load the **already-trained**
encoder (`model.pth`), put it in `.eval()`, and run `encode` to get `h0_damaged`. We do
**not** re-train the encoder per damaged graph — that would defeat the purpose. Node
features stay the identity matrix Stage 0 used.

> **Risk (geometric instability, from §4):** at heavy damage the encoder can push the
> hyperbolic coordinates around unpredictably. Nothing to "fix" in code — just log the
> norm/range of `h0_damaged` so we can see it during calibration.

### (b) Stop the noise at t\*, not at full noise — **modify `p_sample_loop`**

This is the heart of the stage, in `diffusion.p_sample_loop` (direction_diffusion.py,
~line 293). Today it always starts from full noise and always runs all 200 steps:

```python
img = torch.randn(shape, device=device)      # ~line 297  -> pure noise
...
img = img + h2                                # ~line 309  -> noise + normalized latent
imgs = [img]
...
for t in reversed(range(0, self.num_timesteps)):   # ~line 316 -> full 200-step schedule
    img, x_start = self.p_sample(img, h2, adj, direction, lables, u0, t, self_cond, graphset)
```

The function that adds noise up to a _chosen_ step, `q_sample`, **already exists** in the
same class (~line 198) and is already used by `forward()` and `p_sample`. We just call it
from the loop's start. Add a `t_star=None` argument to `p_sample_loop` and change only the
initialization + the loop bound:

```python
def p_sample_loop(self, x, adj, lables, u0, return_all_timesteps=False,
                  graphset=False, t_star=None):     # <-- new arg
    ...
    # h2 is built exactly as before (logmap into tangent space + normalization)
    if t_star is None:
        # --- unchanged baseline behaviour: full noise, full schedule ---
        img = torch.randn(shape, device=device)
        img = self.tran_direction(direction, img, lables)
        img = img + h2
        start_t = self.num_timesteps
    else:
        # --- Phase B: partial (SDEdit-style) start at t* ---
        t_batched = torch.full((batch,), t_star, device=device, dtype=torch.long)
        img = self.q_sample(x_start=h2, t=t_batched, direction=direction,
                            noise=None, labels=lables, u0=u0, restrict=True)
        start_t = t_star
    imgs = [img]
    ...
    for t in reversed(range(0, start_t)):    # <-- t_star instead of num_timesteps
        img, x_start = self.p_sample(img, h2, adj, direction, lables, u0, t, self_cond, graphset)
```

The `restrict=True` path is what keeps the anisotropic (direction/community) constraint —
we keep it on. When `t_star is None` the method behaves **exactly** as before, so the
Stage 0 baseline and its ROC/AP numbers still reproduce unchanged. That "default = old
behaviour" is deliberate: it's how we prove the edit didn't break anything.

> **Risk (choosing t\*, from §4):** `t_star` near 200 ≈ the original generator — it erases
> the surviving structure. `t_star` too small leaves no room to denoise the damage away.
> t\* is a Stage 4 calibration knob; Stage 2 just has to _expose_ it, not pick it.

### (c) Denoise K times, reusing the original graph's centers — **wrap, don't rewrite**

Two separate changes here.

**Fix the centers to the original graph.** In `hyperdiff` the centers come from
`Community_cluster(args)` (diff.py ~line 77) → `center.npy` / `label.npy`, computed on
whatever `embeddings.npy` currently holds. The project book requires these be computed
**once from the undamaged graph** and reused for every damaged copy. In practice: **do not
call `Community_cluster` inside the recovery loop.** Load the original graph's saved
`center.npy` + `label.npy` and pass them straight into `p_sample_loop`. Because Stage 1
never changes the node set, the per-node `label` array and the `center` tensor line up with
the damaged graph with no remapping.

**Loop K times.** `p_sample_loop` starts from a random `torch.randn`, so each call is
stochastic → **K** calls give **K** genuinely different reconstructions. Wrap it:

```python
recons = []
for k in range(K):
    h0_k = h0_damaged.clone()      # IMPORTANT: see the mutation note below
    with torch.no_grad():
        sa_k = diff.p_sample_loop(h0_k, adj_damaged, label_orig, center_orig,
                                  t_star=t_star)
    recons.append(sa_k)            # sa_k = one recovered embedding set
```

> **Mutation gotcha (found in the code):** `p_sample_loop` overwrites its input in place —
> `h[i] = self.manifold.logmap(...)` at ~line 306, where `h = x`. Calling it K times on the
> **same** tensor would feed the 2nd..Kth calls an already-logmapped `h0`. Always pass a
> fresh `h0_damaged.clone()` each iteration (as above). This is the single easiest bug to
> introduce in this stage.

**Where τ actually lives.** For each removed edge `(u,v)` in this copy's `removed_edges`,
the recovery probability is `LPModel.decode(sa_k, removed_edges)` → Fermi-Dirac on the
hyperbolic distance. Stage 2 can either hand Stage 3 the raw `sa_k` embeddings, or
pre-compute these decode-probabilities. **τ (the "did it come back?" cutoff) is applied in
Stage 3**, not here.

---

## Step-by-step

**Step 1 — Back up the two files you'll edit.** Stage 0 already left `.bak` copies
(`direction_diffusion.py.bak`, `base_models.py.bak`, `diff.py.bak`). Confirm they're
current before editing, so the baseline is always recoverable.

**Step 2 — Add `t_star` to `p_sample_loop`.** Make only the (b) edit above in
`models/direction_diffusion.py`: new `t_star=None` argument, branch the initialization,
and bound the loop with `start_t`. Leave the `t_star is None` branch byte-for-byte
equivalent to today.

**Step 3 — Prove the baseline still runs.** Re-run Stage 0's smoke test
(`python stage0_smoke_test.py`, or `python main.py --taskselect lptask --type train
--dataset ba --cuda -1`). Because `t_star` defaults to `None`, ROC/AP should match the
Stage 0 numbers (≈0.57 / ≈0.62). If they don't, the edit changed behaviour — stop and
diff against `.bak`.

**Step 4 — Save the original graph's centers as reusable assets.** Run the original graph
once through the encoder + `Community_cluster` (this already happens in a normal Stage 0
run) and keep `logs/lptask/ba/center.npy` and `label.npy`. These are the frozen
"community centers from the undamaged graph" every damaged copy will reuse. Do **not**
regenerate them per damaged copy.

**Step 5 — Write the recovery driver `recover.py`** (new file, repo root). It does, for
one damaged copy:

```python
# pseudocode-level outline
import torch, numpy as np, os
from config import parser
from models.base_models import LPModel
from models.direction_diffusion import diffusion

def recover_one(args, damaged_edge_index, features, num_nodes,
                t_star, K, save_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # (a) encode the DAMAGED graph with the FROZEN encoder
    adj_damaged = build_adj_train_norm(damaged_edge_index, num_nodes)   # same normalize as Stage 0
    model = LPModel(args); model.load_state_dict(torch.load(os.path.join(save_dir,'model.pth')))
    model.eval()
    with torch.no_grad():
        h0_damaged, _, _, _ = model.encode(features, adj_damaged)

    # frozen diffusion + original-graph centers
    diff = diffusion(args, dim=h0_damaged.shape[1])
    diff.load_state_dict(torch.load(os.path.join(save_dir,'sample.pt'))); diff.eval()
    label_orig  = np.load(os.path.join(save_dir,'label.npy'))
    center_orig = torch.tensor(np.load(os.path.join(save_dir,'center.npy'))).to(device)

    # (b)+(c) K partial-start reconstructions
    recons = []
    for k in range(K):
        with torch.no_grad():
            sa_k = diff.p_sample_loop(h0_damaged.clone().to(device), adj_damaged.to(device),
                                      label_orig, center_orig, t_star=t_star)
        recons.append(sa_k)
    return model, recons            # model kept so Stage 3 can call model.decode(sa_k, removed_edges)
```

Reuse Stage 0's exact adjacency-normalisation for `build_adj_train_norm` (pull it from
`utils/data_utils.load_data` so the damaged graph is normalised identically to how the
encoder was trained).

**Step 6 — Smoke-test recovery on one Stage 1 copy.** Load one file from
`data/ba/perturbed/` (e.g. random, 20%), run `recover_one(..., t_star=100, K=5)`, and
sanity-check: `len(recons)==K`, each `sa_k` has shape `[num_nodes, dim]`, the K copies are
**not** identical (they should differ — proves the K-loop and stochastic start work), and
`model.decode(sa_k, removed_edges)` returns a probability in (0,1) per removed edge. This is
the Stage 2 deliverable check.

**Step 7 — Expose the new knobs in `config.py`.** Add to a config group (e.g.
`work_type_config`): `t_star` (truncation step), `K` (reconstructions per damaged graph),
and `tau` (edge threshold, used by Stage 3). `N` already belongs to Stage 1. Keep the
existing Fermi-Dirac `r`, `t` where they are. Document defaults so Stage 4 can sweep them.

---

## Decisions to flag to the advisor

1. **The `generate()`/0.5 vs. τ correction above.** On the lptask path τ is a threshold on
   `LPModel.decode` probabilities in Stage 3, not the `0.5` inside `generate()`. The code
   and the project book should be reconciled on this point.
2. **t\* is exposed here, chosen in Stage 4.** Stage 2 makes t\* a parameter with a
   safe default (baseline preserved at `t_star=None`); the actual value is a calibration
   result, not a Stage 2 decision.
3. **Everything frozen.** Encoder (`model.pth`) and diffusion (`sample.pt`) are trained
   once on the original graph in Stage 0 and reused read-only. Stage 2 never trains.

## Things to expect

- **`t_star=None` must stay a perfect no-op.** It's the guarantee that the baseline is
  untouched; treat any drift in the Stage 0 ROC/AP as a regression.
- **Clone `h0` every K iteration** — `p_sample_loop` mutates its input in place.
- **Isolated nodes** (from heavy Stage 1 damage) flow through `encode` fine; just watch
  `h0_damaged`'s range for the geometric-instability risk.
- **Centers are per-node and node-count-stable**, so the original graph's `label`/`center`
  drop straight onto damaged copies with no remapping — this is only true because Stage 1
  preserves the node set.

## Deliverables

- `models/direction_diffusion.py` — `p_sample_loop` extended with `t_star` (default `None`
  = unchanged baseline).
- `recover.py` — the recovery driver: damaged `edge_index` → K reconstructed embeddings,
  reusing frozen encoder + diffusion + original-graph centers.
- `config.py` — new `t_star`, `K`, `tau` parameters.
- A one-copy smoke test confirming K distinct reconstructions and valid per-removed-edge
  decode probabilities — the handoff to Stage 3.

---

## Validation findings on BA (empirical, after build)

The engine was validated on a Stage 1 damaged copy (BA, random 10% removal, 38 removed
edges), using a properly trained frozen model (200 encoder / 50 diffusion epochs). The code
is correct; the finding is about the **method on BA**, and it is a genuine result worth
recording for the advisor.

**1. The signal lives in the encoder, and diffusion degrades it on BA.**
Decoding edge probabilities (Fermi-Dirac, decode at `c=0.6`) on three edge groups:

| readout                         | surviving | removed | random | removed − random |
| ------------------------------- | --------- | ------- | ------ | ---------------- |
| raw encoder `h0` (no diffusion) | 0.769     | 0.623   | 0.550  | **+0.073**       |
| recovered `sa`, `t*=2`          | 0.688     | 0.607   | 0.584  | +0.023           |
| recovered `sa`, `t*=5`          | 0.757     | 0.748   | 0.782  | **−0.035**       |
| recovered `sa`, `t*=10–40`      | ~0.75     | ~0.75   | ~0.80  | ≈ −0.04          |
| recovered `sa`, `t*≥80`         | ~0        | ~0      | ~0     | ≈ 0              |

The raw encoder ranks _surviving > removed > random_ at usable magnitudes — exactly the
expected shape (removed edges are real edges the encoder saw disconnected). But **no `t*`
improves on it**: `t*=2` preserves a weaker version of the ranking, `t*≥5` inverts it
(random scores above removed), and `t*≥80` collapses everything. This is the project
book's §4 risk "_t\* too large discards the surviving structure_" observed directly — and
here even small `t*` does not add value.

**2. Root cause — BA has no community structure.** `Community_cluster` (hardcoded
`n_clusters=3`) assigned all 100 nodes to a single cluster, so the anisotropic
community/angular guidance has nothing to steer with. Corroborated by the trained model's
link-reconstruction ROC ≈ 0.48 on the original graph (at/below random). The method is
hub- and community-guided by design; BA supplies neither.

**3. Not a scale/curvature artifact.** Decoding `sa` at the matching curvature `c=1.0`
gives a degenerate constant `sqdist≈1241` (all probs 0), so the curvature mismatch is not
the lever. The problem is that diffusion moves the points, not how we measure them.

### Consequence

- **Stage 2 code is complete and correct** — this is a data/method observation, not a bug.
- **BA is the wrong validation graph** for this method. **SBM** (planted communities) is the
  correct next test, and Stage 5 needs it regardless. If SBM shows `removed ≫ random` at a
  clean `t*`, the method and this engine are jointly validated; if it does not, escalate the
  diffusion-quality question (ROC < 0.5 on the original graph) to the advisor.

### For the advisor

- On BA, guided recovery does not beat the raw encoder; the best link signal is `h0` with no
  diffusion, and larger `t*` harms it. Likely because BA lacks the community structure the
  method exploits.
- Recommend validating on SBM before drawing any conclusion about the framework, and
  separately reviewing why the diffusion's own link ROC on BA is ≈ 0.48.
