# Stage 4 — Calibration + the diffusion-guidance fix (COMPLETE)

Stage 4 began as `t*`/`τ` calibration but uncovered — and fixed — the reason the diffusion was
contributing nothing, then generalised the fix without ground-truth labels. Working env: iritk
machine, `C:\Users\iritk\Desktop\Phase B Code\HypDiff`, CPU, SBM, N=10, K=5.

## STATUS: complete.

Locked, ground-truth model: `frozen_truecomm`, `t*=60`, `τ=0.825`.
Unsupervised model (self-contained): `frozen_louvain`, `t*=60`, `τ=0.85` (Louvain ARI 0.97).

## Change log (everything touched this session)

**New files added to the repo root (all additive — no existing file was edited):**

- `t_star_sweep_score.py` — sweeps `t*`, records stability score + raw recovery signal + null
  baseline + raw/thresholded diffusion variance; clamps embeddings to the c=1.0 ball; NaN diagnostics.
- `retrain_truecomm.py` — rebuild `label.npy`/`center.npy` from true SBM blocks, retrain diffusion.
- `tau_calibrate.py` — recover once, cache removed+null probs, sweep `τ` (score / FPR / Youden J /
  damage sensitivity).
- `noise_scale_sweep.py` — scale the injected SDEdit noise via a runtime monkeypatch on `q_sample`
  (no model-file edit) and sweep it.
- `retrain_unsup.py` — unsupervised community detection (louvain / kmeans / spectral) → retrain.
  Imports helpers from `retrain_truecomm.py` (both must sit in the repo root).

**New frozen model dirs (originals preserved):**

- `logs/lptask/sbm/frozen_truecomm` — diffusion retrained on true SBM communities.
- `logs/lptask/sbm/frozen_louvain` — diffusion retrained on Louvain communities.
- `logs/lptask/sbm/frozen_original` — UNCHANGED (baseline for comparison).

**NOT changed:** `direction_diffusion.py`, `diff.py`, `recover.py`, `config.py`, etc. In particular
`config.py stage2_config` still holds the old placeholders `t_star=15, tau=0.5` — the locked values
live only in this note (optionally update `t_star=60`; `τ` is per frozen model).

**Environment:** `python-louvain` (`import community`) was pip-installed on iritk for Phase 2.

**Key code fact established:** the diffusion's `num_timesteps = 200` (in `models/direction_diffusion.py`),
so `t*` is valid in [1, 199] — the Stage-4 kickoff's "≈1000" guess was wrong.

**Output/cache artifacts produced (repo root):** `stage4_tstar_sweep.*`, `stage4_truecomm_damage.*`,
`stage4_tau_calib.*` (was written for truecomm, then overwritten by the louvain run),
`stage4_noise_scale.*`, `stage4_louvain_damage.*`, and probs caches
`tau_cache_frozen_truecomm_t60.pt`, `tau_cache_frozen_louvain_t60.pt`.

**Outstanding kickoff item:** the canonical `stage3_scores.*` were NOT re-generated at the locked
settings. The finalized degradation curve lives in `stage4_tau_calib.csv` (the `τ=0.825` rows for
`frozen_truecomm`). To produce a clean canonical file, run
`t_star_sweep_score.py --t-stars 60 --tau 0.825 --frozen logs/lptask/sbm/frozen_truecomm --out stage3_scores`.

## 1. The `t*` sweep exposed a hard failure (frozen_original)

Swept `t*` ∈ {5,15,30,60,100,150,199} on SBM p30, tracking the stability score, a **raw
(pre-threshold) recovery signal**, a **null baseline** (random non-edges), and **raw diffusion
variance** across the K reconstructions (the thresholded score alone can't tell "generator dead"
from "generator destroying recovery" — both read 0).

Three regimes, no usable middle:

- `t*` ≤ 30: diffusion **inert** (raw diffusion sd ≈ 2e-4); recovery carried by the encoder.
- `t*` = 60: diffusion **wakes but destroys** — separation collapses to ≈ 0.05.
- `t*` ≥ 100: embeddings **diverge**, every decode prob → 0, score → 0.

## 2. Root cause — the anisotropic guidance was switched off the whole time

The guidance enters via per-community `label.npy` / `center.npy`, used **both to train and to
sample** the diffusion. On SBM those come from an unsupervised `HyperbolicKMeans(n_clusters=3,
max_epochs=5)` that **collapses to one cluster** (verified on the live model: `{0: 99, 2: 1}`,
ARI vs planted blocks ≈ 0). So the diffusion was trained against a degenerate target with no
community structure to preserve — hence it homogenizes then diverges as `t*` rises. Yet the true 3
blocks are **perfectly separable** in the encoder embedding (nearest-true-centroid accuracy 1.000,
silhouette 0.44) — only the clustering failed. `convert_sbm.py` already saves the ground truth
(`data/sbm/sbm_labels.pt`).

## 3. The fix (Phase 1 — ground-truth communities)

`retrain_truecomm.py`: rebuild `label.npy` from the true blocks and `center.npy` as the
per-community Fréchet mean (logmap0 → mean → expmap0, matching `cal_u0`), then retrain the diffusion
(`restrict=True`, Adam 1e-3, same 50 epochs) into `frozen_truecomm`. Only ONE variable changes
(community source). NaN gotcha fixed: the c=0.6 encoder emits a few nodes with ‖x‖ ≥ 1 while the
diffusion is hardcoded to c=1.0, so `logmap0`/`logmap` hit `artanh(≥1)` → NaN; both retrainer and
sweep clamp embeddings just inside the c=1.0 ball with a NaN guard.

## 4. Result — the collapse is gone (frozen_truecomm)

Separation vs `t*` (p30): orig collapses (0.53,0.55,0.45,**0.05,0.00,0.00,0.00**) → fixed holds and
rises (0.49,0.54,0.55,**0.56,0.57,0.58,0.58**). Divergence eliminated across the whole `t*` range —
before/after evidence for the paper's core claim that the anisotropic community constraint prevents
the diffusion from destroying topology. `t*=60` chosen (only point with a real diffusion-variance
bump while separation is preserved).

## 5. Damage sweep — fix holds; placeholder `τ=0.5` saturates

p10–p50 × {random, targeted} at `t*=60`: no NaN, separation ~0.55–0.57 everywhere. But at `τ=0.5`
the score is **flat** (0.92–0.94); `raw_removed_mean` ~0.80 at every p, because on SBM a removed
intra-community edge is predictable from community membership, so `τ=0.5` (which everything clears)
can't discriminate. Saturated _at that threshold_, not fundamentally.

## 6. `τ` calibration — restores the degradation curve (locked `τ=0.825`)

`tau_calibrate.py`. Youden's J points to low `τ` but the null is contaminated (~25% of random
non-edges are intra-community pairs that legitimately decode high, so FPR barely falls — J is the
wrong selector). Damage discrimination peaks at `τ ≈ 0.80–0.85`. Locked `τ=0.825`: random
0.855→0.795 (−0.060), targeted 0.883→0.757 (−0.126). Score declines with damage; smooth/graded.
(The targeted-faster ordering here is NOT robust — see §8.) Vs `τ=0.5` (flat ~0.93) the signal was
hidden by a saturated threshold — recovered **without changing the model**.

## 7. Diffusion diversity — the "thr_diff_sd = 0" gap was mostly a `τ=0.5` artifact (RESOLVED)

`noise_scale_sweep.py` scales the injected noise at `t*` (runtime monkeypatch on `q_sample`; only the
injection uses `noise=None`, so only it is scaled).

- **At the locked `τ=0.825`, `thr_diff_sd = 0.035` at λ=1** (p30) — NOT zero. The earlier
  `thr_diff_sd=0` was measured at `τ=0.5`, where the threshold sat far below the removed-edge probs.
  Moving `τ` into the probability mass for the degradation curve also surfaced the diffusion variance:
  the variance decomposition is non-degenerate (real structural AND diffusion variance).
- **Scaling the noise does not help.** Separation is unbreakable (0.545–0.562 up to λ=16), but raising
  λ trades recovery for modest, non-monotonic diversity, and extreme λ re-converges (guidance snaps
  all K to the same community-prior state). **λ=1 is the right operating point.**

## 8. Phase 2 — unsupervised clustering generalises the fix (frozen_louvain)

`retrain_unsup.py --method louvain`.

- **ARI 0.97 vs true blocks, exact sizes {34,33,33}** (old k-means ≈ 0). No ground truth used.
- Retrained model is collapse-free, separation ~0.56, `raw_removed_mean` ~0.806 (matches
  `frozen_truecomm`), real diffusion variance — mechanism reproduced without labels.
- **`τ` is model-specific.** `τ=0.825` leaves this model saturated; its own point is **`τ=0.85`**,
  same mean score (~0.83), clean decline: random 0.856→0.730 (−0.127), targeted 0.872→0.787 (−0.085).
- **Honest caveat — attack-type ordering is not robust on SBM.** `frozen_truecomm` → targeted faster;
  `frozen_louvain` → random faster (two models differing by ~3 nodes). Balanced SBM has near-uniform
  degrees, so degree-targeted ≈ random. **Robust claim on SBM: score declines with damage. The
  targeted-vs-random comparison belongs on BA (Stage 5).**

## 9. Next steps (Stage 5)

1. **BA / scale-free** for the targeted-vs-random comparison; revisit the BA negative-control thread
   with the fixed pipeline.
2. **Real datasets** (IMDB-B, PROTEINS) — need dgl + padding/masking; Louvain guidance carries over.
3. **Cleaner null** (inter-community non-edges only) for honest separation reporting.
4. (Optional) canonical `stage3_scores.*` refresh at the locked settings (command in the change log);
   update `config.py` `t_star=60`.

## Advisor framing

On SBM the diffusion never contributed because its community guidance was degenerate (k-means
collapse), not because diffusion can't help. Restoring communities — first from ground truth, then
unsupervised via Louvain (ARI 0.97) — removes the high-`t*` collapse entirely and makes the reverse
process structure-preserving; calibrating `τ` per model yields a degradation curve that declines with
damage. Two honest limits: the reconstruction distribution, while non-degenerate, is narrow (strong
guidance contracts the samples), and the targeted-vs-random ordering is not robust on SBM and belongs
on a scale-free graph. Both are matters of experimental design / further datasets, not failures of
the approach.
