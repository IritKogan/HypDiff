# Stage 0 — Setup & Baseline Sanity Check (COMPLETE)

**Goal:** run the published HypDiff repo (RingBDStack/HypDiff) end-to-end, unmodified,
on one small dataset, to establish a known-good baseline and produce reusable weights
before any Phase B modifications.

**Status:** PASS — the `lptask` pipeline runs from data to metrics on the BA dataset.

## Baseline result (BA, CPU, default epochs)

- Encoder: 10 epochs, finishes with `End encoding!`
- Diffusion: 1 epoch, `Loss ≈ 1.003`
- Link-reconstruction metrics: **ROC ≈ 0.57**, **AP ≈ 0.62**

Scores are modest by design — this is a smoke test (10 encoder + 1 diffusion epoch),
not a tuned run. The point is that the unmodified pipeline runs correctly. For a
stronger baseline later: `--epochs 200 --diff_epoc 50`.

## What we had to work out (not in the repo)

The repo ships **no README, no requirements.txt, and no pretrained checkpoints** —
only three example datasets (BA, community, ego). Getting Stage 0 to run required:

1. **Environment, reconstructed by hand.** Dependencies derived from the code's
   imports: torch, torch_geometric, numpy, scipy, scikit-learn, pandas, matplotlib,
   networkx, tqdm, python-louvain (imported as `community`), ogb, pyemd.
   `tensorflow` / `edward` / `observations` are only used by one optional baseline
   (`graph_evaluate/baselines/mmsb.py`) and were skipped.
   `dgl` is **not** needed for the lptask/BA path — it is only used by the graphtask
   pipeline and the multi-graph dataset loaders (needed later for IMDB-B / PROTEINS).

2. **Data conversion (`convert_ba.py`).** The loader for `--dataset ba` reads
   `data/ba/ba_edge_index.pt`, `ba_features.pt`, `ba_labels.pt`, but the repo only
   ships `data/BA.pkl` (a list of 500 networkx graphs). `convert_ba.py` picks one
   graph and writes the three tensors: edge_index (both directions), identity-matrix
   features, and zero labels.

3. **CPU compatibility patch (`cpu_patch.py`).** The diffusion code hardcodes
   `.cuda()` in ~17 places (`models/direction_diffusion.py`, `models/base_models.py`,
   `diff.py`), which crashes on a CPU-only PyTorch build
   ("Torch not compiled with CUDA enabled"). The patch rewrites each to
   `.to('cuda' if torch.cuda.is_available() else 'cpu')` — CPU-safe, and still uses a
   GPU automatically if one is present. This changes tensor placement only, not the
   model's math.

## How to reproduce

```
python convert_ba.py                                                  # once, builds data/ba/
python cpu_patch.py                                                   # once, CPU-safe device fallback
python main.py --taskselect lptask --type train --dataset ba --cuda -1
# or, all-in-one with checks:
python stage0_smoke_test.py
```

## Deliverables

- Working environment (dependency list above; capture exact versions with `pip freeze`).
- Saved weights in `logs/lptask/ba/`: `model.pth` (encoder), `sample.pt` (diffusion),
  `embeddings.npy`, `label.npy`, `center.npy`, `config.json`.
- `stage0_smoke_test.py` — reproducible baseline runner.

## Known issues deferred to later stages

- **dgl on Windows / Python 3.11:** `import dgl` fails (no graphbolt DLL matching the
  installed torch). Not needed until IMDB-B / PROTEINS (Stages 5–6). Fix then with a
  pinned conda env (e.g. Python 3.10 + torch 2.3 + matching dgl build).
- **Environment defaults assume a GPU** (`--cuda 0`); pass `--cuda -1` for CPU.

```

```
