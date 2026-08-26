"""
stage0_smoke_test.py  —  Stage 0 reproducible smoke test for HypDiff
 
Reproduces one full unmodified-baseline run of the lptask pipeline on the BA
dataset and checks that it completes end-to-end:
 
  1. ensures data/ba/ba_*.pt exist (runs convert_ba.py if missing)
  2. runs:  python main.py --taskselect lptask --type train --dataset ba --cuda -1
  3. checks the expected weight/output files were written to logs/lptask/ba/
  4. parses the ROC and AP scores printed at the end
  5. prints PASS / FAIL
 
This is the Stage 0 deliverable: a known-good baseline anyone can re-run to
confirm the environment + code + data are wired correctly before Stage 1.
 
Prereqs: environment set up (torch, torch_geometric, numpy, scipy, sklearn,
networkx, ...), convert_ba.py and cpu_patch.py already applied.
 
Usage:
    python stage0_smoke_test.py
    python stage0_smoke_test.py --cuda 0    # if you have an NVIDIA GPU
"""
 
import argparse
import os
import re
import subprocess
import sys
 
DATA_DIR = os.path.join("data", "ba")
DATA_FILES = ["ba_edge_index.pt", "ba_features.pt", "ba_labels.pt"]
LOG_DIR = os.path.join("logs", "lptask", "ba")
EXPECTED_OUTPUTS = ["model.pth", "embeddings.npy", "label.npy", "center.npy", "sample.pt"]
 
 
def ensure_data():
    missing = [f for f in DATA_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if not missing:
        print(f"[data]  OK - {DATA_DIR} already has the ba_*.pt files")
        return
    print(f"[data]  missing {missing}; running convert_ba.py ...")
    r = subprocess.run([sys.executable, "convert_ba.py"], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr)
        sys.exit("[data]  FAIL - convert_ba.py did not complete")
 
 
def run_pipeline(cuda):
    cmd = [sys.executable, "main.py", "--taskselect", "lptask",
           "--type", "train", "--dataset", "ba", "--cuda", str(cuda)]
    print(f"[run]   {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit("[run]   FAIL - main.py crashed (see traceback above)")
    return r.stdout
 
 
def check_outputs():
    missing = [f for f in EXPECTED_OUTPUTS if not os.path.exists(os.path.join(LOG_DIR, f))]
    if missing:
        sys.exit(f"[files] FAIL - missing output files in {LOG_DIR}: {missing}")
    print(f"[files] OK - all weights present in {LOG_DIR}: {EXPECTED_OUTPUTS}")
 
 
def parse_scores(stdout):
    # The pipeline prints 'End encoding!' then, at the very end, ROC then AP
    # as two bare floats on their own lines.
    floats = re.findall(r"^\s*(\d\.\d+)\s*$", stdout, flags=re.MULTILINE)
    if len(floats) >= 2:
        roc, ap = float(floats[-2]), float(floats[-1])
        print(f"[score] ROC={roc:.4f}  AP={ap:.4f}")
        return roc, ap
    print("[score] WARN - could not parse ROC/AP from output (run still completed)")
    return None, None
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default="-1", help="-1 for CPU, 0 for first GPU")
    args = ap.parse_args()
 
    if not os.path.exists("main.py"):
        sys.exit("Run this from the HypDiff repo root (main.py not found here).")
 
    print("=" * 60)
    print("HypDiff Stage 0 smoke test (lptask / BA baseline)")
    print("=" * 60)
    ensure_data()
    stdout = run_pipeline(args.cuda)
    if "End encoding!" not in stdout:
        sys.exit("[run]   FAIL - encoder phase did not finish ('End encoding!' not seen)")
    check_outputs()
    parse_scores(stdout)
    print("\nRESULT: PASS - unmodified HypDiff baseline runs end-to-end.")
 
 
if __name__ == "__main__":
    main()