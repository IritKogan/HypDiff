"""
cpu_patch.py  —  Stage 0 CPU-compatibility patch for HypDiff (RingBDStack/HypDiff)
 
Why this is needed
------------------
Several files hardcode `.cuda()`, which assumes an NVIDIA GPU. On a CPU-only
PyTorch build this raises:
    AssertionError: Torch not compiled with CUDA enabled
 
This script rewrites every bare `.cuda()` into:
    .to('cuda' if torch.cuda.is_available() else 'cpu')
so the code runs on CPU *and* still uses the GPU automatically if one exists.
It does NOT change any model math — it only changes where tensors live.
 
It leaves the `torch.cuda.is_available()` device-detection lines untouched
(those already fall back to CPU correctly).
 
Usage
-----
    python cpu_patch.py            # patches the files, writes .bak backups first
    python cpu_patch.py --revert   # restores the .bak backups
 
Run it once from the HypDiff repo root, then re-run:
    python main.py --taskselect lptask --type train --dataset ba --cuda -1
"""
 
import argparse
import os
 
# Files that contain bare .cuda() calls in the lptask pipeline (+ generate()).
TARGETS = [
    "models/direction_diffusion.py",
    "models/base_models.py",
    "diff.py",
]
 
OLD = ".cuda()"
NEW = ".to('cuda' if torch.cuda.is_available() else 'cpu')"
 
 
def revert():
    for path in TARGETS:
        bak = path + ".bak"
        if os.path.exists(bak):
            with open(bak, "r", encoding="utf-8") as f:
                text = f.read()
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            os.remove(bak)
            print(f"reverted {path}")
        else:
            print(f"no backup for {path}, skipped")
 
 
def patch():
    for path in TARGETS:
        if not os.path.exists(path):
            print(f"SKIP (not found): {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
 
        # Count only the bare `.cuda()` calls. `torch.cuda.is_available()`
        # contains `.cuda.` not `.cuda()`, so it is not affected.
        n = text.count(OLD)
        if n == 0:
            print(f"no changes needed: {path}")
            continue
 
        # Write a one-time backup (don't overwrite an existing .bak).
        bak = path + ".bak"
        if not os.path.exists(bak):
            with open(bak, "w", encoding="utf-8") as f:
                f.write(text)
 
        text = text.replace(OLD, NEW)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"patched {path}: {n} `.cuda()` call(s) -> CPU/GPU-safe")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="restore .bak backups")
    args = ap.parse_args()
    if args.revert:
        revert()
    else:
        patch()
        print("\nDone. Now run:")
        print("  python main.py --taskselect lptask --type train --dataset ba --cuda -1")