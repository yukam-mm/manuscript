"""
Prerequisites (already set up on server per the earlier steps):
    ~/ReMOVE/                              (git clone of official repo)
    ~/ReMOVE/models/sam_vit_h_4b8939.pth   (SAM ViT-H weights)
    ~/ReMOVE/.venv-remove/                 (Python 3.8.18 venv with SAM installed)

Run on the GPU server, using the ReMOVE venv (has SAM + torch installed):
    source ~/ReMOVE/.venv-remove/bin/activate
    python ~/mdpi_colony_enhancement/scripts/06b_batch_remove.py

Output: metrics/d1_reference_free/d1_remove.csv
    columns: stem, method, remove_score
"""

import os
import sys
import csv
import re
import contextlib
from io import StringIO
from pathlib import Path
from argparse import Namespace

import torch
from segment_anything import sam_model_registry, SamPredictor


REMOVE_REPO = Path.home() / "ReMOVE"
PROJECT     = Path.home() / "mdpi_colony_enhancement"
CHECKPOINT  = REMOVE_REPO / "models" / "sam_vit_h_4b8939.pth"

METHODS = {
    "raw":               PROJECT / "data/inhouse/raw",
    "telea":             PROJECT / "outputs/inpainting/telea",
    "navier_stokes":     PROJECT / "outputs/inpainting/navier_stokes",
    "lama_manual_mask":  PROJECT / "outputs/inpainting/lama_manual_mask",
    "lama_dilated":      PROJECT / "outputs/inpainting/lama_dilated",
    "lama_auto":         PROJECT / "outputs/inpainting/lama_auto",
}
MASK_DIR = PROJECT / "data" / "inhouse" / "masks_final"
OUT_CSV  = PROJECT / "metrics" / "d1_reference_free" / "d1_remove.csv"


if not (REMOVE_REPO / "main.py").exists():
    print(f"error: ReMOVE main.py not found at {REMOVE_REPO}/main.py")
    sys.exit(1)
if not CHECKPOINT.exists():
    print(f"error: SAM checkpoint not found at {CHECKPOINT}")
    sys.exit(1)

# ReMOVE's main.py uses relative paths (e.g. models/); chdir so they resolve.
os.chdir(REMOVE_REPO)
sys.path.insert(0, str(REMOVE_REPO))
from main import get_score          # official ReMOVE function, used verbatim


print(f"Loading SAM ViT-H from {CHECKPOINT} ...")
sam = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT)).cuda()
predictor = SamPredictor(sam)
print("Model ready.\n")


SCORE_RE = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def call_get_score(img_path, mask_path):
    args = Namespace(
        image_path=str(img_path),
        mask_path=str(mask_path),
        crop=False,          # match the smoke-test invocation
        draw=False,
    )
    buf = StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ret = get_score(predictor, args)
    except Exception as e:
        print(f"    exception: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    # If get_score returns a numeric, use that
    if ret is not None:
        try:
            return float(ret)
        except (TypeError, ValueError):
            pass
    # Otherwise scrape the number out of what it printed
    out = buf.getvalue().strip().splitlines()
    for line in reversed(out):
        m = SCORE_RE.search(line)
        if m:
            return float(m.group(1))
    print(f"    unexpected output: {buf.getvalue()!r}", file=sys.stderr)
    return None



OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

raws = sorted(METHODS["raw"].glob("*.png"))
stems = [p.stem for p in raws]
print(f"{len(stems)} images x {len(METHODS)} methods "
      f"= {len(stems) * len(METHODS)} evaluations\n")

rows = []
for i, stem in enumerate(stems, 1):
    mask_path = MASK_DIR / f"{stem}_mask.png"
    if not mask_path.exists():
        print(f"[{i}/{len(stems)}] skip {stem}: no mask")
        continue
    for method, d in METHODS.items():
        img_path = d / f"{stem}.png"
        if not img_path.exists():
            print(f"[{i}/{len(stems)}] {stem:14s} {method:20s} SKIP (missing)")
            continue
        score = call_get_score(img_path, mask_path)
        if score is None:
            print(f"[{i}/{len(stems)}] {stem:14s} {method:20s} FAIL")
            continue
        rows.append({"stem": stem, "method": method, "remove_score": score})
        print(f"[{i}/{len(stems)}] {stem:14s} {method:20s} score={score:.6f}")

if rows:
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "method", "remove_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
else:
    print("\nNo rows produced.")
