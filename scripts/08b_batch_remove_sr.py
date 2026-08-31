"""
Batch ReMOVE on SR outputs using the official ReMOVE implementation.

"""

import os
import sys
import csv
import re
import contextlib
import tempfile
from io import StringIO
from pathlib import Path
from argparse import Namespace

# --- Paths (edit if needed) ---
REMOVE_REPO = Path.home() / "ReMOVE"
PROJECT     = Path(__file__).resolve().parent.parent
CHECKPOINT  = REMOVE_REPO / "models" / "sam_vit_h_4b8939.pth"

# CRITICAL: put ReMOVE's local segment_anything at front of path BEFORE import
if not (REMOVE_REPO / "main.py").exists():
    print(f"error: ReMOVE main.py not at {REMOVE_REPO}/main.py")
    sys.exit(1)
if not CHECKPOINT.exists():
    print(f"error: SAM checkpoint not at {CHECKPOINT}")
    sys.exit(1)
os.chdir(REMOVE_REPO)
sys.path.insert(0, str(REMOVE_REPO))

import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor   # patched
from main import get_score                                       # official

if not hasattr(SamPredictor, "get_aggregate_features"):
    print("error: SamPredictor missing get_aggregate_features — pip SAM shadowed ReMOVE's copy.")
    sys.exit(1)

METHODS = {
    "lama_dilated": PROJECT / "outputs/inpainting/lama_dilated",
    "swin2sr":      PROJECT / "outputs/sr_inhouse/swin2sr/lama_dilated",
    "realesrgan":   PROJECT / "outputs/sr_inhouse/realesrgan/lama_dilated",
    "aesrgan":      PROJECT / "outputs/sr_inhouse/aesrgan/lama_dilated",
}
MASK_DIR = PROJECT / "data" / "inhouse" / "masks_final"
OUT_CSV  = PROJECT / "metrics" / "d1_reference_free" / "sr_remove.csv"

# ---------------------------------------------------------------------------
# Load SAM once
# ---------------------------------------------------------------------------
print(f"Loading SAM ViT-H from {CHECKPOINT} ...")
sam = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT)).cuda()
predictor = SamPredictor(sam)
print("Model ready.\n")

SCORE_RE = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def find_img(dir_path, stem):
    for suffix in ("", "_out"):
        p = dir_path / f"{stem}{suffix}.png"
        if p.exists():
            return p
    return None


def call_get_score(img_path, mask_path):
    args = Namespace(
        image_path=str(img_path),
        mask_path=str(mask_path),
        crop=False,
        draw=False,
    )
    buf = StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ret = get_score(predictor, args)
    except Exception as e:
        print(f"    exception: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    if ret is not None:
        try:
            return float(ret)
        except (TypeError, ValueError):
            pass
    for line in reversed(buf.getvalue().strip().splitlines()):
        m = SCORE_RE.search(line)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Enumerate stems from lama_dilated
inputs = sorted(METHODS["lama_dilated"].glob("*.png"))
stems = [p.stem for p in inputs]
print(f"{len(stems)} images x {len(METHODS)} conditions = "
      f"{len(stems)*len(METHODS)} evaluations\n")

rows = []
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for i, stem in enumerate(stems, 1):
        mask_src = MASK_DIR / f"{stem}_mask.png"
        if not mask_src.exists():
            print(f"[{i}/{len(stems)}] skip {stem}: no mask")
            continue
        mask_input_res = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
        if mask_input_res is None:
            print(f"[{i}/{len(stems)}] skip {stem}: mask unreadable")
            continue

        for method, d in METHODS.items():
            img_path = find_img(d, stem)
            if img_path is None:
                print(f"[{i}/{len(stems)}] {stem:14s} {method:14s} SKIP (missing)")
                continue

            # Resize mask to match this image's dims (SR is 4x)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[{i}/{len(stems)}] {stem:14s} {method:14s} SKIP (unreadable)")
                continue
            H, W = img.shape[:2]
            if mask_input_res.shape[:2] != (H, W):
                mask_scaled = cv2.resize(mask_input_res, (W, H),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                mask_scaled = mask_input_res

            # Write the (possibly-resized) mask to a temp file for ReMOVE
            mask_tmp = tmp / f"{stem}_{method}_mask.png"
            cv2.imwrite(str(mask_tmp), mask_scaled)

            score = call_get_score(img_path, mask_tmp)
            if score is None:
                print(f"[{i}/{len(stems)}] {stem:14s} {method:14s} FAIL")
                continue
            rows.append({"stem": stem, "method": method, "remove_score": score})
            print(f"[{i}/{len(stems)}] {stem:14s} {method:14s} score={score:.6f}")

if rows:
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "method", "remove_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
