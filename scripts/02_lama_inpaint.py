"""
Stage 1 text removal via LaMa deep-learning inpainter.

Uses simple-lama-inpainting, which wraps the pre-trained Big-LaMa checkpoint
from Suvorov et al. (2022). Auto-downloads the ~200 MB checkpoint on first
run to the torch hub cache (~/.cache/torch/hub/checkpoints/).

Follows the paper's protocol: pre-trained big-lama, no fine-tuning, same
LabelMe-derived masks used for the OpenCV Telea baseline.

Inputs:  data/inhouse/raw/<stem>.png
         data/inhouse/masks_final/<stem>_mask.png
Output:  outputs/inpainting/lama/<stem>.png

Usage (from project root):
    python scripts/02_lama_inpaint.py
"""

import os
import glob
from PIL import Image
from simple_lama_inpainting import SimpleLama

RAW_DIR  = "data/inhouse/raw"
MASK_DIR = "data/inhouse/masks_final"
OUT_DIR  = "outputs/inpainting/lama"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading Big-LaMa checkpoint (downloads on first run) ...")
lama = SimpleLama()
print("Ready.\n")

raws = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))
print(f"Found {len(raws)} raw images.\n")

ok = 0
skipped = 0
for raw_path in raws:
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    mask_path = os.path.join(MASK_DIR, f"{stem}_mask.png")

    if not os.path.exists(mask_path):
        print(f"skip {stem}: no mask found")
        skipped += 1
        continue

    try:
        img  = Image.open(raw_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        result = lama(img, mask)

        out_path = os.path.join(OUT_DIR, f"{stem}.png")
        result.save(out_path)
        print(f"saved {out_path}")
        ok += 1
    except Exception as e:
        print(f"FAIL  {stem}: {e}")

print(f"\nDone. ok={ok}  skipped={skipped}  total={len(raws)}")
