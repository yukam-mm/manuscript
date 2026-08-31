"""
LaMa inpainting on the dilated manual masks.

Same as 02_lama_inpaint.py but reads masks from data/inhouse/masks_dilated/
(produced by scripts/01b_dilate_masks.py) and writes to
outputs/inpainting/lama_dilated/.

Usage (from project root, on GPU server):
    python scripts/02b_lama_inpaint_dilated.py
"""

import os
import glob
from PIL import Image
from simple_lama_inpainting import SimpleLama

RAW_DIR  = "data/inhouse/raw"
MASK_DIR = "data/inhouse/masks_dilated"
OUT_DIR  = "outputs/inpainting/lama_dilated"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading Big-LaMa checkpoint (cached from first run) ...")
lama = SimpleLama()
print("Ready.\n")

raws = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))
print(f"Found {len(raws)} raw images.\n")

ok = skipped = 0
for raw_path in raws:
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    mask_path = os.path.join(MASK_DIR, f"{stem}_mask.png")

    if not os.path.exists(mask_path):
        print(f"skip {stem}: no dilated mask found")
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
