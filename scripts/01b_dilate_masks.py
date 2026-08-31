"""
Dilate manual LabelMe masks with a morphological elliptical kernel.

Default: 7x7 elliptical (matches paper size, rounder than paper's square shape).
Alternatives:
    --size 5           # smaller margin, safer near colonies
    --size 9           # larger margin, if pen residue survives inpainting
    --shape rect       # square kernel (paper protocol exactly)

Inputs:  data/inhouse/masks_final/<stem>_mask.png
Output:  data/inhouse/masks_dilated/<stem>_mask.png  (same filename)

Usage (from project root):
    python scripts/01b_dilate_masks.py
    python scripts/01b_dilate_masks.py --size 9
    python scripts/01b_dilate_masks.py --shape rect
    python scripts/01b_dilate_masks.py --out data/inhouse/masks_dilated_9x9
"""

import argparse
import glob
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--src",   default="data/inhouse/masks_final")
parser.add_argument("--out",   default="data/inhouse/masks_dilated")
parser.add_argument("--size",  type=int, default=7,
                    help="kernel size in pixels (default 7)")
parser.add_argument("--shape", choices=["ellipse", "rect"], default="ellipse",
                    help="kernel shape (default ellipse)")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

kshape = cv2.MORPH_ELLIPSE if args.shape == "ellipse" else cv2.MORPH_RECT
kernel = cv2.getStructuringElement(kshape, (args.size, args.size))
print(f"Dilating {args.src} -> {args.out}")
print(f"Kernel: {args.shape} {args.size}x{args.size}\n")

for src in sorted(glob.glob(os.path.join(args.src, "*.png"))):
    fname = os.path.basename(src)
    mask = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"skip {fname}: failed to read")
        continue
    dilated = cv2.dilate(mask, kernel, iterations=1)
    dst = os.path.join(args.out, fname)
    cv2.imwrite(dst, dilated)
    print(f"saved {dst}")
