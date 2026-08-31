"""
NIQE + BRISQUE on SR outputs (three methods) 
"""

import os
import csv
import glob
import cv2
import numpy as np
import torch
import pyiqa

METHODS = {
    "lama_dilated": "outputs/inpainting/lama_dilated",           # SR input baseline
    "swin2sr":      "outputs/sr_inhouse/swin2sr",
    "realesrgan":   "outputs/sr_inhouse/realesrgan",
    "aesrgan":      "outputs/sr_inhouse/aesrgan",
}
MASK_DIR = "data/inhouse/masks_final"
PADDING = 20                # px around mask bbox
MIN_CROP = 96               # NIQE/BRISQUE minimum
OUT_CSV = "metrics/d1_reference_free/sr_nriqa.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("Loading NIQE ...")
niqe = pyiqa.create_metric("niqe", device=device)
print("Loading BRISQUE ...")
brisque = pyiqa.create_metric("brisque", device=device)


def find_img(dir_path, stem):
    """Handle both `<stem>.png` and `<stem>_out.png`."""
    for suffix in ("", "_out"):
        p = os.path.join(dir_path, f"{stem}{suffix}.png")
        if os.path.exists(p):
            return p
    return None


def bbox_from_mask(mask, pad, img_h, img_w):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(img_w, int(xs.max()) + pad + 1)
    y1 = min(img_h, int(ys.max()) + pad + 1)
    if (x1 - x0) < MIN_CROP:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - MIN_CROP // 2)
        x1 = min(img_w, x0 + MIN_CROP)
    if (y1 - y0) < MIN_CROP:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - MIN_CROP // 2)
        y1 = min(img_h, y0 + MIN_CROP)
    return (x0, y0, x1, y1)


def to_tensor(bgr_crop):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return t.to(device)


def score(img_path, mask_input_res):
    """mask_input_res is at input resolution; resize to match img dims."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    if mask_input_res.shape[:2] != (H, W):
        mask = cv2.resize(mask_input_res, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_input_res
    bbox = bbox_from_mask(mask, PADDING, H, W)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    t = to_tensor(crop)
    try:
        n_val = float(niqe(t).item())
    except Exception as e:
        print(f"    niqe failed: {e}")
        n_val = float("nan")
    try:
        b_val = float(brisque(t).item())
    except Exception as e:
        print(f"    brisque failed: {e}")
        b_val = float("nan")

    return {
        "region_h": y1 - y0,
        "region_w": x1 - x0,
        "niqe": n_val,
        "brisque": b_val,
    }


os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
# Enumerate stems from lama_dilated (input to SR)
inputs = sorted(glob.glob(os.path.join(METHODS["lama_dilated"], "*.png")))
stems = [os.path.splitext(os.path.basename(p))[0] for p in inputs]
print(f"\n{len(stems)} images x {len(METHODS)} conditions = "
      f"{len(stems)*len(METHODS)} evaluations\n")

rows = []
for i, stem in enumerate(stems, 1):
    mask_path = os.path.join(MASK_DIR, f"{stem}_mask.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[{i}/{len(stems)}] skip {stem}: no mask")
        continue

    for method, d in METHODS.items():
        img_path = find_img(d, stem)
        if img_path is None:
            print(f"[{i}/{len(stems)}] {stem:12s} {method:15s} SKIP (missing)")
            continue
        res = score(img_path, mask)
        if res is None:
            print(f"[{i}/{len(stems)}] {stem:12s} {method:15s} FAIL")
            continue
        rows.append({"stem": stem, "method": method, **res})
        print(f"[{i}/{len(stems)}] {stem:12s} {method:15s} "
              f"NIQE={res['niqe']:.3f}  BRISQUE={res['brisque']:.3f}")

if rows:
    fields = ["stem", "method", "region_h", "region_w", "niqe", "brisque"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
