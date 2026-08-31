"""
Reference-free image-quality assessment on inpainted outputs.

Metrics:
    NIQE        (pyiqa)    - natural-scene-statistics; sensitive to over-smoothing
    BRISQUE     (pyiqa)    - spatial-domain distortion; trained on natural photos
    EasyOCR     (easyocr)  - sum of detected-text confidences; ↓ = less residual text

"""

import os
import csv
import glob
import cv2
import numpy as np
import torch
import pyiqa
import easyocr


METHODS = {
    "raw":               "data/inhouse/raw",
    "telea":             "outputs/inpainting/telea",
    "navier_stokes":     "outputs/inpainting/navier_stokes",
    "lama_manual_mask":  "outputs/inpainting/lama_manual_mask",
    "lama_dilated":      "outputs/inpainting/lama_dilated",
    "lama_auto":         "outputs/inpainting/lama_auto",
}
MASK_DIR = "data/inhouse/masks_final"   # source of bounding boxes
PADDING = 20                             # px around mask bbox
MIN_CROP = 96                            # NIQE/BRISQUE need this minimum
OUT_CSV = "metrics/d1_reference_free/d1_nriqa.csv"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("Loading NIQE ...")
niqe    = pyiqa.create_metric("niqe",    device=device)
print("Loading BRISQUE ...")
brisque = pyiqa.create_metric("brisque", device=device)
print("Loading EasyOCR (English, no GPU ok) ...")
ocr = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)



def bbox_from_mask(mask_path, pad, img_h, img_w):
    """Return (x0, y0, x1, y1) around white pixels in mask, padded and clipped
    to image bounds. Returns None if mask is empty."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(img_w, int(xs.max()) + pad + 1)
    y1 = min(img_h, int(ys.max()) + pad + 1)
    # Enforce a minimum size so NIQE/BRISQUE don't fail
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
    """BGR np.uint8 (H,W,3) -> torch (1,3,H,W) float in [0,1] on device."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return t.to(device)


def score_one(img_path, bbox):
    img = cv2.imread(img_path)
    if img is None:
        return None
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    t = to_tensor(crop)
    try:
        n_val = float(niqe(t).item())
    except Exception as e:
        print(f"  niqe failed: {e}")
        n_val = float("nan")
    try:
        b_val = float(brisque(t).item())
    except Exception as e:
        print(f"  brisque failed: {e}")
        b_val = float("nan")

    # EasyOCR runs on the numpy crop directly
    try:
        ocr_result = ocr.readtext(crop)   # list of (bbox, text, conf)
        conf_sum = float(sum(r[2] for r in ocr_result))
        n_det = len(ocr_result)
    except Exception as e:
        print(f"  easyocr failed: {e}")
        conf_sum, n_det = float("nan"), 0

    return {
        "region_h": y1 - y0,
        "region_w": x1 - x0,
        "niqe": n_val,
        "brisque": b_val,
        "ocr_conf_sum": conf_sum,
        "ocr_n_detections": n_det,
    }


os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# Enumerate stems from raw folder
raws = sorted(glob.glob(os.path.join(METHODS["raw"], "*.png")))
stems = [os.path.splitext(os.path.basename(p))[0] for p in raws]
print(f"\n{len(stems)} images x {len(METHODS)} methods "
      f"= {len(stems) * len(METHODS)} evaluations\n")

rows = []
for i, stem in enumerate(stems, 1):
    # Load raw to get dimensions, then compute bbox once per stem
    raw_path = os.path.join(METHODS["raw"], f"{stem}.png")
    img = cv2.imread(raw_path)
    if img is None:
        print(f"[{i}/{len(stems)}] skip {stem}: raw not readable")
        continue

    mask_path = os.path.join(MASK_DIR, f"{stem}_mask.png")
    bbox = bbox_from_mask(mask_path, PADDING, img.shape[0], img.shape[1])
    if bbox is None:
        print(f"[{i}/{len(stems)}] skip {stem}: no valid mask")
        continue

    for method, d in METHODS.items():
        img_path = os.path.join(d, f"{stem}.png")
        if not os.path.exists(img_path):
            print(f"[{i}/{len(stems)}] {stem:12s} {method:20s} SKIP (missing)")
            continue
        res = score_one(img_path, bbox)
        if res is None:
            print(f"[{i}/{len(stems)}] {stem:12s} {method:20s} FAIL")
            continue
        row = {"stem": stem, "method": method, **res}
        rows.append(row)
        print(f"[{i}/{len(stems)}] {stem:12s} {method:20s} "
              f"NIQE={res['niqe']:.3f} BRISQUE={res['brisque']:.3f} "
              f"OCR={res['ocr_conf_sum']:.3f} ({res['ocr_n_detections']})")

# Write CSV
if rows:
    fields = ["stem", "method", "region_h", "region_w",
              "niqe", "brisque", "ocr_conf_sum", "ocr_n_detections"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
else:
    print("\nNo rows produced.")
