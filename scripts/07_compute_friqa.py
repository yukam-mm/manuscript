"""
D2 full-reference IQA on Makrai SR outputs.

Metrics (all official reference implementations, no reimplementations):
    PSNR  ↑ - skimage.metrics.peak_signal_noise_ratio  (scikit-image)
    SSIM  ↑ - skimage.metrics.structural_similarity     (scikit-image)
    LPIPS ↓ - lpips.LPIPS(net='alex')                    (Zhang et al. official)

Compares each SR output against its HR ground truth.

Handling of size mismatches:
    SR outputs can be a few pixels smaller (bicubic degrade rounded down)
    or larger (Swin2SR internal padding) than HR. Both are cropped
    top-left to the smallest common dimensions before scoring.

LPIPS note:
    LPIPS at full 4x resolution (up to ~4000x3000 px) needs >8GB GPU/RAM.
    Standard SR-benchmark practice is to downsample to a fixed max-side
    for LPIPS. Default: 512 px (matches BSD100 / RealSR benchmarks).
    PSNR and SSIM are computed at full resolution — no downsampling.

Inputs:
    HR:  data/makrai/sample_20/images/<stem>.jpg
    SR:  outputs/sr_makrai/<method>/<stem>{,_out}.png
Output:
    metrics/d2_reconstruction/d2_fr_metrics.csv
    columns: stem, method, H, W, psnr, ssim, lpips

Usage (from project root):
    python scripts/07_compute_friqa.py
"""

import os
import csv
import glob
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import lpips

HR_DIR = "data/makrai/sample_20/images"
METHODS = {
    "swin2sr":    "outputs/sr_makrai/swin2sr",
    "realesrgan": "outputs/sr_makrai/realesrgan",
    "aesrgan":    "outputs/sr_makrai/aesrgan",
}
OUT_CSV = "metrics/d2_reconstruction/d2_fr_metrics.csv"

LPIPS_MAX_SIDE = 512    # downsample for LPIPS only; PSNR/SSIM at full res

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("Loading LPIPS (AlexNet) ...")
lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def find_sr(dir_path, stem):
    for suffix in ("", "_out"):
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(dir_path, f"{stem}{suffix}{ext}")
            if os.path.exists(p):
                return p
    return None


def align_common(a, b):
    """Top-left crop both to smallest common (H, W)."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def to_lpips_tensor(rgb_uint8):
    """RGB uint8 (H,W,3) -> torch (1,3,H,W) in [-1,1] as LPIPS expects."""
    t = torch.from_numpy(rgb_uint8.transpose(2, 0, 1)).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def downsample_for_lpips(rgb):
    """Resize so max side == LPIPS_MAX_SIDE, keep aspect ratio."""
    H, W = rgb.shape[:2]
    m = max(H, W)
    if m <= LPIPS_MAX_SIDE:
        return rgb
    scale = LPIPS_MAX_SIDE / m
    new_w, new_h = int(W * scale), int(H * scale)
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def score(hr_bgr, sr_bgr):
    hr_a, sr_a = align_common(hr_bgr, sr_bgr)          # BGR uint8
    hr_rgb = cv2.cvtColor(hr_a, cv2.COLOR_BGR2RGB)
    sr_rgb = cv2.cvtColor(sr_a, cv2.COLOR_BGR2RGB)

    # PSNR + SSIM at full resolution
    psnr = float(sk_psnr(hr_rgb, sr_rgb, data_range=255))
    ssim = float(sk_ssim(hr_rgb, sr_rgb, data_range=255, channel_axis=-1))

    # LPIPS on downsampled versions (see header note)
    hr_lp = downsample_for_lpips(hr_rgb)
    sr_lp = downsample_for_lpips(sr_rgb)
    with torch.no_grad():
        l = lpips_model(to_lpips_tensor(hr_lp), to_lpips_tensor(sr_lp)).item()

    return {"H": hr_a.shape[0], "W": hr_a.shape[1],
            "psnr": psnr, "ssim": ssim, "lpips": float(l)}


os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
hrs = sorted(glob.glob(os.path.join(HR_DIR, "*.jpg")))
stems = [os.path.splitext(os.path.basename(p))[0] for p in hrs]
print(f"\n{len(stems)} HR images x {len(METHODS)} methods = "
      f"{len(stems) * len(METHODS)} evaluations\n")

rows = []
for i, stem in enumerate(stems, 1):
    hr_path = os.path.join(HR_DIR, f"{stem}.jpg")
    hr = cv2.imread(hr_path)
    if hr is None:
        print(f"[{i:2d}/{len(stems)}] skip {stem}: HR unreadable")
        continue

    for method, d in METHODS.items():
        sr_path = find_sr(d, stem)
        if sr_path is None:
            print(f"[{i:2d}/{len(stems)}] {stem:14s} {method:12s} SKIP (missing)")
            continue
        sr = cv2.imread(sr_path)
        if sr is None:
            print(f"[{i:2d}/{len(stems)}] {stem:14s} {method:12s} SKIP (unreadable)")
            continue

        try:
            res = score(hr, sr)
            rows.append({"stem": stem, "method": method, **res})
            print(f"[{i:2d}/{len(stems)}] {stem:14s} {method:12s} "
                  f"PSNR={res['psnr']:6.2f}  SSIM={res['ssim']:.4f}  LPIPS={res['lpips']:.4f}")
        except Exception as e:
            print(f"[{i:2d}/{len(stems)}] {stem:14s} {method:12s} FAIL: {type(e).__name__}: {e}")

if rows:
    fields = ["stem", "method", "H", "W", "psnr", "ssim", "lpips"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
