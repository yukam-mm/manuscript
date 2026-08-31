"""
Swin2SR 4x super-resolution on Makrai LR images (D2 pipeline).

Mirror of 05_swin2sr_lama_dilated.py — same official HuggingFace model
(caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr), only the input/output
directories differ.

Inputs:  data/makrai/lr/*.jpg           (bicubic-downsampled Makrai)
Output:  outputs/sr_makrai/swin2sr/*.png

Usage (in .venv-sr on the server):
    python scripts/05_swin2sr_makrai.py
"""

import os
import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

PROJECT   = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT / "data" / "makrai" / "lr"
OUT_DIR   = PROJECT / "outputs" / "sr_makrai" / "swin2sr"

MODEL_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
UPSCALE = 4
DIRECT_MAX = 512
TILE = 256
OVERLAP = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("Loading Swin2SR model + processor ...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Swin2SRForImageSuperResolution.from_pretrained(
    MODEL_ID, use_safetensors=True
).to(device).eval()
print("Ready.\n")


def sr_direct(pil_img):
    inputs = processor(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    arr = out.reconstruction.data.squeeze().cpu().numpy()
    arr = np.clip(arr, 0, 1).transpose(1, 2, 0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def sr_tiled(pil_img):
    img = np.array(pil_img.convert("RGB"))
    H, W = img.shape[:2]
    step = TILE - OVERLAP
    out = np.zeros((H * UPSCALE, W * UPSCALE, 3), dtype=np.float32)
    wsum = np.zeros((H * UPSCALE, W * UPSCALE, 1), dtype=np.float32)
    y = 0
    while y < H:
        x = 0
        while x < W:
            y0, y1 = y, min(y + TILE, H)
            x0, x1 = x, min(x + TILE, W)
            tile = Image.fromarray(img[y0:y1, x0:x1])
            sr_tile = np.array(sr_direct(tile), dtype=np.float32)
            # Swin2SR internally pads to a multiple of the window size, so
            # sr_tile can be a few pixels larger than expected. Crop to fit.
            expected_h = (y1 - y0) * UPSCALE
            expected_w = (x1 - x0) * UPSCALE
            sr_tile = sr_tile[:expected_h, :expected_w]
            oy0, oy1 = y0 * UPSCALE, y1 * UPSCALE
            ox0, ox1 = x0 * UPSCALE, x1 * UPSCALE
            out[oy0:oy1, ox0:ox1]  += sr_tile
            wsum[oy0:oy1, ox0:ox1] += 1.0
            if x1 == W: break
            x += step
        if y1 == H: break
        y += step
    out /= np.maximum(wsum, 1e-6)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def sr_one(pil_img):
    H, W = pil_img.size[1], pil_img.size[0]
    if max(H, W) <= DIRECT_MAX:
        return sr_direct(pil_img)
    return sr_tiled(pil_img)


OUT_DIR.mkdir(parents=True, exist_ok=True)
imgs = sorted(list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png")))
print(f"Found {len(imgs)} images in {INPUT_DIR.name}/\n")

ok = failed = 0
for i, ip in enumerate(imgs, 1):
    op = OUT_DIR / (ip.stem + ".png")
    try:
        pil = Image.open(ip).convert("RGB")
        H, W = pil.size[1], pil.size[0]
        mode = "direct" if max(H, W) <= DIRECT_MAX else "tiled"
        sr = sr_one(pil)
        sr.save(op)
        print(f"[{i:2d}/{len(imgs)}] {ip.name:20s} {mode:6s} {W}x{H} -> {sr.size[0]}x{sr.size[1]}")
        ok += 1
    except Exception as e:
        print(f"[{i:2d}/{len(imgs)}] {ip.name:20s} FAIL: {type(e).__name__}: {e}")
        failed += 1

print(f"\nDone. ok={ok}  failed={failed}  total={len(imgs)}")
print(f"Outputs in: {OUT_DIR}")
