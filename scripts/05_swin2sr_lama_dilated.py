"""
Swin2SR 4x super-resolution on inpainted images (lama_dilated).

"""

import os
import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# Paths auto-detected from script's location
PROJECT   = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT / "outputs" / "inpainting" / "lama_dilated"
OUT_DIR   = PROJECT / "outputs" / "sr_inhouse" / "swin2sr" / "lama_dilated"

MODEL_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
UPSCALE = 4              # this checkpoint is x4
DIRECT_MAX = 512         # if max(H,W) <= this, run direct; otherwise tile
TILE = 256               # tile size (input, pre-SR)
OVERLAP = 32             # overlap between tiles (input px)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("Loading Swin2SR model + processor ...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_ID, use_safetensors=True).to(device).eval()
print("Ready.\n")


def sr_direct(pil_img):
    """Direct SR — one forward pass."""
    inputs = processor(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    arr = out.reconstruction.data.squeeze().cpu().numpy()          # (3, H*4, W*4)
    arr = np.clip(arr, 0, 1).transpose(1, 2, 0)                    # (H*4, W*4, 3)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def sr_tiled(pil_img):
    """Tile-based SR for large images to avoid GPU OOM."""
    img = np.array(pil_img.convert("RGB"))
    H, W = img.shape[:2]
    step = TILE - OVERLAP
    # Output canvas (upscaled) + weight canvas for blending overlaps
    out = np.zeros((H * UPSCALE, W * UPSCALE, 3), dtype=np.float32)
    wsum = np.zeros((H * UPSCALE, W * UPSCALE, 1), dtype=np.float32)

    y = 0
    while y < H:
        x = 0
        while x < W:
            y0, y1 = y, min(y + TILE, H)
            x0, x1 = x, min(x + TILE, W)
            tile = Image.fromarray(img[y0:y1, x0:x1])
            sr_tile = np.array(sr_direct(tile), dtype=np.float32)   # (h*4, w*4, 3)
            oy0, oy1 = y0 * UPSCALE, y0 * UPSCALE + sr_tile.shape[0]
            ox0, ox1 = x0 * UPSCALE, x0 * UPSCALE + sr_tile.shape[1]
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


# --- Main loop ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
imgs = sorted(INPUT_DIR.glob("*.png"))
print(f"Found {len(imgs)} images in {INPUT_DIR.name}/\n")

ok = failed = 0
for i, ip in enumerate(imgs, 1):
    op = OUT_DIR / ip.name
    try:
        pil = Image.open(ip).convert("RGB")
        H, W = pil.size[1], pil.size[0]
        mode = "direct" if max(H, W) <= DIRECT_MAX else "tiled"
        sr = sr_one(pil)
        sr.save(op)
        print(f"[{i:3d}/{len(imgs)}] {ip.name:20s} {mode:6s} {W}x{H} -> {sr.size[0]}x{sr.size[1]}")
        ok += 1
    except Exception as e:
        print(f"[{i:3d}/{len(imgs)}] {ip.name:20s} FAIL: {type(e).__name__}: {e}")
        failed += 1

print(f"\nDone. ok={ok}  failed={failed}  total={len(imgs)}")
print(f"Outputs in: {OUT_DIR}")