"""
D2 Step 1 — Controlled degradation of Makrai HR images.

Bicubic 4x downsample.
Uses cv2.resize with cv2.INTER_CUBIC (official OpenCV bicubic).

python scripts/04_degrade_makrai.py
"""

import os
import glob
import cv2

HR_DIR = "data/makrai/sample_20/images"
LR_DIR = "data/makrai/lr"
SCALE = 4                        # HR -> LR ratio (matches 4x SR upscale)
INTERPOLATION = cv2.INTER_CUBIC   # official OpenCV bicubic

os.makedirs(LR_DIR, exist_ok=True)

hrs = sorted(glob.glob(os.path.join(HR_DIR, "*.jpg")))
print(f"Found {len(hrs)} HR images in {HR_DIR}\n")

for i, hr_path in enumerate(hrs, 1):
    stem = os.path.splitext(os.path.basename(hr_path))[0]
    img = cv2.imread(hr_path)
    if img is None:
        print(f"[{i:2d}/{len(hrs)}] skip {stem}: unreadable")
        continue

    H, W = img.shape[:2]
    new_H, new_W = H // SCALE, W // SCALE
    lr = cv2.resize(img, (new_W, new_H), interpolation=INTERPOLATION)

    out_path = os.path.join(LR_DIR, f"{stem}.jpg")
    cv2.imwrite(out_path, lr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[{i:2d}/{len(hrs)}] {stem}  {W}x{H} -> {new_W}x{new_H}  saved {out_path}")

print(f"\nDone. LR images in {LR_DIR}/")
