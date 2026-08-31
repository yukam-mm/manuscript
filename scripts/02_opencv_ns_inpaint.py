import cv2
import glob
import os

RAW_DIR = "data/inhouse/raw"
MASK_DIR = "data/inhouse/masks_final"
OUT_DIR = "outputs/inpainting/navier_stokes"

os.makedirs(OUT_DIR, exist_ok=True)

for raw_path in glob.glob(os.path.join(RAW_DIR, "*.png")):
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    mask_path = os.path.join(MASK_DIR, f"{stem}_mask.png")

    if not os.path.exists(mask_path):
        print(f"skip {stem}: no mask found")
        continue

    img = cv2.imread(raw_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"skip {stem}: failed to decode image or mask")
        continue

    result = cv2.inpaint(img, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)

    out_path = os.path.join(OUT_DIR, f"{stem}.png")
    cv2.imwrite(out_path, result)
    print(f"saved {out_path}")
