import os
import sys
import csv
import glob
from pathlib import Path

import cv2
import numpy as np


COUNTER_DIR = Path(__file__).resolve().parent.parent / "colony_counter" / "stage2_colony_final"
sys.path.insert(0, str(COUNTER_DIR))
import test_kmeans as tk   # noqa: E402   -> official module, unchanged



PROJECT = Path(__file__).resolve().parent.parent
HR_DIR  = PROJECT / "data" / "makrai" / "sample_20" / "images"
METHODS = {
    "raw":     HR_DIR,
    "swin2sr": PROJECT / "outputs" / "sr_makrai" / "swin2sr",
    "aesrgan": PROJECT / "outputs" / "sr_makrai" / "aesrgan",
}
OUT_ROOT = PROJECT / "outputs" / "colony_counts"



def crop_circle_silent(img, center, radius):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)
    x, y = center
    x1 = max(x - radius, 0)
    y1 = max(y - radius, 0)
    x2 = min(x + radius, img.shape[1])
    y2 = min(y + radius, img.shape[0])
    return result[y1:y2, x1:x2]


def find_contours_silent(cropped_img):
    if cropped_img is None:
        return None
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    return cnts



def detect_plate_boundary(bgr):
    """
    Return (center, radius) of the dish.
    Try HoughCircles on a downscaled gray image first; on failure fall back
    to centered crop with radius = 0.48 * min(H, W).
    """
    H, W = bgr.shape[:2]
    scale = 800.0 / max(H, W)
    small = cv2.resize(bgr, (int(W * scale), int(H * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 11)
    sh, sw = gray.shape[:2]

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=min(sh, sw),
        param1=100, param2=30,
        minRadius=int(min(sh, sw) * 0.30),
        maxRadius=int(min(sh, sw) * 0.55),
    )
    if circles is not None:
        x, y, r = circles[0][0]
        inv = 1.0 / scale if scale < 1 else 1.0
        return (int(x * inv), int(y * inv)), int(r * inv)

    # Fallback: centered crop
    return (W // 2, H // 2), int(min(H, W) * 0.48)



def find_image(dir_path, stem):
    for suffix in ("", "_out"):
        for ext in (".jpg", ".png", ".jpeg"):
            p = dir_path / f"{stem}{suffix}{ext}"
            if p.exists():
                return p
    return None


def run_one(img_path, out_csv, out_vis):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    center, radius = detect_plate_boundary(img)
    cropped = crop_circle_silent(img, center, radius)
    if cropped is None or cropped.size == 0:
        return None

    contours = find_contours_silent(cropped)
    if not contours or len(contours) < 3:
        # Save empty
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["x_full", "y_full", "r"])
        cv2.imwrite(str(out_vis), img)
        return {"n_contours": len(contours) if contours else 0, "n_circles": 0,
                "cx": center[0], "cy": center[1], "R_plate": radius}

    # Cluster (identical to test_kmeans.main)
    n_clusters = 4
    cluster_labels, cluster_info, _ = tk.cluster_contours_kmeans(
        contours, n_clusters=n_clusters
    )
    selected = tk.select_best_clusters(cluster_info, n_clusters=3)
    all_circles, _, _ = tk.detect_circles_with_params_from_clusters(
        cropped, cluster_info, selected
    )

    # Coordinates in `all_circles` are in the cropped-image frame.
    # Translate back to full-image coordinates: crop origin = (center-radius).
    ox = max(center[0] - radius, 0)
    oy = max(center[1] - radius, 0)

    # Save circles CSV in full-image coords
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_full", "y_full", "r"])
        for (x, y, r) in all_circles:
            w.writerow([x + ox, y + oy, r])

    # Save annotated visualization on the full image
    vis = img.copy()
    for (x, y, r) in all_circles:
        cv2.circle(vis, (x + ox, y + oy), r, (0, 255, 0), 3)
        cv2.circle(vis, (x + ox, y + oy), 4, (0, 255, 0), -1)
    cv2.circle(vis, center, radius, (0, 0, 255), 4)
    cv2.putText(vis, f"count={len(all_circles)}", (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    cv2.imwrite(str(out_vis), vis)

    return {"n_contours": len(contours), "n_circles": len(all_circles),
            "cx": center[0], "cy": center[1], "R_plate": radius}



def main():
    hrs = sorted(HR_DIR.glob("*.jpg"))
    stems = [p.stem for p in hrs]
    print(f"{len(stems)} images x {len(METHODS)} conditions = "
          f"{len(stems) * len(METHODS)} runs\n")

    summary_rows = []
    for method, d in METHODS.items():
        out_dir = OUT_ROOT / method
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {method} ===  ({d})")
        for i, stem in enumerate(stems, 1):
            ip = find_image(d, stem)
            if ip is None:
                print(f"[{i:2d}/{len(stems)}] {stem:16s} SKIP (missing)")
                continue
            out_csv = out_dir / f"{stem}_circles.csv"
            out_vis = out_dir / f"{stem}_vis.png"
            try:
                res = run_one(ip, out_csv, out_vis)
                if res is None:
                    print(f"[{i:2d}/{len(stems)}] {stem:16s} FAIL (unreadable)")
                    continue
                summary_rows.append({"stem": stem, "method": method, **res})
                print(f"[{i:2d}/{len(stems)}] {stem:16s} "
                      f"contours={res['n_contours']:4d}  "
                      f"circles={res['n_circles']:4d}  "
                      f"plate=({res['cx']},{res['cy']}) R={res['R_plate']}")
            except Exception as e:
                print(f"[{i:2d}/{len(stems)}] {stem:16s} FAIL: {type(e).__name__}: {e}")

    if summary_rows:
        summary_csv = OUT_ROOT / "counts_summary.csv"
        fields = ["stem", "method", "n_contours", "n_circles", "cx", "cy", "R_plate"]
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote summary to {summary_csv}")


if __name__ == "__main__":
    main()
