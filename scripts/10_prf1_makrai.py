"""
Downstream evaluation for the Makrai dataset: Precision / Recall / F1
of the k-means colony counter against the COCO bbox ground truth.

Metric implementation: pycocotools (the official COCO API, github.com/cocodataset/cocoapi)
    - Precision and Recall are extracted from COCOeval.eval['precision'] and
      COCOeval.eval['recall'] at IoU = 0.5, area = 'all', maxDets = 1000.
    - F1 is computed as the standard harmonic mean 2PR/(P+R).
No detection logic is re-implemented here; this only converts the counter's
circle outputs to COCO-format bboxes and calls the official evaluator.

Ground truth : data/makrai/sample_20/annot_COCO_sample20.json     (2991 bboxes, 20 imgs)
Predictions  : outputs/colony_counts/<method>/<stem>_circles.csv  (x_full,y_full,r)

For each detected circle (x, y, r) we emit a COCO bbox:
    [x - r, y - r, 2r, 2r]           (xywh, tight square around the circle)
All detections in an image get the image's own category_id — each Makrai
plate is a single-species plate, so this is unambiguous. All detections
get score = 1.0 (the counter has no confidence output).

Output: metrics/downstream/makrai_prf1.csv
    columns: method, iou, precision, recall, f1,
             n_gt, n_pred, TP_est, FP_est, FN_est
Also writes a per-image breakdown to metrics/downstream/makrai_prf1_per_image.csv.

Usage (from project root):
    pip install pycocotools numpy
    python scripts/10_prf1_makrai.py
"""

import os
import csv
import json
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT   = Path(__file__).resolve().parent.parent
GT_JSON   = PROJECT / "data" / "makrai" / "sample_20" / "annot_COCO_sample20.json"
PRED_ROOT = PROJECT / "outputs" / "colony_counts"
METHODS   = ["raw", "swin2sr", "aesrgan"]
OUT_DIR   = PROJECT / "metrics" / "downstream"
OUT_CSV   = OUT_DIR / "makrai_prf1.csv"
OUT_PER   = OUT_DIR / "makrai_prf1_per_image.csv"

IOU_THR = 0.5          # standard object-detection threshold
MAX_DETS = 1000        # dense plates (up to 479 bboxes) — raise past default 100


def load_predictions(method: str, coco_gt: COCO):
    """Convert per-image circle CSVs into a COCO detection list."""
    dets = []
    for img_id in coco_gt.getImgIds():
        info = coco_gt.loadImgs(img_id)[0]
        stem = Path(info["file_name"]).stem
        # every image in Makrai has a single species -> single category
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns    = coco_gt.loadAnns(ann_ids)
        cat_id  = anns[0]["category_id"] if anns else 4  # default sp04

        csv_path = PRED_ROOT / method / f"{stem}_circles.csv"
        if not csv_path.exists():
            print(f"  [{method}] {stem}: no predictions CSV")
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row["x_full"]); y = float(row["y_full"]); r = float(row["r"])
                dets.append({
                    "image_id":    int(img_id),
                    "category_id": int(cat_id),
                    "bbox":        [x - r, y - r, 2 * r, 2 * r],
                    "score":       1.0,
                })
    return dets


def evaluate_one_method(method: str, coco_gt: COCO):
    dets = load_predictions(method, coco_gt)
    if not dets:
        return None, None

    # COCOeval requires a JSON file on disk
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dets, f)
        det_json = f.name

    coco_dt = coco_gt.loadRes(det_json)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Override iouThrs so index 0 = the threshold we care about
    ev.params.iouThrs = np.array([IOU_THR])
    ev.params.maxDets = [1, 10, MAX_DETS]
    ev.evaluate()
    ev.accumulate()
    # No summarize() — its stdout is only informational

    # precision shape: [T, R, K, A, M] = [iou, recall_lvls, cats, area, maxdet]
    # recall    shape: [T, K, A, M]
    prec = ev.eval["precision"][0, :, :, 0, -1]   # (R, K)
    rec  = ev.eval["recall"][0, :, 0, -1]          # (K,)

    prec = prec[prec > -1]
    P = float(prec.mean()) if prec.size else 0.0
    rec  = rec[rec > -1]
    R = float(rec.mean()) if rec.size else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    # Rough per-method TP/FP/FN totals from IoU matching (indicative, not COCO-official)
    tp = fp = fn = 0
    per_image_rows = []
    for img_id in coco_gt.getImgIds():
        info = coco_gt.loadImgs(img_id)[0]
        stem = Path(info["file_name"]).stem
        gt_boxes = [a["bbox"] for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))]
        pd_boxes = [d["bbox"] for d in dets if d["image_id"] == img_id]

        t, p, f = _greedy_match(gt_boxes, pd_boxes, IOU_THR)
        tp += t; fp += p; fn += f
        img_P = t / (t + p) if (t + p) else 0.0
        img_R = t / (t + f) if (t + f) else 0.0
        img_F = 2 * img_P * img_R / (img_P + img_R) if (img_P + img_R) else 0.0
        per_image_rows.append({
            "method": method, "stem": stem,
            "n_gt": len(gt_boxes), "n_pred": len(pd_boxes),
            "TP": t, "FP": p, "FN": f,
            "precision": img_P, "recall": img_R, "f1": img_F,
        })

    os.unlink(det_json)
    return {
        "method": method,
        "iou": IOU_THR,
        "precision": P, "recall": R, "f1": F1,
        "n_gt": int(sum(1 for _ in coco_gt.getAnnIds())),
        "n_pred": len(dets),
        "TP_est": tp, "FP_est": fp, "FN_est": fn,
    }, per_image_rows


def _iou_xywh(a, b):
    ax0, ay0, aw, ah = a; ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = b; bx1, by1 = bx0 + bw, by0 + bh
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(gts, preds, iou_thr):
    if not gts:
        return 0, len(preds), 0
    if not preds:
        return 0, 0, len(gts)
    ious = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            ious[i, j] = _iou_xywh(p, g)
    tp = 0
    matched_gt = set()
    order = np.argsort(-ious.max(axis=1))
    for i in order:
        j = int(ious[i].argmax())
        if ious[i, j] >= iou_thr and j not in matched_gt:
            matched_gt.add(j); tp += 1
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def main():
    coco_gt = COCO(str(GT_JSON))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    all_per_image = []
    for m in METHODS:
        print(f"\n=== {m} ===")
        summary, per_img = evaluate_one_method(m, coco_gt)
        if summary is None:
            print(f"  no predictions for {m}, skipping")
            continue
        rows.append(summary)
        all_per_image.extend(per_img)
        print(f"  P={summary['precision']:.4f}  "
              f"R={summary['recall']:.4f}  "
              f"F1={summary['f1']:.4f}  "
              f"TP={summary['TP_est']:5d}  FP={summary['FP_est']:5d}  FN={summary['FN_est']:5d}")

    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {OUT_CSV}")

    if all_per_image:
        with open(OUT_PER, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_per_image[0].keys()))
            w.writeheader(); w.writerows(all_per_image)
        print(f"Wrote {OUT_PER}")


if __name__ == "__main__":
    main()
