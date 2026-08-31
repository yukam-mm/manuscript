"""
Downstream evaluation for the IN-HOUSE dataset: count error of the k-means
colony counter vs. manual counts.

Metric implementations: scikit-learn (the official reference implementations)
    - mean_absolute_error, mean_absolute_percentage_error, r2_score
Bias and RMSE are computed with numpy in the standard textbook form.

Ground truth : data/inhouse/ground_truth/manual_counts.csv        (image_name, manual_count)
Predictions  : outputs/colony_counts_inhouse/counts_summary.csv   (stem, method, n_circles, ...)

Restricted to the 20 stratified stems used by 09b_batch_kmeans_counter_inhouse.py.

Output: metrics/downstream/inhouse_count_error.csv
    columns: method, n, MAE, RMSE, MAPE_pct, bias_mean, bias_std, R2
Plus a per-image breakdown to metrics/downstream/inhouse_count_error_per_image.csv.

Usage (from project root):
    pip install scikit-learn numpy
    python scripts/10b_count_error_inhouse.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

PROJECT     = Path(__file__).resolve().parent.parent
GT_CSV      = PROJECT / "data" / "inhouse" / "ground_truth" / "manual_counts.csv"
PRED_CSV    = PROJECT / "outputs" / "colony_counts_inhouse" / "counts_summary.csv"
OUT_DIR     = PROJECT / "metrics" / "downstream"
OUT_CSV     = OUT_DIR / "inhouse_count_error.csv"
OUT_PER     = OUT_DIR / "inhouse_count_error_per_image.csv"

STEMS_20 = {
    "sc_h_12","sc_h_41","sc_h_15","sc_33_19","sc_33_17",
    "sc_33_24","sc_33_35","sc_h_11","sc_h_28","sc_33_18",
    "sc_h_39","sc_h_13","sc_33_20","sc_33_16","sc_h_27",
    "sc_33_1","sc_h_32","sc_33_40","sc_33_30","sc_h_42",
}
METHODS = ["raw", "swin2sr", "aesrgan"]


def load_gt():
    """Return {stem: manual_count} for the 20 chosen stems only."""
    gt = {}
    with open(GT_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            stem = row["image_name"].strip().replace(".png", "")
            if stem not in STEMS_20:
                continue
            val = row["manual_count"].strip()
            if val == "-" or val == "":
                continue
            try:
                gt[stem] = int(val)
            except ValueError:
                continue
    return gt


def load_predictions():
    """Return {(method, stem): n_circles}."""
    preds = {}
    with open(PRED_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            preds[(row["method"], row["stem"])] = int(row["n_circles"])
    return preds


def evaluate_method(method, gt, preds):
    y_true, y_pred, per_image = [], [], []
    for stem, gt_count in gt.items():
        pred = preds.get((method, stem))
        if pred is None:
            continue
        y_true.append(gt_count)
        y_pred.append(pred)
        per_image.append({
            "method": method, "stem": stem,
            "manual": gt_count, "predicted": pred,
            "error": pred - gt_count,
            "abs_error": abs(pred - gt_count),
        })

    if not y_true:
        return None, per_image

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAPE: sklearn's implementation divides by max(|y_true|, epsilon), which
    # blows up when GT==0. Filter GT==0 for MAPE only.
    mask = y_true != 0
    mape = (mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
            if mask.any() else float("nan"))

    return {
        "method":     method,
        "n":          len(y_true),
        "MAE":        float(mean_absolute_error(y_true, y_pred)),
        "RMSE":       float(np.sqrt(((y_pred - y_true) ** 2).mean())),
        "MAPE_pct":   float(mape),
        "bias_mean":  float((y_pred - y_true).mean()),   # + = overcount
        "bias_std":   float((y_pred - y_true).std()),
        "R2":         float(r2_score(y_true, y_pred)),
    }, per_image


def main():
    gt = load_gt()
    print(f"Loaded {len(gt)} in-house GT counts (of {len(STEMS_20)} target stems)")
    if len(gt) < len(STEMS_20):
        missing = STEMS_20 - set(gt)
        print(f"  missing: {sorted(missing)}")

    preds = load_predictions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    all_per_image = []
    for m in METHODS:
        summary, per_img = evaluate_method(m, gt, preds)
        all_per_image.extend(per_img)
        if summary is None:
            print(f"\n{m}: no predictions matched")
            continue
        rows.append(summary)
        print(f"\n=== {m} ===")
        print(f"  n           = {summary['n']}")
        print(f"  MAE         = {summary['MAE']:.3f}")
        print(f"  RMSE        = {summary['RMSE']:.3f}")
        print(f"  MAPE %      = {summary['MAPE_pct']:.2f}")
        print(f"  bias (mean) = {summary['bias_mean']:+.3f} (+ overcount, - undercount)")
        print(f"  bias  std   = {summary['bias_std']:.3f}")
        print(f"  R^2         = {summary['R2']:+.4f}")

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
