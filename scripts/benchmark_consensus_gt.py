#!/usr/bin/env python3
"""Re-compute benchmark metrics on the consensus ground truth.

Consensus GT: items where BOTH annotators agree on the binary label
(Lebanese+MostlyLB=1, NotLebanese=0).  Items where annotators disagree
on the binary class are excluded from this evaluation.

Usage:
    python scripts/benchmark_consensus_gt.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANN1_PATH = PROJECT_ROOT / "data" / "annotations.csv"
ANN2_PATH = PROJECT_ROOT / "data" / "annotations_a2.csv"
PRED_DIR  = PROJECT_ROOT / "data" / "benchmark_predictions"
OUT_PATH  = PROJECT_ROOT / "data" / "benchmark_results_consensus.csv"

POSITIVE_LABELS = {"lebanese", "mostly_lebanese"}
NEGATIVE_LABELS = {"not_lebanese"}


def load_annotations(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["item_id"])] = row["ground_truth"].strip()
    return out


def binary_label(label: str):
    if label in POSITIVE_LABELS:
        return 1
    if label in NEGATIVE_LABELS:
        return 0
    return None  # unclear / skip


def bootstrap_ci(y_true, y_score, metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        yt, ys = y_true[s], y_score[s]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric_fn(yt, ys))
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    lo = np.percentile(vals, 2.5)
    hi = np.percentile(vals, 97.5)
    return lo, hi


def main():
    ann1 = load_annotations(ANN1_PATH)
    ann2 = load_annotations(ANN2_PATH)

    common = sorted(set(ann1) & set(ann2))
    print(f"Items in both annotation files: {len(common)}")

    # Build consensus GT
    consensus = {}   # item_id -> binary label (0/1)
    disagreed = 0
    excluded_unclear = 0
    for iid in common:
        b1 = binary_label(ann1[iid])
        b2 = binary_label(ann2[iid])
        if b1 is None or b2 is None:
            excluded_unclear += 1
            continue
        if b1 == b2:
            consensus[iid] = b1
        else:
            disagreed += 1

    n_pos = sum(v for v in consensus.values())
    n_neg = sum(1 - v for v in consensus.values())
    print(f"Binary-agreed consensus GT: {len(consensus)} items "
          f"(positive={n_pos}, negative={n_neg})")
    print(f"Excluded — binary disagreement: {disagreed}, unclear/skip: {excluded_unclear}")
    print()

    # Save consensus annotations CSV
    cons_path = PROJECT_ROOT / "data" / "annotations_consensus.csv"
    with cons_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "ground_truth", "ground_truth_a2"])
        w.writeheader()
        for iid in sorted(consensus):
            w.writerow({
                "item_id": iid,
                "ground_truth": ann1[iid],
                "ground_truth_a2": ann2[iid],
            })
    print(f"Saved consensus GT to {cons_path}")
    print()

    # Load all prediction files
    pred_files = sorted(PRED_DIR.glob("*.json"))
    pred_files = [p for p in pred_files if not p.name.endswith(".partial.json")]

    rows = []
    print(f"{'System':<45} {'AUC':>6}  {'CI_lo':>6}  {'CI_hi':>6}  {'F1@0.5':>7}  n")
    print("-" * 90)

    for pf in pred_files:
        data = json.loads(pf.read_text(encoding="utf-8"))
        preds = data.get("predictions", [])
        if not preds:
            continue

        y_true_list, y_score_list = [], []
        for p in preds:
            iid = int(p["item_id"])
            if iid not in consensus:
                continue
            prob = float(p["prob"])
            y_true_list.append(consensus[iid])
            y_score_list.append(prob)

        if len(set(y_true_list)) < 2 or len(y_true_list) < 10:
            continue

        y_true  = np.array(y_true_list)
        y_score = np.array(y_score_list)
        y_pred  = (y_score >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_score)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ci_lo, ci_hi = bootstrap_ci(y_true, y_score,
                                     lambda yt, ys: roc_auc_score(yt, ys))

        system = data.get("system", pf.stem)
        print(f"{system:<45} {auc:.4f}  {ci_lo:.4f}  {ci_hi:.4f}  {f1:.4f}  {len(y_true)}")
        rows.append({
            "system": system,
            "gt": "consensus",
            "n": len(y_true),
            "roc_auc": round(auc, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "macro_f1_0p5": round(f1, 4),
        })

    # Save CSV
    if rows:
        rows.sort(key=lambda r: -r["roc_auc"])
        fieldnames = ["system", "gt", "n", "roc_auc", "ci_lo", "ci_hi", "macro_f1_0p5"]
        with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
