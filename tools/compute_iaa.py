#!/usr/bin/env python3
"""Compute inter-annotator agreement between two annotation CSV files.

Usage:
    python tools/compute_iaa.py \
        --a1 data/annotations.csv \
        --a2 data/annotations_a2.csv

Outputs:
  - Cohen's kappa on the 5-way label (lebanese / mostly_lebanese / not_lebanese / unclear / skip)
  - Cohen's kappa on the binary label (positive = lebanese + mostly_lebanese; negative = not_lebanese)
  - Confusion matrix between the two annotators (5-way)
  - Per-label agreement breakdown
  - Items where annotators disagree (saved to data/iaa_disagreements.csv for review)
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


POSITIVE_LABELS = {"lebanese", "mostly_lebanese"}
NEGATIVE_LABELS = {"not_lebanese"}
EXCLUDE_LABELS  = {"unclear", "skip"}

LABEL_ORDER = ["lebanese", "mostly_lebanese", "not_lebanese", "unclear", "skip"]


def load_csv(path: Path) -> dict:
    """Load annotations CSV, return {item_id: ground_truth}."""
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item_id = int(row["item_id"])
            label   = row["ground_truth"].strip()
            out[item_id] = label
    return out


def cohen_kappa(labels_a: list, labels_b: list, categories: list) -> float:
    n = len(labels_a)
    if n == 0:
        return float("nan")
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        if a in cat_idx and b in cat_idx:
            conf[cat_idx[a]][cat_idx[b]] += 1
    observed = sum(conf[i][i] for i in range(k)) / n
    row_sums = [sum(conf[i]) for i in range(k)]
    col_sums = [sum(conf[r][i] for r in range(k)) for i in range(k)]
    expected = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def print_confusion(labels_a: list, labels_b: list, categories: list, name_a: str, name_b: str):
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        if a in cat_idx and b in cat_idx:
            conf[cat_idx[a]][cat_idx[b]] += 1
    col_w = max(len(c) for c in categories) + 2
    header = f"{'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in categories)
    print(f"\nConfusion matrix (rows = {name_a}, cols = {name_b}):")
    print(header)
    for i, c in enumerate(categories):
        row = f"{c:>{col_w}}" + "".join(f"{conf[i][j]:>{col_w}}" for j in range(k))
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", default="data/annotations.csv",
                        help="First annotator CSV (default: data/annotations.csv)")
    parser.add_argument("--a2", default="data/annotations_a2.csv",
                        help="Second annotator CSV (default: data/annotations_a2.csv)")
    parser.add_argument("--disagreements-out", default="data/iaa_disagreements.csv",
                        help="Where to save disagreement rows")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path_a1 = root / args.a1 if not Path(args.a1).is_absolute() else Path(args.a1)
    path_a2 = root / args.a2 if not Path(args.a2).is_absolute() else Path(args.a2)

    if not path_a1.exists():
        sys.exit(f"File not found: {path_a1}")
    if not path_a2.exists():
        sys.exit(f"File not found: {path_a2}")

    ann1 = load_csv(path_a1)
    ann2 = load_csv(path_a2)

    common_ids = sorted(set(ann1) & set(ann2))
    only_in_1  = sorted(set(ann1) - set(ann2))
    only_in_2  = sorted(set(ann2) - set(ann1))

    print(f"Annotator 1 ({path_a1.name}): {len(ann1)} items")
    print(f"Annotator 2 ({path_a2.name}): {len(ann2)} items")
    print(f"Common items: {len(common_ids)}")
    if only_in_1:
        print(f"Only in A1 (not yet labeled by A2): {len(only_in_1)} items")
    if only_in_2:
        print(f"Only in A2: {len(only_in_2)} items")

    if not common_ids:
        sys.exit("No common items to compare.")

    labels1 = [ann1[i] for i in common_ids]
    labels2 = [ann2[i] for i in common_ids]

    # ── 5-way kappa ────────────────────────────────────────────────────────
    kappa_5 = cohen_kappa(labels1, labels2, LABEL_ORDER)
    print(f"\n{'─'*50}")
    print(f"5-way Cohen's kappa : {kappa_5:.4f}")

    # ── Binary kappa (exclude unclear/skip from both) ───────────────────────
    bin1, bin2 = [], []
    excluded_count = 0
    for a, b in zip(labels1, labels2):
        if a in EXCLUDE_LABELS or b in EXCLUDE_LABELS:
            excluded_count += 1
            continue
        bin1.append("positive" if a in POSITIVE_LABELS else "negative")
        bin2.append("positive" if b in POSITIVE_LABELS else "negative")

    kappa_bin = cohen_kappa(bin1, bin2, ["positive", "negative"])
    print(f"Binary Cohen's kappa: {kappa_bin:.4f}  "
          f"(excluding {excluded_count} items where either annotator said unclear/skip)")

    # Kappa interpretation
    def kappa_label(k):
        if k < 0:    return "poor (worse than chance)"
        if k < 0.20: return "slight"
        if k < 0.40: return "fair"
        if k < 0.60: return "moderate"
        if k < 0.80: return "substantial"
        return "almost perfect"

    print(f"  → 5-way: {kappa_label(kappa_5)}")
    print(f"  → binary: {kappa_label(kappa_bin)}")

    # ── Confusion matrix ───────────────────────────────────────────────────
    print_confusion(labels1, labels2, LABEL_ORDER,
                    path_a1.name, path_a2.name)

    # ── Agreement breakdown ────────────────────────────────────────────────
    agree   = sum(a == b for a, b in zip(labels1, labels2))
    disagree= len(common_ids) - agree
    print(f"\nExact agreement  : {agree}/{len(common_ids)} = {100*agree/len(common_ids):.1f}%")
    print(f"Disagreements    : {disagree}")

    # Disagreement breakdown by type
    dis_types = Counter()
    disagree_rows = []
    for item_id, a, b in zip(common_ids, labels1, labels2):
        if a != b:
            dis_types[f"{a} vs {b}"] += 1
            disagree_rows.append({"item_id": item_id, "a1": a, "a2": b})

    print("\nTop disagreement patterns:")
    for pair, count in dis_types.most_common(10):
        print(f"  {pair}: {count}")

    # ── Save disagreements ─────────────────────────────────────────────────
    dis_path = root / args.disagreements_out
    with dis_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "a1", "a2"])
        writer.writeheader()
        writer.writerows(disagree_rows)
    print(f"\nDisagreement rows saved to: {dis_path}")
    print("Open these items in the annotation tool to adjudicate.")


if __name__ == "__main__":
    main()
