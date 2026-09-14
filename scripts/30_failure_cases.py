#!/usr/bin/env python3
"""
ROADMAP v2 Day 7 — Failure-case selection.

For each system in data/benchmark_predictions/, picks the top 10 items where
the model was most confidently wrong (highest |probability - true_label|).
Joins with transcripts so each failure case shows the actual Arabic text.

Outputs:
  data/failure_cases.csv   — per-(system, rank) with transcript + audio_path

Run:
  python scripts/30_failure_cases.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

ANNOTATIONS_CSV = Path("data/annotations.csv")
PREDS_DIR = Path("data/benchmark_predictions")
TRANSCRIPTS_DIR = Path("data/transcripts")
FAILURE_CSV = Path("data/failure_cases.csv")

GT_LABEL_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}
GT_EXCLUDE = {"unclear", "skip", "", None}
TOP_N = 10

FIELDNAMES = [
    "system", "rank", "item_id", "platform", "true_label", "true_y",
    "pred_prob", "error", "failure_type",
    "transcript_text", "audio_path",
]


def load_gt() -> dict[int, dict]:
    items = {}
    with open(ANNOTATIONS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("ground_truth") or "").strip().lower()
            if raw in GT_EXCLUDE:
                continue
            y = GT_LABEL_MAP.get(raw)
            if y is None:
                continue
            iid = int(row["item_id"])
            items[iid] = {
                "item_id": iid,
                "y": y,
                "raw_label": raw,
                "platform": (row.get("platform") or "").strip(),
                "audio_path": (row.get("audio_path") or "").strip(),
            }
    return items


def load_transcript_text(item_id: int) -> str:
    path = TRANSCRIPTS_DIR / f"clip_{item_id}_screening.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = data.get("screening_samples", [])
        texts = [c.get("text", "").strip() for c in chunks if c.get("text")]
        return " | ".join(texts)[:500]
    except Exception:
        return ""


def main() -> None:
    print("=== Day 7: Failure case selection ===\n")

    gt = load_gt()
    print(f"GT items: {len(gt)}")

    all_rows = []

    pred_files = sorted(f for f in PREDS_DIR.glob("*.json")
                        if not f.name.endswith(".partial.json"))
    print(f"Systems: {len(pred_files)}")

    for pred_path in pred_files:
        sys_name = pred_path.stem
        try:
            data = json.loads(pred_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Skipping {sys_name}: {e}")
            continue

        # Collect errors
        errors = []
        for rec in data.get("predictions", []):
            iid = int(rec["item_id"])
            if iid not in gt:
                continue
            prob = float(rec["prob"])
            y = gt[iid]["y"]
            err = abs(prob - y)
            errors.append((err, iid, prob, y))

        if not errors:
            continue

        # Sort by error descending, take top N
        errors.sort(key=lambda x: x[0], reverse=True)
        top = errors[:TOP_N]

        for rank, (err, iid, prob, y) in enumerate(top, start=1):
            item = gt[iid]
            # False positive: true=0, predicted high
            # False negative: true=1, predicted low
            if y == 0 and prob >= 0.5:
                failure_type = "false_positive"
            elif y == 1 and prob < 0.5:
                failure_type = "false_negative"
            else:
                # Confident but below 0.5 threshold — still high error on probability scale
                failure_type = "false_positive" if y == 0 else "false_negative"

            transcript = load_transcript_text(iid)
            all_rows.append({
                "system": sys_name,
                "rank": rank,
                "item_id": iid,
                "platform": item["platform"],
                "true_label": item["raw_label"],
                "true_y": y,
                "pred_prob": round(prob, 4),
                "error": round(err, 4),
                "failure_type": failure_type,
                "transcript_text": transcript,
                "audio_path": item["audio_path"],
            })

        print(f"  {sys_name}: top-{TOP_N} errors range [{top[-1][0]:.3f}, {top[0][0]:.3f}]")

    # Write CSV
    with open(FAILURE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {FAILURE_CSV}")
    print(f"({len(pred_files)} systems × top {TOP_N} each)")

    # Quick summary: most common failure items across systems
    from collections import Counter
    fp_counts = Counter(
        r["item_id"] for r in all_rows if r["failure_type"] == "false_positive"
    )
    fn_counts = Counter(
        r["item_id"] for r in all_rows if r["failure_type"] == "false_negative"
    )
    def safe_print(s: str) -> None:
        print(s.encode("ascii", errors="replace").decode("ascii"))

    safe_print("\nTop 5 items appearing most often as false positives (not-Lebanese called Lebanese):")
    for iid, cnt in fp_counts.most_common(5):
        item = gt[iid]
        snippet = load_transcript_text(iid)[:80]
        safe_print(f"  id={iid} ({item['platform']}) -- {cnt} systems -- '{snippet}'")

    safe_print("\nTop 5 items appearing most often as false negatives (Lebanese missed):")
    for iid, cnt in fn_counts.most_common(5):
        item = gt[iid]
        snippet = load_transcript_text(iid)[:80]
        safe_print(f"  id={iid} ({item['platform']}) -- {cnt} systems -- '{snippet}'")


if __name__ == "__main__":
    main()
