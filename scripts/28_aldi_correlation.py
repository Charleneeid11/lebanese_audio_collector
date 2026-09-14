#!/usr/bin/env python3
"""
ROADMAP v2 Day 6 — ALDi (Arabic Level of Dialectness) correlation analysis.

For every GT item, scores its screening transcript with AMR-KELEG/ALDi
(a BERT-based regression model predicting Arabic dialectness in [0,1]).
Then correlates ALDi score with per-system prediction error across all 14 systems.

Outputs:
  data/aldi_scores.csv          — per-item ALDi score + per-system error
  data/aldi_correlation.json    — Spearman r per system, sorted by |r|

Appends FINDINGS Section 24 (idempotent).

Run:
  python scripts/28_aldi_correlation.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

ANNOTATIONS_CSV = Path("data/annotations.csv")
TRANSCRIPTS_DIR = Path("data/transcripts")
PREDS_DIR = Path("data/benchmark_predictions")
ALDI_CSV = Path("data/aldi_scores.csv")
ALDI_CORR_JSON = Path("data/aldi_correlation.json")
FINDINGS_PATH = Path("FINDINGS.md")

GT_LABEL_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}
GT_EXCLUDE = {"unclear", "skip", "", None}

ALDI_MODEL = "AMR-KELEG/ALDi"
BATCH_SIZE = 16
MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gt() -> list[dict]:
    items = []
    with open(ANNOTATIONS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("ground_truth") or "").strip().lower()
            if raw in GT_EXCLUDE:
                continue
            y = GT_LABEL_MAP.get(raw)
            if y is None:
                continue
            items.append({
                "item_id": int(row["item_id"]),
                "y": y,
                "platform": (row.get("platform") or "").strip().lower() or "unknown",
            })
    return items


def load_transcript_text(item_id: int) -> str | None:
    """Return concatenated screening chunk texts for item_id, or None."""
    path = TRANSCRIPTS_DIR / f"clip_{item_id}_screening.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = data.get("screening_samples", [])
        texts = [c.get("text", "").strip() for c in chunks if c.get("text")]
        return " ".join(texts) if texts else None
    except Exception:
        return None


def load_all_predictions(items: list[dict]) -> dict[str, dict[int, float]]:
    """Load all prediction files. Returns {system_name: {item_id: prob}}."""
    preds: dict[str, dict[int, float]] = {}
    valid_ids = {it["item_id"] for it in items}
    for path in sorted(PREDS_DIR.glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sys_name = path.stem
            preds[sys_name] = {
                int(r["item_id"]): float(r["prob"])
                for r in data.get("predictions", [])
                if int(r["item_id"]) in valid_ids
            }
        except Exception as e:
            print(f"  Warning: could not load {path.name}: {e}")
    return preds


def score_aldi(texts: list[str]) -> list[float]:
    """Score list of Arabic texts with AMR-KELEG/ALDi. Returns [0,1] scores."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"  Loading {ALDI_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(ALDI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ALDI_MODEL)
    model.eval()

    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)

        # ALDi outputs regression logits. If num_labels==1 the model was trained
        # with MSE loss and logits are direct predictions in [0,1]. If the range
        # falls outside [0,1] apply sigmoid.
        raw = logits.numpy()
        if raw.min() < -0.1 or raw.max() > 1.1:
            raw = 1.0 / (1.0 + np.exp(-raw))
        scores.extend(raw.tolist() if raw.ndim > 0 else [float(raw)])

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Scored {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")

    print()
    return scores


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(x) < 4:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def spearman_r_manual(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman r without scipy — rank then pearson."""
    if len(x) < 4:
        return float("nan")

    def rank(a: np.ndarray) -> np.ndarray:
        tmp = np.argsort(a)
        r = np.empty_like(tmp, dtype=float)
        r[tmp] = np.arange(len(a), dtype=float)
        return r

    rx, ry = rank(x), rank(y)
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Day 6: ALDi correlation analysis ===\n")

    # 1. Load GT
    items = load_gt()
    print(f"GT items: {len(items)}")

    # 2. Load transcripts for each item
    texts: list[str | None] = [load_transcript_text(it["item_id"]) for it in items]
    n_missing = sum(1 for t in texts if t is None)
    print(f"Transcripts found: {len(items) - n_missing}/{len(items)}")

    # 3. Check if ALDi scores already exist (resumable)
    if ALDI_CSV.exists():
        print(f"\nFound existing {ALDI_CSV} — loading cached scores.")
        cached: dict[int, float] = {}
        with open(ALDI_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                iid = int(row["item_id"])
                score_val = row.get("aldi_score", "")
                if score_val and score_val != "":
                    try:
                        cached[iid] = float(score_val)
                    except ValueError:
                        pass
        to_score_items = [it for it in items if it["item_id"] not in cached]
        print(f"Already scored: {len(cached)}, remaining: {len(to_score_items)}")
    else:
        cached = {}
        to_score_items = items

    # 4. Score with ALDi
    new_scores: dict[int, float] = {}
    if to_score_items:
        valid_pairs = [
            (it, texts[items.index(it)])
            for it in to_score_items
            if texts[items.index(it)] is not None
        ]
        if valid_pairs:
            scored_items, scored_texts = zip(*valid_pairs)
            raw_scores = score_aldi(list(scored_texts))
            for it, sc in zip(scored_items, raw_scores):
                new_scores[it["item_id"]] = sc

    aldi_scores = {**cached, **new_scores}
    print(f"\nTotal ALDi-scored items: {len(aldi_scores)}")

    # 5. Load all benchmark predictions
    print("Loading benchmark predictions ...")
    all_preds = load_all_predictions(items)
    system_names = sorted(all_preds.keys())
    print(f"Systems loaded: {len(system_names)}")

    # 6. Write aldi_scores.csv
    rows = []
    for it in items:
        iid = it["item_id"]
        aldi_val = aldi_scores.get(iid)
        row: dict = {
            "item_id": iid,
            "platform": it["platform"],
            "y": it["y"],
            "aldi_score": f"{aldi_val:.4f}" if aldi_val is not None else "",
        }
        for sys_name in system_names:
            prob = all_preds[sys_name].get(iid)
            if prob is not None and aldi_val is not None:
                row[f"err_{sys_name}"] = f"{abs(prob - it['y']):.4f}"
            else:
                row[f"err_{sys_name}"] = ""
        rows.append(row)

    fieldnames = ["item_id", "platform", "y", "aldi_score"] + [f"err_{s}" for s in system_names]
    with open(ALDI_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {ALDI_CSV}")

    # 7. Compute Spearman r(aldi_score, error) per system
    scored_ids = {iid for iid, s in aldi_scores.items() if s is not None}
    aldi_arr = np.array([aldi_scores[it["item_id"]] for it in items
                         if it["item_id"] in scored_ids])

    correlations = {}
    for sys_name in system_names:
        preds_sys = all_preds[sys_name]
        errors = []
        aldi_vals_sys = []
        for it in items:
            iid = it["item_id"]
            if iid not in scored_ids:
                continue
            prob = preds_sys.get(iid)
            if prob is None:
                continue
            errors.append(abs(prob - it["y"]))
            aldi_vals_sys.append(aldi_scores[iid])

        if len(errors) < 4:
            correlations[sys_name] = {"r": None, "n": len(errors)}
            continue

        r = spearman_r_manual(np.array(aldi_vals_sys), np.array(errors))
        correlations[sys_name] = {"r": round(r, 4), "n": len(errors)}

    # Sort by |r| descending
    sorted_corr = dict(
        sorted(correlations.items(),
               key=lambda kv: abs(kv[1]["r"]) if kv[1]["r"] is not None else 0,
               reverse=True)
    )
    ALDI_CORR_JSON.write_text(json.dumps(sorted_corr, indent=2), encoding="utf-8")
    print(f"Wrote {ALDI_CORR_JSON}")

    # Print summary table
    print("\nSpearman r (ALDi dialectness vs prediction error), |r| sorted:")
    print(f"  {'System':<40} {'r':>8}  {'n':>5}")
    print("  " + "-" * 56)
    for sys_name, val in sorted_corr.items():
        r_str = f"{val['r']:+.4f}" if val["r"] is not None else "  N/A "
        print(f"  {sys_name:<40} {r_str:>8}  {val['n']:>5}")

    # 8. Compute per-platform ALDi means
    print("\nMean ALDi score by platform and label:")
    platform_groups: dict[str, list] = {}
    for it in items:
        iid = it["item_id"]
        if iid not in scored_ids:
            continue
        key = f"{it['platform']}|{it['y']}"
        platform_groups.setdefault(key, []).append(aldi_scores[iid])

    for key, vals in sorted(platform_groups.items()):
        plat, label = key.split("|")
        label_name = "Lebanese" if label == "1" else "Non-Lebanese"
        print(f"  {plat:<15} {label_name:<15} mean={np.mean(vals):.3f}  n={len(vals)}")

    # 9. Append FINDINGS Section 24
    _append_findings(correlations, sorted_corr, aldi_scores, items, scored_ids, platform_groups)
    print("\nFINDINGS Section 24 appended.")


def _append_findings(
    correlations: dict,
    sorted_corr: dict,
    aldi_scores: dict,
    items: list[dict],
    scored_ids: set,
    platform_groups: dict,
) -> None:
    text = FINDINGS_PATH.read_text(encoding="utf-8")

    # Idempotent: strip prior Section 24 block
    import re
    text = re.sub(
        r"\n## Section 24[^\n]*\n.*?(?=\n## Section |\Z)",
        "",
        text,
        flags=re.DOTALL,
    )

    n_scored = len(scored_ids)
    n_total = len(items)

    # Best and worst correlated systems
    valid = [(k, v) for k, v in sorted_corr.items() if v["r"] is not None]
    most_pos = max(valid, key=lambda kv: kv[1]["r"]) if valid else None
    most_neg = min(valid, key=lambda kv: kv[1]["r"]) if valid else None

    # ALDi means per platform/label
    pg_lines = []
    for key, vals in sorted(platform_groups.items()):
        plat, label = key.split("|")
        label_name = "Lebanese" if label == "1" else "Non-Lebanese"
        pg_lines.append(f"  - {plat} / {label_name}: mean={np.mean(vals):.3f}, n={len(vals)}")

    # Correlation table
    corr_rows = []
    for sys_name, val in sorted_corr.items():
        r_str = f"{val['r']:+.4f}" if val["r"] is not None else "N/A"
        corr_rows.append(f"| {sys_name} | {r_str} | {val['n']} |")

    section = f"""

## Section 24 — ALDi Correlation Analysis (Day 6)

### 24.1 Setup

Model: `AMR-KELEG/ALDi` — BERT-based Arabic dialectness regressor outputting [0,1]
(1.0 = fully dialectal, 0.0 = fully MSA)

GT items scored: {n_scored}/{n_total}
Metric: Spearman r between per-item ALDi score and absolute prediction error |p − y|

### 24.2 ALDi scores by platform and label

{chr(10).join(pg_lines)}

### 24.3 Spearman r (ALDi dialectness vs prediction error)

| System | Spearman r | n |
|---|---|---|
{chr(10).join(corr_rows)}

Positive r means more dialectal text → higher prediction error for that system.
Negative r means more dialectal text → lower prediction error (system benefits from clear dialect signal).

### 24.4 Key findings

- Most positively correlated (r={most_pos[1]['r']:+.4f}): **{most_pos[0]}** — this system struggles more as dialect becomes stronger
- Most negatively correlated (r={most_neg[1]['r']:+.4f}): **{most_neg[0]}** — this system improves as dialect signal grows

### 24.5 Interpretation

ALDi measures how dialectal (vs MSA) a transcript is. Correlating this with per-system error
reveals whether systems fail on strongly dialectal speech (positive r) or on near-MSA items
(negative r). Text-based systems should show negative r (clearer dialect → easier classification);
acoustic systems may show different patterns if their errors are driven by recording domain
rather than linguistic content.
"""

    FINDINGS_PATH.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


if __name__ == "__main__":
    main()
