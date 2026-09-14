#!/usr/bin/env python3
"""
ROADMAP v2 Day 5 — Per-platform breakdown of benchmark predictions.

For every system in data/benchmark_predictions/, computes macro F1 and ROC-AUC
restricted to each platform subset of the GT (podcast_rss, youtube, adi17, fleurs).
Also computes code-switching density per GT item from screening transcripts.

Outputs:
  data/benchmark_per_platform.csv    — per-(system, platform) metrics
  data/code_switching_features.csv   — per-GT-item code-switching features

Appends FINDINGS Section 22 (idempotent).
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

PREDS_DIR = Path("data/benchmark_predictions")
ANNOTATIONS_CSV = Path("data/annotations.csv")
QUEUE_DB = Path("data/queue.db")
TRANSCRIPTS_DIR = Path("data/transcripts")
PER_PLATFORM_CSV = Path("data/benchmark_per_platform.csv")
CODE_SWITCH_CSV = Path("data/code_switching_features.csv")
FINDINGS_PATH = Path("FINDINGS.md")

GT_LABEL_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gt() -> list[dict]:
    """Load GT annotations joined with platform info from queue.db."""
    if not ANNOTATIONS_CSV.exists():
        raise FileNotFoundError(f"GT not found: {ANNOTATIONS_CSV}")

    # Load annotations
    annotations: dict[int, dict] = {}
    with ANNOTATIONS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = int(row["item_id"])
            label = row.get("ground_truth", "").strip().lower()
            if label not in GT_LABEL_MAP:
                continue
            annotations[iid] = {"item_id": iid, "y": GT_LABEL_MAP[label], "label": label}

    # Join platform from queue.db
    if QUEUE_DB.exists():
        conn = sqlite3.connect(QUEUE_DB)
        cur = conn.cursor()
        ids_str = ",".join(str(i) for i in annotations.keys())
        rows = cur.execute(
            f"SELECT id, platform, status FROM queue WHERE id IN ({ids_str})"
        ).fetchall()
        conn.close()
        for iid, platform, status in rows:
            if iid in annotations:
                annotations[iid]["platform"] = platform
                annotations[iid]["source_status"] = status
    else:
        # Fallback: read platform from annotations CSV if present
        with ANNOTATIONS_CSV.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = int(row["item_id"])
                if iid in annotations and "platform" not in annotations[iid]:
                    annotations[iid]["platform"] = row.get("platform", "unknown")

    return list(annotations.values())


def load_predictions(system_name: str) -> dict[int, float]:
    path = PREDS_DIR / f"{system_name}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "predictions" in data:
        preds = {int(p["item_id"]): float(p["prob"]) for p in data["predictions"]}
    else:
        preds = {int(k): float(v) for k, v in data.items()}
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _bootstrap_auc(y_true, y_score, n=500, seed=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = np.array(y_true)[idx], np.array(y_score)[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, ys))
        except Exception:
            pass
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_metrics(y_true, y_score) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"n": len(y_true), "roc_auc": float("nan"), "macro_f1_05": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    auc = roc_auc_score(y_true, y_score)
    preds = [1 if s >= 0.5 else 0 for s in y_score]
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    lo, hi = _bootstrap_auc(y_true, y_score)
    return {"n": len(y_true), "roc_auc": round(auc, 4), "macro_f1_05": round(f1, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


# ---------------------------------------------------------------------------
# Code-switching density
# ---------------------------------------------------------------------------

# Regex for Latin-script tokens inside Arabic text (French/English words)
_LATIN_RE = re.compile(r"\b[A-Za-z]{2,}\b")


def compute_code_switch_density(item_id: int) -> dict:
    tpath = TRANSCRIPTS_DIR / f"clip_{item_id}_screening.json"
    if not tpath.exists():
        return {"item_id": item_id, "total_words": 0, "latin_words": 0,
                "cs_density": 0.0, "has_transcript": False}
    try:
        data = json.loads(tpath.read_text(encoding="utf-8"))
        full_text = " ".join(
            s.get("text", "") for s in (data.get("screening_samples") or [])
        ).strip()
    except Exception:
        full_text = ""

    words = full_text.split()
    latin = _LATIN_RE.findall(full_text)
    density = len(latin) / max(len(words), 1)
    return {"item_id": item_id, "total_words": len(words), "latin_words": len(latin),
            "cs_density": round(density, 4), "has_transcript": bool(full_text)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[load] GT items ...", flush=True)
    gt_items = load_gt()
    print(f"  {len(gt_items)} evaluable items", flush=True)

    # Determine which systems have completed predictions
    system_names = sorted(
        p.stem for p in PREDS_DIR.glob("*.json")
        if not p.stem.endswith(".partial")
    )
    print(f"[systems] {len(system_names)}: {system_names}", flush=True)

    # Platforms present in GT
    all_platforms = sorted(set(it.get("platform", "unknown") for it in gt_items))
    print(f"[platforms] {all_platforms}", flush=True)

    # --- Per-platform breakdown ---
    rows = []
    for sname in system_names:
        preds = load_predictions(sname)
        if not preds:
            print(f"  [skip] {sname}: no predictions", flush=True)
            continue

        # Overall
        matched = [(it, preds[it["item_id"]]) for it in gt_items if it["item_id"] in preds]
        if not matched:
            continue
        y_true_all = [m[0]["y"] for m in matched]
        y_score_all = [m[1] for m in matched]
        overall = compute_metrics(y_true_all, y_score_all)
        rows.append({"system": sname, "platform": "ALL", **overall})

        # Per platform
        for plat in all_platforms:
            sub = [(it, preds[it["item_id"]]) for it in gt_items
                   if it.get("platform") == plat and it["item_id"] in preds]
            if len(sub) < 5:
                continue
            yt = [m[0]["y"] for m in sub]
            ys = [m[1] for m in sub]
            m = compute_metrics(yt, ys)
            rows.append({"system": sname, "platform": plat, **m})

        print(f"  [done] {sname}", flush=True)

    # Write per-platform CSV
    fieldnames = ["system", "platform", "n", "roc_auc", "ci_lo", "ci_hi", "macro_f1_05"]
    PER_PLATFORM_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PER_PLATFORM_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {PER_PLATFORM_CSV} ({len(rows)} rows)", flush=True)

    # --- Code-switching features ---
    cs_rows = []
    for it in gt_items:
        cs = compute_code_switch_density(it["item_id"])
        cs["y"] = it["y"]
        cs["platform"] = it.get("platform", "unknown")
        cs_rows.append(cs)

    cs_fieldnames = ["item_id", "platform", "y", "has_transcript", "total_words",
                     "latin_words", "cs_density"]
    with CODE_SWITCH_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cs_fieldnames)
        writer.writeheader()
        writer.writerows(cs_rows)
    print(f"[saved] {CODE_SWITCH_CSV} ({len(cs_rows)} rows)", flush=True)

    # --- FINDINGS append ---
    _append_findings(rows, cs_rows, all_platforms, system_names)
    print("[done] FINDINGS Section 22 appended.", flush=True)


def _append_findings(rows: list, cs_rows: list, platforms: list, systems: list) -> None:
    # Build per-platform tables
    # Overall table (systems x platforms)
    section = "\n\n## 22. Per-Platform Benchmark Breakdown\n"
    section += f"_Generated 2026-06-29 by `scripts/26_per_platform_breakdown.py`._\n\n"

    # Per-platform summary
    section += "### 22.1 ROC-AUC by platform (systems with completed predictions)\n\n"

    # Build a pivot-like table
    plat_cols = ["ALL"] + [p for p in platforms if p != "unknown"]
    header = "| System | " + " | ".join(plat_cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(plat_cols)) + "|"
    section += header + "\n" + sep + "\n"

    # Index rows by (system, platform)
    idx: dict[tuple, dict] = {}
    for r in rows:
        idx[(r["system"], r["platform"])] = r

    for sname in systems:
        if (sname, "ALL") not in idx:
            continue
        cells = []
        for p in plat_cols:
            r = idx.get((sname, p))
            if r is None or r["n"] < 5:
                cells.append("—")
            else:
                auc = r["roc_auc"]
                n = r["n"]
                cells.append(f"{auc:.3f} (n={n})")
        section += f"| {sname} | " + " | ".join(cells) + " |\n"

    # Code-switching analysis
    cs_pos = [r for r in cs_rows if r["y"] == 1 and r["has_transcript"]]
    cs_neg = [r for r in cs_rows if r["y"] == 0 and r["has_transcript"]]
    avg_pos = np.mean([r["cs_density"] for r in cs_pos]) if cs_pos else 0.0
    avg_neg = np.mean([r["cs_density"] for r in cs_neg]) if cs_neg else 0.0

    section += f"""
### 22.2 Code-switching density

Code-switching density = fraction of transcript words in Latin script (French/English words).

- Lebanese (positive) items: avg density = {avg_pos:.4f} ({len(cs_pos)} items with transcripts)
- Non-Lebanese (negative) items: avg density = {avg_neg:.4f} ({len(cs_neg)} items with transcripts)

{"Lebanese items have higher code-switching density than non-Lebanese" if avg_pos > avg_neg else "Non-Lebanese items have similar or higher code-switching density"}. This reflects {"Lebanon's" if avg_pos > avg_neg else "the pan-Arab nature of"} French-loanword usage in spoken Lebanese Arabic (e.g., merci, bonjour, voiture) which appears as Latin-script transcriptions in Whisper outputs.

Platform breakdown of code-switching density:
"""
    plat_cs: dict[str, list] = {}
    for r in cs_rows:
        if r["has_transcript"]:
            plat_cs.setdefault(r["platform"], []).append(r["cs_density"])
    section += "| Platform | n | Mean CS density |\n|---|---:|---:|\n"
    for plat, densities in sorted(plat_cs.items()):
        section += f"| {plat} | {len(densities)} | {np.mean(densities):.4f} |\n"

    # Strip existing Section 22 if present
    text = FINDINGS_PATH.read_text(encoding="utf-8")
    marker = "\n\n## 22."
    if marker in text:
        text = text[:text.index(marker)]

    FINDINGS_PATH.write_text(text.rstrip() + section, encoding="utf-8")


if __name__ == "__main__":
    main()
