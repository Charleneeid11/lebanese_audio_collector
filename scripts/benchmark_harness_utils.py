"""
Shared utilities for the Lebanese ADI benchmark (Day 1+).

Imported by:
  scripts/20_benchmark_harness.py
  scripts/22_v1_ablation.py
  scripts/23_whisper_lid_baseline.py (planned)
  scripts/24_eval_public_systems.py (planned)
  ...

Contract: any script that adds a "benchmark row" should call:
  items = load_gt_items()
  probs = your_predict(items)  # dict[item_id -> prob_lebanese]
  save_predictions(PREDS_DIR / f"{system_name}.json", system_name, items, probs)
  row = compute_metrics(system_name, family, items, probs)
  upsert_csv_row(RESULTS_CSV, row)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_auc_score,
)

ANNOTATIONS_CSV = Path("data/annotations.csv")
RESULTS_CSV = Path("data/benchmark_results.csv")
PREDS_DIR = Path("data/benchmark_predictions")

GT_BINARY_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}
GT_EXCLUDE = {"unclear", "skip", "", None}

THRESHOLD_SWEEP = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DEFAULT_THR = 0.50
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def load_gt_items() -> list[dict]:
    items = []
    with open(ANNOTATIONS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("ground_truth") or "").strip().lower()
            if raw in GT_EXCLUDE:
                continue
            y = GT_BINARY_MAP.get(raw)
            if y is None:
                continue
            try:
                iid = int(row["item_id"])
            except (KeyError, ValueError):
                continue
            items.append({
                "item_id": iid,
                "y": y,
                "raw_label": raw,
                "platform": (row.get("platform") or "").strip().lower() or "unknown",
            })
    return items


def _macro_f1_at_thr(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    preds = (p >= thr).astype(int)
    return float(f1_score(y, preds, labels=[0, 1], average="macro", zero_division=0))


def bootstrap_ci_macro_f1(y: np.ndarray, p: np.ndarray, thr: float,
                          n_resamples: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        try:
            stats.append(_macro_f1_at_thr(y[idx], p[idx], thr))
        except Exception:
            continue
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_roc_auc(y: np.ndarray, p: np.ndarray,
                          n_resamples: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        try:
            stats.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            continue
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def compute_metrics(system_name: str, family: str,
                    items: list[dict], probs_by_iid: dict[int, float]) -> dict:
    aligned = [(it["y"], it["item_id"], it["platform"], probs_by_iid[it["item_id"]])
               for it in items if it["item_id"] in probs_by_iid]
    if not aligned:
        return {"system": system_name, "family": family, "n_evaluated": 0, "error": "no predictions"}
    y = np.array([a[0] for a in aligned], dtype=int)
    p = np.array([a[3] for a in aligned], dtype=float)

    preds_default = (p >= DEFAULT_THR).astype(int)
    pr, rc, f1, support = precision_recall_fscore_support(y, preds_default, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, preds_default, labels=[0, 1])

    sweep = [(thr, _macro_f1_at_thr(y, p, thr)) for thr in THRESHOLD_SWEEP]
    best_thr, best_mf1 = max(sweep, key=lambda kv: kv[1])

    roc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")

    mf1_lo, mf1_hi = bootstrap_ci_macro_f1(y, p, DEFAULT_THR)
    auc_lo, auc_hi = bootstrap_ci_roc_auc(y, p)

    return {
        "system": system_name,
        "family": family,
        "n_evaluated": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "accuracy_at_0.5": float(accuracy_score(y, preds_default)),
        "macro_f1_at_0.5": float(f1.mean()),
        "macro_f1_ci95_lo": mf1_lo,
        "macro_f1_ci95_hi": mf1_hi,
        "best_threshold": float(best_thr),
        "macro_f1_at_best": float(best_mf1),
        "f1_neg_at_0.5": float(f1[0]),
        "f1_pos_at_0.5": float(f1[1]),
        "precision_neg_at_0.5": float(pr[0]),
        "recall_neg_at_0.5": float(rc[0]),
        "precision_pos_at_0.5": float(pr[1]),
        "recall_pos_at_0.5": float(rc[1]),
        "support_neg": int(support[0]),
        "support_pos": int(support[1]),
        "roc_auc": roc,
        "roc_auc_ci95_lo": auc_lo,
        "roc_auc_ci95_hi": auc_hi,
        "pr_auc": pr_auc,
        "confusion_matrix_at_0.5": cm.tolist(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


CSV_FIELDS = [
    "system", "family", "n_evaluated", "n_positive", "n_negative",
    "accuracy_at_0.5", "macro_f1_at_0.5", "macro_f1_ci95_lo", "macro_f1_ci95_hi",
    "best_threshold", "macro_f1_at_best",
    "f1_neg_at_0.5", "f1_pos_at_0.5",
    "precision_neg_at_0.5", "recall_neg_at_0.5",
    "precision_pos_at_0.5", "recall_pos_at_0.5",
    "support_neg", "support_pos",
    "roc_auc", "roc_auc_ci95_lo", "roc_auc_ci95_hi", "pr_auc",
    "confusion_matrix_at_0.5", "generated_at",
]


def upsert_csv_row(path: Path, row: dict) -> None:
    rows = []
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("system") != row["system"]:
                    rows.append(r)
    rows.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items() if k in CSV_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_predictions(path: Path, system_name: str, items: list[dict],
                     probs_by_iid: dict[int, float]) -> None:
    payload = {
        "system": system_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_items_requested": len(items),
        "n_items_predicted": len(probs_by_iid),
        "predictions": [
            {"item_id": it["item_id"], "y": it["y"],
             "platform": it["platform"], "prob": probs_by_iid[it["item_id"]]}
            for it in items if it["item_id"] in probs_by_iid
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_predictions(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["item_id"]): float(r["prob"]) for r in data["predictions"]}
