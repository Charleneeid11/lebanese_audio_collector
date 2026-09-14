#!/usr/bin/env python3
"""
ROADMAP v2 Day 6 — Same-source control experiment (P4).

Addresses the confound diagnosis from §12.6.3: V2 was trained on a mix of
platforms (ADI17, FLEURS, YouTube, podcast_rss) and the frozen XLS-R embeddings
encode platform-of-origin with 89% accuracy. The question: if we train V2 only on
podcast_rss data (same domain as the majority of GT), does performance recover?

Protocol:
  Train:  podcast_rss items with POTENTIAL_LB or WEAK_POSITIVE label = 1,
          podcast_rss items with WEAK_NEGATIVE or REJECTED label = 0
          (GT items excluded from training)
  Test:   podcast_rss subset of data/annotations.csv (n≈252)
  Model:  MLP on 1024-d XLS-R embeddings (same as V2)
  Output: one row appended to data/benchmark_results.csv as "v2_same_source"
          predictions saved to data/benchmark_predictions/v2_same_source.json

Appends FINDINGS Section 25 (idempotent).

Run:
  python scripts/29_same_source_control.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_harness_utils import (
    load_gt_items,
    compute_metrics,
    upsert_csv_row,
    save_predictions,
    PREDS_DIR,
    RESULTS_CSV,
)

EMB_PARQUET = Path("data/embeddings_with_labels.parquet")
QUEUE_DB = Path("data/queue.db")
MODEL_OUT = Path("models/dialect_classifier_v2_same_source.joblib")
FINDINGS_PATH = Path("FINDINGS.md")

SYSTEM_NAME = "v2_same_source"
RANDOM_STATE = 42

# DB statuses → binary label
POSITIVE_STATUSES = {"POTENTIAL_LB", "WEAK_POSITIVE", "BORDERLINE_LB"}
NEGATIVE_STATUSES = {"WEAK_NEGATIVE", "REJECTED"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_platform_map(item_ids: list[int]) -> dict[int, str]:
    """Query queue.db for platform of each item_id."""
    conn = sqlite3.connect(str(QUEUE_DB))
    cur = conn.cursor()
    platform_map: dict[int, str] = {}
    batch_size = 500
    for i in range(0, len(item_ids), batch_size):
        batch = item_ids[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = cur.execute(
            f"SELECT id, platform FROM queue WHERE id IN ({placeholders})", batch
        ).fetchall()
        for iid, plat in rows:
            platform_map[iid] = plat or "unknown"
    conn.close()
    return platform_map


def get_status_map(item_ids: list[int]) -> dict[int, str]:
    """Query queue.db for pipeline status of each item_id."""
    conn = sqlite3.connect(str(QUEUE_DB))
    cur = conn.cursor()
    status_map: dict[int, str] = {}
    batch_size = 500
    for i in range(0, len(item_ids), batch_size):
        batch = item_ids[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = cur.execute(
            f"SELECT id, status FROM queue WHERE id IN ({placeholders})", batch
        ).fetchall()
        for iid, status in rows:
            status_map[iid] = status or "unknown"
    conn.close()
    return status_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Day 6: Same-source control (V2 podcast_rss only) ===\n")

    # 1. Load embeddings
    print(f"Loading {EMB_PARQUET} ...")
    df = pd.read_parquet(EMB_PARQUET)
    print(f"  {len(df)} rows, columns: {list(df.columns[:6])} ...")

    # Embeddings are stored as a single 'embedding' column (1024-d numpy array per row)
    if "embedding" not in df.columns:
        raise ValueError(f"Expected 'embedding' column. Got: {list(df.columns)}")
    emb_dim = len(df["embedding"].iloc[0])
    print(f"  Embedding dims: {emb_dim}")

    # Normalise item_id column name
    if "item_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "item_id"})

    item_ids = df["item_id"].tolist()

    # 2. Platform and status are already in the parquet (from script 13_load_embeddings)
    # Fall back to queue.db if columns are missing
    if "platform" not in df.columns or "status" not in df.columns:
        print("Fetching platform + status from queue.db ...")
        platform_map = get_platform_map(item_ids)
        status_map = get_status_map(item_ids)
        df["platform"] = df["item_id"].map(platform_map).fillna("unknown")
        df["status"] = df["item_id"].map(status_map).fillna("unknown")
    else:
        print("  Platform and status loaded from parquet.")

    # 3. Load GT to exclude GT items from training
    gt_items = load_gt_items()
    gt_ids = {it["item_id"] for it in gt_items}
    print(f"GT items (excluded from training): {len(gt_ids)}")

    # 4. Build training set: podcast_rss only, not in GT
    # Use the parquet's in_ground_truth flag when available
    if "in_ground_truth" in df.columns:
        podcast_mask = (df["platform"] == "podcast_rss") & (~df["in_ground_truth"])
    else:
        podcast_mask = (df["platform"] == "podcast_rss") & (~df["item_id"].isin(gt_ids))
    df_train_pool = df[podcast_mask].copy()
    print(f"Podcast_rss pool (non-GT): {len(df_train_pool)} rows")

    # Assign binary labels from status
    def status_to_label(s: str) -> int | None:
        if s in POSITIVE_STATUSES:
            return 1
        if s in NEGATIVE_STATUSES:
            return 0
        return None

    df_train_pool["y_train"] = df_train_pool["status"].map(status_to_label)
    df_train_labeled = df_train_pool[df_train_pool["y_train"].notna()].copy()
    df_train_labeled["y_train"] = df_train_labeled["y_train"].astype(int)

    n_pos = int((df_train_labeled["y_train"] == 1).sum())
    n_neg = int((df_train_labeled["y_train"] == 0).sum())
    print(f"Training set: {len(df_train_labeled)} labeled ({n_pos} pos, {n_neg} neg)")

    if len(df_train_labeled) < 10:
        raise RuntimeError("Not enough training data in podcast_rss. Check embeddings and status.")

    X_train = np.stack(df_train_labeled["embedding"].values).astype(np.float32)
    y_train = df_train_labeled["y_train"].values

    # 5. Train MLP (same architecture as V2)
    print("Training MLP ...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256,),
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=200,
            random_state=RANDOM_STATE,
        )),
    ])
    clf.fit(X_train, y_train)
    print(f"  MLP stopped at iteration {clf['mlp'].n_iter_}")

    # 6. Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"  Model saved to {MODEL_OUT}")

    # 7. Evaluate on podcast_rss subset of GT
    gt_podcast = [it for it in gt_items if it["platform"] == "podcast_rss"]
    gt_podcast_ids = [it["item_id"] for it in gt_podcast]
    print(f"\nEvaluating on podcast_rss GT subset: {len(gt_podcast)} items")

    # Get embeddings for GT podcast items
    gt_emb_df = df[df["item_id"].isin(gt_podcast_ids)].copy()
    matched_ids = set(gt_emb_df["item_id"].tolist())
    n_missing = len(gt_podcast_ids) - len(matched_ids)
    if n_missing > 0:
        print(f"  Warning: {n_missing} GT items have no embedding (skipped)")

    # Build ordered arrays
    probs_by_iid: dict[int, float] = {}
    if len(gt_emb_df) > 0:
        X_eval = np.stack(gt_emb_df["embedding"].values).astype(np.float32)
        proba = clf.predict_proba(X_eval)[:, 1]
        for iid, prob in zip(gt_emb_df["item_id"].tolist(), proba.tolist()):
            probs_by_iid[iid] = float(prob)

    # Quick metrics on podcast subset
    aligned = [(it["y"], probs_by_iid[it["item_id"]])
               for it in gt_podcast if it["item_id"] in probs_by_iid]
    y_eval = np.array([a[0] for a in aligned])
    p_eval = np.array([a[1] for a in aligned])

    if len(set(y_eval)) > 1:
        roc = float(roc_auc_score(y_eval, p_eval))
        mf1 = float(f1_score(y_eval, (p_eval >= 0.5).astype(int),
                              labels=[0, 1], average="macro", zero_division=0))
    else:
        roc, mf1 = float("nan"), float("nan")

    print(f"  podcast_rss GT — ROC-AUC: {roc:.4f}  macro F1: {mf1:.4f}  n={len(aligned)}")

    # 8. Evaluate on ALL GT items (to add a full benchmark row)
    print("\nEvaluating on full GT ...")
    gt_emb_all = df[df["item_id"].isin(gt_ids)].copy()
    probs_all: dict[int, float] = {}
    if len(gt_emb_all) > 0:
        X_all = np.stack(gt_emb_all["embedding"].values).astype(np.float32)
        proba_all = clf.predict_proba(X_all)[:, 1]
        for iid, prob in zip(gt_emb_all["item_id"].tolist(), proba_all.tolist()):
            probs_all[iid] = float(prob)

    row = compute_metrics(SYSTEM_NAME, "acoustic", gt_items, probs_all)
    print(f"  Full GT — ROC-AUC: {row.get('roc_auc', 'N/A'):.4f}  "
          f"macro F1: {row.get('macro_f1_at_0.5', 'N/A'):.4f}  "
          f"n={row.get('n_evaluated', 0)}")

    # 9. Save predictions + update benchmark CSV
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(PREDS_DIR / f"{SYSTEM_NAME}.json", SYSTEM_NAME, gt_items, probs_all)
    upsert_csv_row(RESULTS_CSV, row)
    print(f"\nSaved predictions to {PREDS_DIR / f'{SYSTEM_NAME}.json'}")
    print(f"Updated {RESULTS_CSV}")

    # 10. Append FINDINGS Section 25
    _append_findings(
        n_train=len(df_train_labeled),
        n_pos=n_pos, n_neg=n_neg,
        podcast_roc=roc, podcast_mf1=mf1, podcast_n=len(aligned),
        full_roc=row.get("roc_auc", float("nan")),
        full_mf1=row.get("macro_f1_at_0.5", float("nan")),
        full_n=row.get("n_evaluated", 0),
        mlp_iter=clf["mlp"].n_iter_,
    )
    print("FINDINGS Section 25 appended.")


def _append_findings(
    n_train: int, n_pos: int, n_neg: int,
    podcast_roc: float, podcast_mf1: float, podcast_n: int,
    full_roc: float, full_mf1: float, full_n: int,
    mlp_iter: int,
) -> None:
    text = FINDINGS_PATH.read_text(encoding="utf-8")

    text = re.sub(
        r"\n## Section 25[^\n]*\n.*?(?=\n## Section |\Z)",
        "",
        text,
        flags=re.DOTALL,
    )

    # Compare to V2 frozen (from benchmark_results.csv if available)
    v2_podcast_roc = "0.7906"  # from benchmark_per_platform.csv podcast_rss row

    section = f"""

## Section 25 — Same-source Control: V2 Trained on Podcast-only Data (Day 6)

### 25.1 Motivation

§12.6.3 diagnosed a recording-domain confound: frozen XLS-R embeddings encode
platform-of-origin (podcast_rss, YouTube, ADI17, FLEURS) with 89.12% accuracy
(Section 16). V2's training pool mixes all platforms. The question: if we train V2
*only* on podcast_rss items — the same domain as 85% of the GT — does performance
recover?

### 25.2 Protocol

- **Model:** MLP(256) + StandardScaler on 1024-d XLS-R embeddings (identical to V2)
- **Training pool:** podcast_rss items, not in GT annotations, with labeled status:
  - Positive: POTENTIAL_LB, WEAK_POSITIVE, BORDERLINE_LB
  - Negative: WEAK_NEGATIVE, REJECTED
  - Training set: {n_train} items ({n_pos} pos, {n_neg} neg)
- **Test (primary):** podcast_rss subset of GT ({podcast_n} items)
- **Test (full GT):** all {full_n} GT items (for benchmark row comparison)
- MLP stopped at iteration {mlp_iter}

### 25.3 Results

| Scope | ROC-AUC | Macro F1 (0.5) |
|---|---|---|
| podcast_rss GT subset | {podcast_roc:.4f} | {podcast_mf1:.4f} |
| Full GT (V2 same-source) | {full_roc:.4f} | {full_mf1:.4f} |
| V2 frozen (cross-domain, full GT) | 0.7865 | 0.3794 |
| V2 frozen (podcast_rss subset) | {v2_podcast_roc} | 0.3318 |

### 25.4 Interpretation

If same-source training substantially recovers performance on the podcast_rss subset,
it confirms the confound is the dominant cause of V2's underperformance: the model
was not learning dialect signal, it was learning domain signal. If performance does
*not* recover, it suggests the 1024-d XLS-R representation genuinely lacks Lebanese
dialect information regardless of training domain — a stronger null result for acoustic
features.

This directly disambiguates §12.6.3's "domain shortcut" explanation from the
alternative "weak intrinsic acoustic signal" explanation.
"""

    FINDINGS_PATH.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


if __name__ == "__main__":
    main()
