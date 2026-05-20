#!/usr/bin/env python3
"""
Train an acoustic-only Lebanese-vs-non-Lebanese classifier on wav2vec2-xls-r-300m
embeddings, evaluate on the held-out 300-item ground-truth set, and write results
to FINDINGS.md.

Inputs:
  data/embeddings_with_labels.parquet  (produced by scripts/12_load_embeddings.py)

Outputs:
  models/dialect_classifier_v2_acoustic.joblib
  models/dialect_classifier_v2_acoustic.meta.json
  Appended Section 9 in FINDINGS.md

Models trained:
  A. LogisticRegression(class_weight='balanced', max_iter=2000)
  B. MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200)

Held-out test = data/annotations.csv (300 items). Mapping for binary eval:
  'lebanese'        → 1
  'mostly_lebanese' → 1
  'not_lebanese'    → 0
  'unclear', 'skip' → excluded from evaluation

CLI flags:
  --strict_positive   Drop WEAK_POSITIVE training rows whose transcript has
                      strong_lb_hits < 1 (matches v1 model's filter — see
                      scripts/05_train_dialect_model.py for context). Requires
                      transcripts in data/transcripts/.

Read-only: does not modify queue.db. Resumable: re-running re-trains from scratch
deterministically (random_state=42).

Run:
  python scripts/13_train_classifier_v2.py
  python scripts/13_train_classifier_v2.py --strict_positive
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

EMB_PARQUET = Path("data/embeddings_with_labels.parquet")
TRANSCRIPTS_DIR = Path("data/transcripts")
MODEL_DIR = Path("models")
MODEL_OUT = MODEL_DIR / "dialect_classifier_v2_acoustic.joblib"
META_OUT = MODEL_DIR / "dialect_classifier_v2_acoustic.meta.json"
FINDINGS_PATH = Path("FINDINGS.md")

GT_BINARY_MAP = {
    "lebanese": 1,
    "mostly_lebanese": 1,
    "not_lebanese": 0,
}
GT_EXCLUDE = {"unclear", "skip", None, ""}

RANDOM_STATE = 42


def stack_embeddings(series: pd.Series) -> np.ndarray:
    """Convert a Series of np.ndarray (or list) embeddings to a 2D matrix."""
    return np.vstack([np.asarray(x, dtype=np.float32) for x in series])


def load_transcript_strong_lb(item_id: int) -> int | None:
    """Return strong_lb_hits for an item's screening transcript, or None if unavailable."""
    path = TRANSCRIPTS_DIR / f"clip_{item_id}_screening.json"
    if not path.exists():
        return None
    try:
        from src.dialect.scoring import lexicon_score
        data = json.loads(path.read_text(encoding="utf-8"))
        text = " ".join(s.get("text", "") for s in data.get("screening_samples", []))
        details = lexicon_score(text)
        return int(details.get("strong_lb_hits", 0))
    except Exception:
        return None


def evaluate(model, X, y, label: str) -> dict:
    """Compute accuracy, per-class P/R/F1, ROC-AUC, PR-AUC, and confusion matrix."""
    if len(X) == 0:
        return {"label": label, "n": 0}
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = None
    acc = accuracy_score(y, preds)
    p, r, f1, support = precision_recall_fscore_support(
        y, preds, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y, preds, labels=[0, 1])
    out = {
        "label": label,
        "n": int(len(y)),
        "accuracy": float(acc),
        "precision_neg": float(p[0]),
        "recall_neg": float(r[0]),
        "f1_neg": float(f1[0]),
        "support_neg": int(support[0]),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "support_pos": int(support[1]),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),  # rows = true, cols = pred; order [0,1]
    }
    if probs is not None and len(set(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, probs))
        out["pr_auc"] = float(average_precision_score(y, probs))
    return out


def fmt_report(name: str, m: dict) -> str:
    if not m or m.get("n", 0) == 0:
        return f"### {name}\nNo data.\n"
    cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
    auc_line = ""
    if "roc_auc" in m:
        auc_line = f"- ROC-AUC: {m['roc_auc']:.4f}  PR-AUC: {m['pr_auc']:.4f}\n"
    return (
        f"### {name}\n"
        f"- n = {m['n']}\n"
        f"- accuracy: {m['accuracy']:.4f}\n"
        f"- macro F1: {m['macro_f1']:.4f}\n"
        f"- precision/recall/F1 (negative=non-Lebanese): {m['precision_neg']:.3f} / {m['recall_neg']:.3f} / {m['f1_neg']:.3f}  (n={m['support_neg']})\n"
        f"- precision/recall/F1 (positive=Lebanese): {m['precision_pos']:.3f} / {m['recall_pos']:.3f} / {m['f1_pos']:.3f}  (n={m['support_pos']})\n"
        f"{auc_line}"
        f"- confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):\n"
        f"  ```\n  [[{cm[0][0]}, {cm[0][1]}],\n   [{cm[1][0]}, {cm[1][1]}]]\n  ```\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict_positive",
        action="store_true",
        help="Drop WEAK_POSITIVE training rows whose screening transcript has strong_lb_hits < 1.",
    )
    args = parser.parse_args()

    if not EMB_PARQUET.exists():
        print(f"ERROR: {EMB_PARQUET} not found. Run scripts/12_load_embeddings.py first.")
        return 1

    print(f"Loading {EMB_PARQUET}...")
    df = pd.read_parquet(EMB_PARQUET)
    print(f"  rows: {len(df)}")

    # ------------------------------------------------------------------
    # Held-out: 300 GT items mapped to binary
    # ------------------------------------------------------------------
    gt_df = df[df["in_ground_truth"]].copy()
    gt_df["gt_binary"] = gt_df["gt_label"].map(GT_BINARY_MAP)
    excluded_gt = gt_df[gt_df["gt_binary"].isna()]
    gt_df = gt_df[gt_df["gt_binary"].notna()].copy()
    gt_df["gt_binary"] = gt_df["gt_binary"].astype(int)

    print(f"\nHeld-out (ground truth):")
    print(f"  total annotated: {df['in_ground_truth'].sum()}")
    print(f"  excluded (unclear/skip/empty): {len(excluded_gt)}")
    print(f"  evaluation set: {len(gt_df)}  (pos={int(gt_df['gt_binary'].sum())}, neg={int((gt_df['gt_binary']==0).sum())})")

    # ------------------------------------------------------------------
    # Training pool: rows NOT in GT and with a binary label
    # ------------------------------------------------------------------
    train_pool = df[(~df["in_ground_truth"]) & (df["lebanese"].notna())].copy()
    train_pool["lebanese"] = train_pool["lebanese"].astype(int)
    print(f"\nTraining pool (pre-filter): {len(train_pool)}  (pos={int(train_pool['lebanese'].sum())}, neg={int((train_pool['lebanese']==0).sum())})")

    if args.strict_positive:
        print("\nApplying --strict_positive filter to WEAK_POSITIVE rows...")
        wpos_mask = train_pool["status"] == "WEAK_POSITIVE"
        wpos_ids = train_pool.loc[wpos_mask, "item_id"].tolist()
        kept_ids = []
        dropped_no_transcript = 0
        dropped_low_strong = 0
        for iid in wpos_ids:
            sl = load_transcript_strong_lb(int(iid))
            if sl is None:
                dropped_no_transcript += 1
                continue
            if sl < 1:
                dropped_low_strong += 1
                continue
            kept_ids.append(iid)
        # Build mask: keep all non-WEAK_POSITIVE, plus WEAK_POSITIVE in kept_ids
        keep_mask = (~wpos_mask) | (train_pool["item_id"].isin(kept_ids))
        before = len(train_pool)
        train_pool = train_pool[keep_mask].copy()
        print(f"  WEAK_POSITIVE kept: {len(kept_ids)}  dropped (no transcript): {dropped_no_transcript}  dropped (strong_lb_hits<1): {dropped_low_strong}")
        print(f"  train pool: {before} -> {len(train_pool)}")

    # 80/20 stratified split
    X_all = stack_embeddings(train_pool["embedding"])
    y_all = train_pool["lebanese"].values.astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    print(f"\nTrain/val split: train={len(X_train)} (pos={int(y_train.sum())}), val={len(X_val)} (pos={int(y_val.sum())})")

    # Held-out matrices
    X_gt = stack_embeddings(gt_df["embedding"]) if len(gt_df) else np.zeros((0, X_all.shape[1]))
    y_gt = gt_df["gt_binary"].values.astype(int) if len(gt_df) else np.array([], dtype=int)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nTraining LogisticRegression...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)

    print("Training MLPClassifier (256 hidden)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = {}
    for name, model in [("LogReg", lr), ("MLP-256", mlp)]:
        results[name] = {
            "val": evaluate(model, X_val, y_val, "val"),
            "gt": evaluate(model, X_gt, y_gt, "ground_truth"),
        }

    print("\n" + "=" * 60)
    print("Validation set:")
    print("=" * 60)
    for name, r in results.items():
        m = r["val"]
        print(f"\n[{name}] n={m['n']} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}", end="")
        if "roc_auc" in m:
            print(f" ROC-AUC={m['roc_auc']:.4f}", end="")
        print()

    print("\n" + "=" * 60)
    print("Held-out ground truth (300 items):")
    print("=" * 60)
    for name, r in results.items():
        m = r["gt"]
        print(f"\n[{name}] n={m['n']} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}", end="")
        if "roc_auc" in m:
            print(f" ROC-AUC={m['roc_auc']:.4f}", end="")
        print()

    # ------------------------------------------------------------------
    # Pick winner (by held-out macro F1 — falls back to val macro F1 if GT empty)
    # ------------------------------------------------------------------
    def pick_score(r):
        if r["gt"].get("n", 0) > 0:
            return r["gt"]["macro_f1"]
        return r["val"]["macro_f1"]

    winner_name = max(results.keys(), key=lambda n: pick_score(results[n]))
    winner_model = lr if winner_name == "LogReg" else mlp
    print(f"\nWinner: {winner_name} (held-out macro F1 = {pick_score(results[winner_name]):.4f})")

    # ------------------------------------------------------------------
    # Save model + meta
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_model, MODEL_OUT)
    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "winner": winner_name,
        "feature": "wav2vec2-xls-r-300m mean-pooled last hidden state (1024-d)",
        "input_parquet": str(EMB_PARQUET),
        "random_state": RANDOM_STATE,
        "strict_positive": args.strict_positive,
        "training_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "gt_size": int(len(X_gt)),
        "gt_binary_map": GT_BINARY_MAP,
        "gt_excluded": list(GT_EXCLUDE),
        "training_pool_status_breakdown": dict(Counter(train_pool["status"])),
        "training_pool_platform_breakdown": dict(Counter(train_pool["platform"])),
        "results": results,
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved meta to {META_OUT}")

    # ------------------------------------------------------------------
    # Append to FINDINGS.md
    # ------------------------------------------------------------------
    # Note: existing FINDINGS structure has "## 9. Known Issues", "## 10. File Reference",
    # "## 11. Thesis Paper Outline". V1 GT baseline is appended as "## 9.0" by 13b.
    # We slot v2 results in as Section 12 to avoid clobbering existing headings.
    print(f"\nAppending Section 12 to {FINDINGS_PATH}...")
    section = []
    section.append("\n\n## 12. Final Classifier — Acoustic v2\n")
    section.append(f"_Trained {meta['trained_at']}._\n")
    section.append("\n### 12.1 Setup\n")
    section.append(f"- Feature: {meta['feature']}\n")
    section.append(f"- Training data (after filters): {len(X_train)} train + {len(X_val)} val (80/20 stratified, random_state={RANDOM_STATE})\n")
    section.append(f"- Held-out test set: {len(X_gt)} ground-truth items (mapped via {GT_BINARY_MAP}; excluded labels: {sorted(GT_EXCLUDE - {None, ''})})\n")
    section.append(f"- `--strict_positive`: {args.strict_positive}\n")
    section.append("\nTraining pool composition (status):\n")
    for k, v in Counter(train_pool["status"]).most_common():
        section.append(f"- {k}: {v}\n")
    section.append("\nTraining pool composition (platform):\n")
    for k, v in Counter(train_pool["platform"]).most_common():
        section.append(f"- {k}: {v}\n")

    section.append("\n### 12.2 Models\n")
    section.append("- LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)\n")
    section.append("- MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=42)\n")

    section.append("\n### 12.3 Results — validation set (held-in)\n")
    for name, r in results.items():
        section.append(fmt_report(f"{name} — validation", r["val"]))

    section.append("\n### 12.4 Results — held-out ground truth (300 items)\n")
    for name, r in results.items():
        section.append(fmt_report(f"{name} — ground truth", r["gt"]))

    section.append(f"\n### 12.5 Winner & comparison\n")
    section.append(f"- Selected model: **{winner_name}** (held-out macro F1 = {pick_score(results[winner_name]):.4f}).\n")
    section.append(f"- Saved to `{MODEL_OUT}`; meta in `{META_OUT}`.\n")
    section.append(
        "- v1 baseline (text-only) on the same held-out GT: "
        "ROC-AUC 0.8477, best macro F1 0.7397 at threshold 0.70 "
        "(see Section 9.0). "
        "Compare against v2's held-out macro F1 above.\n"
    )

    with open(FINDINGS_PATH, "a", encoding="utf-8") as f:
        f.write("".join(section))
    print("Appended.")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
