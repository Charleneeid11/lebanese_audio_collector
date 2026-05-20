#!/usr/bin/env python3
"""
Evaluate v1 (text-only) dialect classifier on the held-out 300-item ground-truth set.

Provides the baseline number that v2 (acoustic, scripts/14_train_classifier_v2.py)
must beat for the thesis comparison. v1's previously reported 92% / ROC-AUC 0.9774
is from the train/val split, NOT the held-out GT — this script produces the GT
number directly.

Inputs (read-only):
  models/dialect_classifier.joblib                   — v1 LogisticRegression
  data/annotations.csv                                — 300 GT items
  data/transcripts/clip_<item_id>_screening.json     — per-item screening transcripts

Output:
  Appends Section 9.0 "V1 Text-Only Baseline on Held-Out Ground Truth" to FINDINGS.md.
  Prints full metrics to stdout.

Feature pipeline (mirrors scripts/05_train_dialect_model.py exactly):
  text = " ".join(s["text"] for s in screening_samples)
  diagnostics = final_dialect_score(text)
  lex_features = [lb, msa, strong_lb_hits, msa_ratio_core, final_score]   # 5 dims
  embedding = embed_text(text)                                             # 384 dims
  feature_vector = concat(lex_features, embedding)                         # 389 dims

GT label mapping (must match scripts/14_train_classifier_v2.py for fair v1↔v2 comparison):
  lebanese, mostly_lebanese → 1
  not_lebanese              → 0
  unclear, skip, empty      → excluded

Reported metrics:
  - Accuracy, precision/recall/F1 per class, ROC-AUC, PR-AUC
  - Confusion matrices at thresholds 0.5, 0.7, 0.75
  - Per-threshold summary table

Run: python scripts/13b_eval_v1_on_gt.py
Read-only: does not modify queue.db, the v1 model, or any data files
           (only appends to FINDINGS.md).
"""

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.cfg import Settings
from src.dialect.scoring import final_dialect_score
from src.embeddings.engine import embed_text


MODEL_PATH = Path("models/dialect_classifier.joblib")
ANNOTATIONS_CSV = Path("data/annotations.csv")
FINDINGS_PATH = Path("FINDINGS.md")

GT_BINARY_MAP = {
    "lebanese": 1,
    "mostly_lebanese": 1,
    "not_lebanese": 0,
}
GT_EXCLUDE = {"unclear", "skip", "", None}

THRESHOLDS = [0.5, 0.7, 0.75]


def load_transcript_text(item_id: int, transcripts_dir: Path) -> str | None:
    """Replica of scripts/05_train_dialect_model.py:load_transcript_text."""
    path = transcripts_dir / f"clip_{item_id}_screening.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = data.get("screening_samples") or []
        return " ".join(s.get("text", "") for s in samples)
    except Exception:
        return None


def build_feature_vector(text: str, diagnostics: dict) -> np.ndarray:
    """Replica of scripts/05_train_dialect_model.py:build_feature_vector."""
    lex = diagnostics["lexicon_details"]
    lex_features = np.array([
        lex["lb"],
        lex["msa"],
        lex["strong_lb_hits"],
        lex["msa_ratio_core"],
        diagnostics["final_score"],
    ], dtype=float)
    embedding = embed_text(text).astype(float)
    return np.concatenate([lex_features, embedding])


def load_annotations() -> list[dict]:
    rows = []
    with open(ANNOTATIONS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def map_label(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in GT_EXCLUDE:
        return None
    return GT_BINARY_MAP.get(s)


def metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    preds = (probs >= thr).astype(int)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    return {
        "threshold": thr,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_neg": float(p[0]),
        "recall_neg": float(r[0]),
        "f1_neg": float(f1[0]),
        "support_neg": int(support[0]),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "support_pos": int(support[1]),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm,  # rows=true (0,1), cols=pred (0,1)
    }


def fmt_threshold_block(m: dict) -> str:
    cm = m["confusion_matrix"]
    return (
        f"#### Threshold = {m['threshold']:.2f}\n"
        f"- Accuracy: {m['accuracy']:.4f}\n"
        f"- Macro F1: {m['macro_f1']:.4f}\n"
        f"- Precision/Recall/F1 (negative=non-Lebanese): "
        f"{m['precision_neg']:.3f} / {m['recall_neg']:.3f} / {m['f1_neg']:.3f}  "
        f"(n={m['support_neg']})\n"
        f"- Precision/Recall/F1 (positive=Lebanese):     "
        f"{m['precision_pos']:.3f} / {m['recall_pos']:.3f} / {m['f1_pos']:.3f}  "
        f"(n={m['support_pos']})\n"
        f"- Confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):\n"
        f"  ```\n  [[{cm[0][0]}, {cm[0][1]}],\n   [{cm[1][0]}, {cm[1][1]}]]\n  ```\n"
    )


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        return 1
    if not ANNOTATIONS_CSV.exists():
        print(f"ERROR: {ANNOTATIONS_CSV} not found.")
        return 1

    settings = Settings.load()
    transcripts_dir = Path(settings.transcription.transcripts_dir)
    print(f"Loading v1 model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"  classes: {model.classes_.tolist()}")
    n_features_expected = model.coef_.shape[1] if hasattr(model, "coef_") else None
    print(f"  expected feature dim: {n_features_expected}")

    print(f"\nLoading {ANNOTATIONS_CSV}...")
    rows = load_annotations()
    print(f"  rows: {len(rows)}")

    raw_label_counts = Counter((r.get("ground_truth") or "").strip().lower() for r in rows)
    print(f"  raw GT label distribution: {dict(raw_label_counts)}")

    # Build feature matrix and binary labels for items we can score
    X = []
    y = []
    excluded_labels = []         # gt_label outside binary map
    missing_transcripts = []     # no screening transcript on disk
    feature_errors = []          # build_feature_vector failed
    used_item_ids = []

    for r in rows:
        try:
            iid = int(r["item_id"])
        except (KeyError, ValueError):
            continue
        bin_label = map_label(r.get("ground_truth"))
        if bin_label is None:
            excluded_labels.append((iid, r.get("ground_truth")))
            continue
        text = load_transcript_text(iid, transcripts_dir)
        if not text or not text.strip():
            missing_transcripts.append(iid)
            continue
        try:
            diagnostics = final_dialect_score(text)
            fv = build_feature_vector(text, diagnostics)
        except Exception as e:
            feature_errors.append((iid, f"{type(e).__name__}: {e}"))
            continue
        X.append(fv)
        y.append(bin_label)
        used_item_ids.append(iid)

    print(
        f"\n  evaluable: {len(X)} | excluded by label: {len(excluded_labels)} "
        f"| missing transcripts: {len(missing_transcripts)} "
        f"| feature errors: {len(feature_errors)}"
    )
    if not X:
        print("ERROR: no items could be scored. Aborting.")
        return 1

    X = np.vstack(X)
    y = np.array(y, dtype=int)
    print(f"  X.shape = {X.shape}, y.shape = {y.shape}, pos = {int(y.sum())}, neg = {int((y == 0).sum())}")

    if n_features_expected and X.shape[1] != n_features_expected:
        print(
            f"WARNING: feature dim {X.shape[1]} != model expected {n_features_expected}. "
            "Predictions may be invalid."
        )

    probs = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, probs)) if len(set(y)) > 1 else None
    pr_auc = float(average_precision_score(y, probs)) if len(set(y)) > 1 else None

    threshold_results = [metrics_at_threshold(y, probs, thr) for thr in THRESHOLDS]
    best = max(threshold_results, key=lambda m: m["macro_f1"])

    # ------------------------------------------------------------------
    # Print to stdout
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"v1 baseline on held-out ground truth (n={len(y)})")
    print(f"{'=' * 60}")
    print(f"ROC-AUC: {auc:.4f}" if auc is not None else "ROC-AUC: undefined (single-class)")
    print(f"PR-AUC:  {pr_auc:.4f}" if pr_auc is not None else "PR-AUC: undefined")
    for m in threshold_results:
        print()
        print(fmt_threshold_block(m))
    print(f"\nBest threshold by macro F1: {best['threshold']:.2f} (macro F1 = {best['macro_f1']:.4f})")

    # ------------------------------------------------------------------
    # Append Section 9.0 to FINDINGS.md
    # ------------------------------------------------------------------
    print(f"\nAppending Section 9.0 to {FINDINGS_PATH}...")
    section = []
    section.append(f"\n\n## 9.0 V1 Text-Only Baseline on Held-Out Ground Truth\n")
    section.append(f"_Generated {datetime.utcnow().isoformat()}Z by `scripts/13b_eval_v1_on_gt.py`._\n\n")
    section.append(
        "This is the direct comparison number for the v2 acoustic classifier (Section 9 / "
        "appended by `scripts/14_train_classifier_v2.py`). v1's previously reported "
        "92% accuracy / ROC-AUC 0.9774 is from a train/val split, NOT the held-out 300-item "
        "ground-truth set. This section reports v1 evaluated on the same held-out set v2 will "
        "be evaluated on, so v1↔v2 comparison is apples-to-apples.\n\n"
    )
    section.append("### Setup\n")
    section.append(f"- Model: `{MODEL_PATH}` (LogisticRegression, class_weight='balanced')\n")
    section.append(
        f"- Feature vector (389 dims): "
        f"5 lexical features `[lb, msa, strong_lb_hits, msa_ratio_core, final_score]` "
        f"+ 384-dim sentence embedding (`paraphrase-multilingual-MiniLM-L12-v2`).\n"
    )
    section.append(f"- Annotations: `{ANNOTATIONS_CSV}` ({len(rows)} rows).\n")
    section.append(f"- GT binary mapping: {GT_BINARY_MAP}; excluded labels: {sorted(GT_EXCLUDE - {None, ''})}.\n")
    section.append(
        f"- Evaluation set: {len(y)} items "
        f"(positives = {int(y.sum())}, negatives = {int((y == 0).sum())}).\n"
    )
    section.append(
        f"- Excluded: {len(excluded_labels)} (label outside binary map), "
        f"{len(missing_transcripts)} (no screening transcript), "
        f"{len(feature_errors)} (feature-build error).\n"
    )

    section.append("\n### Threshold-independent metrics\n")
    section.append(f"- ROC-AUC: {auc:.4f}\n" if auc is not None else "- ROC-AUC: undefined\n")
    section.append(f"- PR-AUC:  {pr_auc:.4f}\n" if pr_auc is not None else "- PR-AUC: undefined\n")

    section.append("\n### Per-threshold metrics\n\n")
    for m in threshold_results:
        section.append(fmt_threshold_block(m))
        section.append("\n")
    section.append(
        f"**Best threshold by macro F1:** {best['threshold']:.2f} "
        f"(macro F1 = {best['macro_f1']:.4f}, accuracy = {best['accuracy']:.4f}).\n"
    )

    section.append("\n### Notes\n")
    section.append(
        "- The threshold corresponds to the calibrated probability output of the v1 LogisticRegression. "
        "The pipeline's POTENTIAL_LB cutoff is 0.75; BORDERLINE_LB is 0.50. Both are reported above.\n"
        "- This v1 baseline is text-only — feature inputs are derived from the screening transcript only. "
        "v2 (Section 9) uses wav2vec2-xls-r-300m acoustic embeddings instead.\n"
        "- For thesis Section 7 (results), report v1 vs v2 at the same threshold (recommended: the "
        "threshold that maximizes macro F1 on each model independently — both reported in their "
        "respective FINDINGS sections).\n"
    )

    with open(FINDINGS_PATH, "a", encoding="utf-8") as f:
        f.write("".join(section))
    print("Appended Section 9.0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
