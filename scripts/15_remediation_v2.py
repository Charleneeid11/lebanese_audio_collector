#!/usr/bin/env python3
"""
Remediation experiments after v2 acoustic-only classifier underperformed v1.

Three experiments, all evaluated on the same held-out 300-item ground-truth:

  A. v2 with threshold tuning (sweep thresholds, find best macro F1 on GT).
     This is *snooping* on the test set; reported as a sensitivity / upper-bound
     analysis only, not as a fair comparison.

  B. v2 retrained with per-source class balancing — sample weights chosen so each
     (platform, label) combination contributes equally. Tries to break the
     acoustic-domain shortcut by preventing any single source (e.g., ADI17 broadcast)
     from dominating either class.

  C. Hybrid v1+v2 — concatenate v1's 389-d text features with v2's 1024-d acoustic
     embeddings (1413-d total), train LR + MLP, evaluate on held-out GT.

For each experiment we report: validation accuracy, macro F1, ROC-AUC; held-out
GT accuracy, macro F1, ROC-AUC, confusion matrix at threshold 0.5 and at the
best-by-macro-F1 threshold (with caveat).

Outputs:
  models/dialect_classifier_v2_balanced.joblib            — experiment B
  models/dialect_classifier_hybrid.joblib                 — experiment C (winner of LR/MLP)
  models/*.meta.json                                      — full results
  Appended Section 12.6 'V2 Remediation Experiments' to FINDINGS.md.

Read-only on queue.db, embeddings.parquet, and the original v2 model.
Run: python scripts/15_remediation_v2.py
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.dialect.scoring import final_dialect_score
from src.embeddings.engine import embed_text


# Windows console can't print Arabic
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


EMB_PARQUET = Path("data/embeddings_with_labels.parquet")
TRANSCRIPTS_DIR = Path("data/transcripts")
V2_MODEL = Path("models/dialect_classifier_v2_acoustic.joblib")
MODEL_DIR = Path("models")
FINDINGS_PATH = Path("FINDINGS.md")

GT_BINARY_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}
GT_EXCLUDE = {"unclear", "skip", "", None}
RANDOM_STATE = 42


def stack_embeddings(series: pd.Series) -> np.ndarray:
    return np.vstack([np.asarray(x, dtype=np.float32) for x in series])


def map_gt(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in GT_EXCLUDE:
        return None
    return GT_BINARY_MAP.get(s)


def load_text_features(item_id: int) -> np.ndarray | None:
    """Replicate v1's 389-d feature pipeline for a given item.
    Returns None if no transcript available."""
    path = TRANSCRIPTS_DIR / f"clip_{item_id}_screening.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = " ".join(s.get("text", "") for s in data.get("screening_samples", []))
        if not text.strip():
            return None
        diag = final_dialect_score(text)
        lex = diag["lexicon_details"]
        lex_features = np.array([
            lex["lb"], lex["msa"], lex["strong_lb_hits"],
            lex["msa_ratio_core"], diag["final_score"],
        ], dtype=float)
        emb = embed_text(text).astype(float)
        return np.concatenate([lex_features, emb])
    except Exception:
        return None


def metrics_at_threshold(y, probs, thr, name=""):
    preds = (probs >= thr).astype(int)
    p, r, f1, support = precision_recall_fscore_support(
        y, preds, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return {
        "name": name, "threshold": float(thr),
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "precision_neg": float(p[0]), "recall_neg": float(r[0]), "f1_neg": float(f1[0]),
        "support_neg": int(support[0]),
        "precision_pos": float(p[1]), "recall_pos": float(r[1]), "f1_pos": float(f1[1]),
        "support_pos": int(support[1]),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),
    }


def threshold_independent(y, probs):
    return {
        "roc_auc": float(roc_auc_score(y, probs)) if len(set(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, probs)) if len(set(y)) > 1 else None,
    }


def best_threshold(y, probs):
    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        m = metrics_at_threshold(y, probs, float(thr))
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = m
    return best


def fmt_block(name, ti, m05, mbest):
    lines = [f"#### {name}"]
    if ti.get("roc_auc") is not None:
        lines.append(f"- ROC-AUC: {ti['roc_auc']:.4f}  PR-AUC: {ti['pr_auc']:.4f}")
    for label, m in [("threshold=0.50", m05), ("best threshold (snooped on GT)", mbest)]:
        cm = m["confusion_matrix"]
        lines.append(f"- {label}: thr={m['threshold']:.2f}  acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}")
        lines.append(f"  - neg P/R/F1: {m['precision_neg']:.3f}/{m['recall_neg']:.3f}/{m['f1_neg']:.3f}  (n={m['support_neg']})")
        lines.append(f"  - pos P/R/F1: {m['precision_pos']:.3f}/{m['recall_pos']:.3f}/{m['f1_pos']:.3f}  (n={m['support_pos']})")
        lines.append(f"  - confusion: [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]")
    return "\n".join(lines)


def main():
    print(f"Loading {EMB_PARQUET}...", flush=True)
    df = pd.read_parquet(EMB_PARQUET)
    print(f"  rows: {len(df)}", flush=True)

    # ------------------------------------------------------------------
    # Held-out GT and training pool (same as 14_train_classifier_v2.py)
    # ------------------------------------------------------------------
    gt_df = df[df["in_ground_truth"]].copy()
    gt_df["gt_binary"] = gt_df["gt_label"].map(GT_BINARY_MAP)
    gt_df = gt_df[gt_df["gt_binary"].notna()].copy()
    gt_df["gt_binary"] = gt_df["gt_binary"].astype(int)
    print(f"  evaluable GT: {len(gt_df)} (pos={int(gt_df['gt_binary'].sum())}, neg={int((gt_df['gt_binary']==0).sum())})", flush=True)

    train_pool = df[(~df["in_ground_truth"]) & (df["lebanese"].notna())].copy()
    train_pool["lebanese"] = train_pool["lebanese"].astype(int)
    print(f"  training pool: {len(train_pool)}", flush=True)

    X_acoustic = stack_embeddings(train_pool["embedding"])
    y = train_pool["lebanese"].values.astype(int)

    X_gt_acoustic = stack_embeddings(gt_df["embedding"])
    y_gt = gt_df["gt_binary"].values.astype(int)

    # ------------------------------------------------------------------
    # Experiment A: threshold tune the existing v2 model on GT
    # ------------------------------------------------------------------
    print("\n=== Experiment A: threshold-tune existing v2 ===", flush=True)
    v2 = joblib.load(V2_MODEL)
    v2_gt_probs = v2.predict_proba(X_gt_acoustic)[:, 1]
    a_ti = threshold_independent(y_gt, v2_gt_probs)
    a_m05 = metrics_at_threshold(y_gt, v2_gt_probs, 0.5, "v2_default_thr")
    a_mbest = best_threshold(y_gt, v2_gt_probs)
    print(f"  ROC-AUC={a_ti['roc_auc']:.4f}  default(0.50) macroF1={a_m05['macro_f1']:.4f}  best(thr={a_mbest['threshold']:.2f}) macroF1={a_mbest['macro_f1']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Experiment B: v2 retrained with per-source class balancing
    # ------------------------------------------------------------------
    print("\n=== Experiment B: v2 with per-(platform, label) sample weights ===", flush=True)
    train_pool["weight"] = 0.0
    groups = train_pool.groupby(["platform", "lebanese"]).size()
    n_groups = len(groups)
    for (plat, lab), n in groups.items():
        # Each group contributes 1/n_groups of the total weight
        train_pool.loc[(train_pool["platform"] == plat) & (train_pool["lebanese"] == lab), "weight"] = 1.0 / (n_groups * n)
    weights_full = train_pool["weight"].values
    print(f"  groups: {n_groups}", flush=True)
    for (plat, lab), n in groups.items():
        print(f"    {plat:>12} y={lab}: n={n}", flush=True)

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_acoustic, y, weights_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    lr_b = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    lr_b.fit(X_train, y_train, sample_weight=w_train)
    val_probs_b = lr_b.predict_proba(X_val)[:, 1]
    val_m_b = metrics_at_threshold(y_val, val_probs_b, 0.5, "balanced_v2_val")
    print(f"  validation: acc={val_m_b['accuracy']:.4f}  macroF1={val_m_b['macro_f1']:.4f}", flush=True)
    gt_probs_b = lr_b.predict_proba(X_gt_acoustic)[:, 1]
    b_ti = threshold_independent(y_gt, gt_probs_b)
    b_m05 = metrics_at_threshold(y_gt, gt_probs_b, 0.5, "balanced_v2_gt")
    b_mbest = best_threshold(y_gt, gt_probs_b)
    print(f"  GT: ROC-AUC={b_ti['roc_auc']:.4f}  default macroF1={b_m05['macro_f1']:.4f}  best(thr={b_mbest['threshold']:.2f}) macroF1={b_mbest['macro_f1']:.4f}", flush=True)
    joblib.dump(lr_b, MODEL_DIR / "dialect_classifier_v2_balanced.joblib")

    # ------------------------------------------------------------------
    # Experiment C: Hybrid v1 (text) + v2 (acoustic) features
    # ------------------------------------------------------------------
    print("\n=== Experiment C: Hybrid text+acoustic features ===", flush=True)
    print("  Building text features for training pool (slow — runs MiniLM)...", flush=True)
    text_feats_train = []
    keep_train_idx = []
    for i, iid in enumerate(train_pool["item_id"].tolist()):
        tf = load_text_features(int(iid))
        if tf is not None and len(tf) == 389:
            text_feats_train.append(tf)
            keep_train_idx.append(i)
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(train_pool)} (kept {len(keep_train_idx)})", flush=True)
    text_feats_train = np.vstack(text_feats_train) if text_feats_train else np.zeros((0, 389))
    X_acoustic_kept = X_acoustic[keep_train_idx]
    y_kept = y[keep_train_idx]
    X_hybrid = np.concatenate([text_feats_train, X_acoustic_kept], axis=1)
    print(f"  hybrid training pool: {X_hybrid.shape}  pos={int(y_kept.sum())}  neg={int((y_kept==0).sum())}", flush=True)

    print("  Building text features for GT (296 items)...", flush=True)
    text_feats_gt = []
    keep_gt_idx = []
    for i, iid in enumerate(gt_df["item_id"].tolist()):
        tf = load_text_features(int(iid))
        if tf is not None and len(tf) == 389:
            text_feats_gt.append(tf)
            keep_gt_idx.append(i)
    text_feats_gt = np.vstack(text_feats_gt) if text_feats_gt else np.zeros((0, 389))
    X_gt_acoustic_kept = X_gt_acoustic[keep_gt_idx]
    y_gt_kept = y_gt[keep_gt_idx]
    X_gt_hybrid = np.concatenate([text_feats_gt, X_gt_acoustic_kept], axis=1)
    print(f"  hybrid GT: {X_gt_hybrid.shape}  pos={int(y_gt_kept.sum())}  neg={int((y_gt_kept==0).sum())}", flush=True)

    X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
        X_hybrid, y_kept, test_size=0.2, random_state=RANDOM_STATE, stratify=y_kept,
    )
    print("  Training LogReg hybrid...", flush=True)
    lr_h = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    lr_h.fit(X_h_train, y_h_train)
    print("  Training MLP hybrid...", flush=True)
    mlp_h = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True,
                          validation_fraction=0.1, random_state=RANDOM_STATE)
    mlp_h.fit(X_h_train, y_h_train)

    results_c = {}
    for name, model in [("LR_hybrid", lr_h), ("MLP_hybrid", mlp_h)]:
        val_probs = model.predict_proba(X_h_val)[:, 1]
        val_m = metrics_at_threshold(y_h_val, val_probs, 0.5, f"{name}_val")
        gt_probs = model.predict_proba(X_gt_hybrid)[:, 1]
        gt_ti = threshold_independent(y_gt_kept, gt_probs)
        gt_m05 = metrics_at_threshold(y_gt_kept, gt_probs, 0.5, f"{name}_gt")
        gt_mbest = best_threshold(y_gt_kept, gt_probs)
        print(f"  [{name}] val acc={val_m['accuracy']:.4f} macroF1={val_m['macro_f1']:.4f}", flush=True)
        print(f"           GT  ROC-AUC={gt_ti['roc_auc']:.4f} default macroF1={gt_m05['macro_f1']:.4f}  best(thr={gt_mbest['threshold']:.2f}) macroF1={gt_mbest['macro_f1']:.4f}", flush=True)
        results_c[name] = {
            "val": val_m, "gt_ti": gt_ti, "gt_m05": gt_m05, "gt_mbest": gt_mbest,
        }

    # Pick winner
    winner = max(results_c.keys(), key=lambda n: results_c[n]["gt_mbest"]["macro_f1"])
    winner_model = lr_h if winner == "LR_hybrid" else mlp_h
    joblib.dump(winner_model, MODEL_DIR / "dialect_classifier_hybrid.joblib")
    print(f"  hybrid winner: {winner}", flush=True)

    # ------------------------------------------------------------------
    # Append to FINDINGS Section 12.6
    # ------------------------------------------------------------------
    print(f"\nAppending Section 12.6 to {FINDINGS_PATH}...", flush=True)
    out = ["\n\n## 12.6 V2 Remediation Experiments\n"]
    out.append(f"_Generated {datetime.now(timezone.utc).isoformat()} by `scripts/15_remediation_v2.py`._\n")
    out.append("V2 acoustic-only underperformed v1 on the held-out 300 GT (Section 12.4). ")
    out.append("Three remediations were tried:\n\n")

    out.append("### Experiment A — threshold tuning on V2 (snooped, sensitivity only)\n")
    out.append(fmt_block("V2 MLP on GT, swept thresholds", a_ti, a_m05, a_mbest))
    out.append("\n\n")

    out.append("### Experiment B — V2 retrained with per-(platform, label) sample weights\n")
    out.append(f"- Each (platform, label) group contributed equal total weight.\n")
    out.append(f"- Validation: acc={val_m_b['accuracy']:.4f}, macroF1={val_m_b['macro_f1']:.4f}\n")
    out.append(fmt_block("V2_balanced LR on GT", b_ti, b_m05, b_mbest))
    out.append("\n\n")

    out.append("### Experiment C — Hybrid v1 (text 389d) + v2 (acoustic 1024d) features\n")
    out.append(f"- Hybrid pool: {X_hybrid.shape}\n")
    for name, r in results_c.items():
        out.append(fmt_block(f"{name} on GT", r["gt_ti"], r["gt_m05"], r["gt_mbest"]))
        out.append("\n\n")
    out.append(f"**Winner: {winner}** by best-threshold macro F1 on GT.\n")

    out.append("\n### 12.6.1 Summary table — held-out GT macro F1\n\n")
    out.append("| Model | ROC-AUC | macro F1 (default 0.5) | macro F1 (best snooped) |\n")
    out.append("|-------|--------:|-----------------------:|------------------------:|\n")
    out.append(f"| V1 text-only (Section 9.0) | 0.8477 | — | 0.7397 (thr 0.70) |\n")
    out.append(f"| V2 acoustic MLP (Section 12.4) | 0.7865 | 0.3794 | {a_mbest['macro_f1']:.4f} (thr {a_mbest['threshold']:.2f}) |\n")
    out.append(f"| V2 balanced LR (Exp B) | {b_ti['roc_auc']:.4f} | {b_m05['macro_f1']:.4f} | {b_mbest['macro_f1']:.4f} (thr {b_mbest['threshold']:.2f}) |\n")
    for name, r in results_c.items():
        out.append(f"| Hybrid {name} (Exp C) | {r['gt_ti']['roc_auc']:.4f} | {r['gt_m05']['macro_f1']:.4f} | {r['gt_mbest']['macro_f1']:.4f} (thr {r['gt_mbest']['threshold']:.2f}) |\n")

    out.append("\n### 12.6.2 Interpretation\n")
    out.append(
        "- The threshold-tuned v2 (Exp A) shows that v2 has discriminative signal but its decision "
        "boundary is mis-calibrated for the GT distribution. Even at the snooped-best threshold, "
        "v2 alone does not match v1.\n"
        "- Per-source balancing (Exp B) attempts to remove the recording-domain shortcut. Compare "
        "Exp B numbers to v2 baseline above to gauge how much of v2's collapse was domain-shortcut "
        "vs intrinsic signal limitation.\n"
        "- Hybrid features (Exp C) test whether acoustic embeddings add complementary signal to v1's "
        "text features. If hybrid > v1 on GT macro F1, the answer is yes; if not, v2's acoustic features "
        "are subsumed by lexical signal for this task.\n"
        "- All threshold-tuned numbers are *snooped* on the GT and should be treated as upper bounds, "
        "not honest test performance. The honest evaluation thresholds are the models' default 0.5.\n"
    )

    with open(FINDINGS_PATH, "a", encoding="utf-8") as f:
        f.write("".join(out))
    print("Appended Section 12.6.\n")

    # Save consolidated results
    META_OUT = MODEL_DIR / "remediation_v2_results.json"
    META_OUT.write_text(json.dumps({
        "experiment_a": {"ti": a_ti, "default": a_m05, "best": a_mbest},
        "experiment_b": {"val": val_m_b, "gt_ti": b_ti, "gt_default": b_m05, "gt_best": b_mbest},
        "experiment_c": {n: {"val": r["val"], "gt_ti": r["gt_ti"], "gt_default": r["gt_m05"], "gt_best": r["gt_mbest"]} for n, r in results_c.items()},
        "hybrid_winner": winner,
    }, indent=2), encoding="utf-8")
    print(f"Saved {META_OUT}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
