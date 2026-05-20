#!/usr/bin/env python3
"""
Unified benchmark harness for the Lebanese cross-domain DID evaluation.

Standardized interface: each system contributes a `predict_*` function that
returns `{item_id: prob_lebanese}` over the 296-item held-out GT.

Per-system pause/resume:
  - Each system's raw predictions are saved to data/benchmark_predictions/<name>.json
  - Re-running the harness skips any system whose prediction file already exists
    (unless --force or --only <name> is passed).
  - Safe to Ctrl-C between systems. If interrupted *during* a system, that
    system's file is not written and will be re-run cleanly next time.

Metrics computed per system:
  - Accuracy / macro F1 / per-class P/R/F1 at threshold 0.5
  - Macro F1 at best snooped threshold + the threshold value
  - ROC-AUC, PR-AUC
  - Percentile bootstrap 95% CI (n=1000) on macro F1 @ 0.5 and ROC-AUC
  - Confusion matrix @ 0.5

Outputs:
  data/benchmark_results.csv         — one row per system
  data/benchmark_predictions/*.json  — per-system raw probabilities (item_id -> prob)

Usage:
  python scripts/20_benchmark_harness.py                 # run all systems
  python scripts/20_benchmark_harness.py --force         # re-run all
  python scripts/20_benchmark_harness.py --only v1_text_only
  python scripts/20_benchmark_harness.py --skip v25_finetuned
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# GT loading (shared)
# ---------------------------------------------------------------------------

def load_gt_items() -> list[dict]:
    """Read annotations.csv, return list of evaluable GT items."""
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


# ---------------------------------------------------------------------------
# Predict functions per system
# ---------------------------------------------------------------------------

def predict_v1_text_only(items: list[dict]) -> dict[int, float]:
    """V1: 5 lexical features + 384-d sentence embedding -> LR."""
    import joblib
    from src.cfg import Settings
    from src.dialect.scoring import final_dialect_score
    from src.embeddings.engine import embed_text

    settings = Settings.load()
    transcripts_dir = Path(settings.transcription.transcripts_dir)
    model = joblib.load("models/dialect_classifier.joblib")

    probs: dict[int, float] = {}
    for it in items:
        iid = it["item_id"]
        tpath = transcripts_dir / f"clip_{iid}_screening.json"
        if not tpath.exists():
            continue
        try:
            data = json.loads(tpath.read_text(encoding="utf-8"))
            text = " ".join(s.get("text", "") for s in (data.get("screening_samples") or []))
            if not text.strip():
                continue
            diagnostics = final_dialect_score(text)
            lex = diagnostics["lexicon_details"]
            lex_features = np.array([
                lex["lb"], lex["msa"], lex["strong_lb_hits"],
                lex["msa_ratio_core"], diagnostics["final_score"],
            ], dtype=float)
            emb = embed_text(text).astype(float)
            fv = np.concatenate([lex_features, emb]).reshape(1, -1)
            probs[iid] = float(model.predict_proba(fv)[0, 1])
        except Exception as e:
            print(f"  [warn] v1 failed for item {iid}: {e}", flush=True)
            continue
    return probs


def _load_v2_embeddings_for_gt(items: list[dict]) -> tuple[np.ndarray, list[int]]:
    """Load V2 acoustic embeddings (1024-d) for the GT items that have them."""
    df = pd.read_parquet("data/embeddings_with_labels.parquet")
    df_gt = df[df["item_id"].isin([it["item_id"] for it in items])]
    if len(df_gt) == 0:
        return np.zeros((0, 1024)), []
    embs = np.stack(df_gt["embedding"].values).astype(np.float32)
    iids = df_gt["item_id"].astype(int).tolist()
    return embs, iids


def predict_v2_frozen_mlp(items: list[dict]) -> dict[int, float]:
    """V2: frozen XLS-R 1024-d -> MLP. Uses scripts/14_train_classifier_v2.py output."""
    import joblib
    model = joblib.load("models/dialect_classifier_v2_acoustic.joblib")
    embs, iids = _load_v2_embeddings_for_gt(items)
    if len(iids) == 0:
        return {}
    p = model.predict_proba(embs)[:, 1]
    return dict(zip(iids, p.astype(float).tolist()))


def predict_v2_balanced(items: list[dict]) -> dict[int, float]:
    """V2 + per-(platform, label) sample-weighted LR. Section 12.6 Exp B."""
    import joblib
    model = joblib.load("models/dialect_classifier_v2_balanced.joblib")
    embs, iids = _load_v2_embeddings_for_gt(items)
    if len(iids) == 0:
        return {}
    p = model.predict_proba(embs)[:, 1]
    return dict(zip(iids, p.astype(float).tolist()))


def predict_hybrid_v1v2_mlp(items: list[dict]) -> dict[int, float]:
    """Hybrid: V1 text features (389-d) + V2 acoustic (1024-d) -> MLP. Section 12.6 Exp C."""
    import joblib
    from src.cfg import Settings
    from src.dialect.scoring import final_dialect_score
    from src.embeddings.engine import embed_text

    model = joblib.load("models/dialect_classifier_hybrid.joblib")

    settings = Settings.load()
    transcripts_dir = Path(settings.transcription.transcripts_dir)

    embs, emb_iids = _load_v2_embeddings_for_gt(items)
    emb_by_iid = dict(zip(emb_iids, embs))

    probs: dict[int, float] = {}
    for it in items:
        iid = it["item_id"]
        if iid not in emb_by_iid:
            continue
        tpath = transcripts_dir / f"clip_{iid}_screening.json"
        if not tpath.exists():
            continue
        try:
            data = json.loads(tpath.read_text(encoding="utf-8"))
            text = " ".join(s.get("text", "") for s in (data.get("screening_samples") or []))
            if not text.strip():
                continue
            diagnostics = final_dialect_score(text)
            lex = diagnostics["lexicon_details"]
            lex_features = np.array([
                lex["lb"], lex["msa"], lex["strong_lb_hits"],
                lex["msa_ratio_core"], diagnostics["final_score"],
            ], dtype=float)
            emb_text = embed_text(text).astype(float)
            text_features = np.concatenate([lex_features, emb_text])  # 389-d
            full = np.concatenate([text_features, emb_by_iid[iid]]).reshape(1, -1)
            probs[iid] = float(model.predict_proba(full)[0, 1])
        except Exception as e:
            print(f"  [warn] hybrid failed for item {iid}: {e}", flush=True)
            continue
    return probs


def predict_v25_finetuned(items: list[dict]) -> dict[int, float]:
    """V2.5: locally fine-tuned XLS-R-300m. Loads HF model from models/xlsr_finetuned/."""
    import librosa
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    MODEL_DIR = Path("models/xlsr_finetuned")
    CLIPS_DIR = Path("data/audio_clips_compact")
    SAMPLE_RATE = 16000
    MAX_LEN = SAMPLE_RATE * 10

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_DIR))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"  [v25] device={device}", flush=True)

    probs: dict[int, float] = {}
    n = len(items)
    for i, it in enumerate(items):
        iid = it["item_id"]
        clip = CLIPS_DIR / f"item_{iid}.mp3"
        if not clip.exists():
            continue
        try:
            y_audio, _ = librosa.load(str(clip), sr=SAMPLE_RATE, mono=True)
            if y_audio.shape[0] > MAX_LEN:
                y_audio = y_audio[:MAX_LEN]
            inputs = feature_extractor(y_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            iv = inputs["input_values"].to(device)
            am = inputs.get("attention_mask")
            if am is not None:
                am = am.to(device)
            with torch.no_grad():
                out = model(input_values=iv, attention_mask=am)
            p = torch.softmax(out.logits, dim=-1)[0, 1].item()
            probs[iid] = float(p)
        except Exception as e:
            print(f"  [warn] v25 failed for item {iid}: {e}", flush=True)
            continue
        if (i + 1) % 25 == 0:
            print(f"  [v25] {i+1}/{n}", flush=True)
    return probs


SYSTEMS: dict[str, tuple[str, callable]] = {
    # name: (family, predict_fn)
    "v1_text_only": ("lexical", predict_v1_text_only),
    "v2_frozen_mlp": ("acoustic", predict_v2_frozen_mlp),
    "v2_balanced": ("acoustic", predict_v2_balanced),
    "hybrid_v1v2_mlp": ("hybrid", predict_hybrid_v1v2_mlp),
    "v25_finetuned": ("acoustic", predict_v25_finetuned),
}


# ---------------------------------------------------------------------------
# Metrics + bootstrap CIs
# ---------------------------------------------------------------------------

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
    # Align predictions to items (drop items with no prediction)
    aligned = [(it["y"], it["item_id"], it["platform"], probs_by_iid[it["item_id"]])
               for it in items if it["item_id"] in probs_by_iid]
    if not aligned:
        return {"system": system_name, "family": family, "n_evaluated": 0, "error": "no predictions"}
    y = np.array([a[0] for a in aligned], dtype=int)
    p = np.array([a[3] for a in aligned], dtype=float)

    # Default-threshold metrics
    preds_default = (p >= DEFAULT_THR).astype(int)
    pr, rc, f1, support = precision_recall_fscore_support(y, preds_default, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, preds_default, labels=[0, 1])

    # Threshold sweep + best
    sweep = [(thr, _macro_f1_at_thr(y, p, thr)) for thr in THRESHOLD_SWEEP]
    best_thr, best_mf1 = max(sweep, key=lambda kv: kv[1])

    # ROC + PR
    roc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")

    # Bootstrap CIs (default-threshold macro F1 + ROC-AUC)
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


# ---------------------------------------------------------------------------
# CSV upsert + predictions persistence
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-run all systems even if predictions exist")
    ap.add_argument("--only", action="append", default=[], help="Only run named system(s); repeatable")
    ap.add_argument("--skip", action="append", default=[], help="Skip named system(s); repeatable")
    args = ap.parse_args()

    items = load_gt_items()
    print(f"[gt] loaded {len(items)} evaluable GT items "
          f"(pos={sum(1 for it in items if it['y']==1)}, "
          f"neg={sum(1 for it in items if it['y']==0)})", flush=True)

    selected = list(SYSTEMS.items())
    if args.only:
        selected = [(n, v) for n, v in selected if n in args.only]
    if args.skip:
        selected = [(n, v) for n, v in selected if n not in args.skip]

    print(f"[plan] {len(selected)} system(s): {[n for n, _ in selected]}", flush=True)

    for system_name, (family, predict_fn) in selected:
        preds_path = PREDS_DIR / f"{system_name}.json"
        t0 = time.time()
        if preds_path.exists() and not args.force:
            print(f"\n[skip ] {system_name} ({family})  - predictions on disk, loading", flush=True)
            probs_by_iid = load_predictions(preds_path)
        else:
            print(f"\n[start] {system_name} ({family})", flush=True)
            try:
                probs_by_iid = predict_fn(items)
            except Exception as e:
                print(f"[error] {system_name} crashed: {type(e).__name__}: {e}", flush=True)
                continue
            save_predictions(preds_path, system_name, items, probs_by_iid)
            print(f"[save ] {preds_path}  ({len(probs_by_iid)}/{len(items)} predictions, {time.time()-t0:.1f}s)", flush=True)

        row = compute_metrics(system_name, family, items, probs_by_iid)
        upsert_csv_row(RESULTS_CSV, row)
        print(
            f"[done ] {system_name}  n={row.get('n_evaluated')}  "
            f"macroF1@0.5={row.get('macro_f1_at_0.5'):.4f}  "
            f"(CI95 {row.get('macro_f1_ci95_lo'):.3f}-{row.get('macro_f1_ci95_hi'):.3f})  "
            f"best={row.get('macro_f1_at_best'):.4f} @thr={row.get('best_threshold'):.2f}  "
            f"ROC-AUC={row.get('roc_auc'):.4f}  "
            f"(CI95 {row.get('roc_auc_ci95_lo'):.3f}-{row.get('roc_auc_ci95_hi'):.3f})",
            flush=True,
        )

    print(f"\n[final] {RESULTS_CSV} updated.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
