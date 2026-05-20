#!/usr/bin/env python3
"""
Error analysis of v1 (text-only) classifier on the held-out 300-item ground-truth
set. Identifies which items v1 gets wrong and surfaces patterns by source platform,
audio duration, and transcript length.

Inputs (read-only):
  models/dialect_classifier.joblib                — v1 LogisticRegression
  data/annotations.csv                             — 300 GT items
  data/transcripts/clip_<item_id>_screening.json   — per-item screening transcripts
  data/queue.db                                    — for platform / duration / metadata join

Outputs:
  data/v1_error_analysis.json                      — full breakdown
  FINDINGS.md  ← appended Section 9.0.1 "V1 Error Patterns"

Methodology:
  - Replicates scripts/05_train_dialect_model.py and scripts/13b_eval_v1_on_gt.py
    feature pipeline exactly (5 lexical + 384-d embedding = 389 dims).
  - Uses threshold 0.70 (the best-by-macro-F1 threshold from FINDINGS 9.0). The
    threshold can be overridden via --threshold for sensitivity analysis.
  - Classifies each evaluable item as one of:
      correct_pos  — model >= threshold, GT == 1
      correct_neg  — model <  threshold, GT == 0
      false_pos    — model >= threshold, GT == 0  (over-eager Lebanese prediction)
      false_neg    — model <  threshold, GT == 1  (missed Lebanese)
    Items with gt_label in {unclear, skip, ""} are excluded from the binary eval.

Breakdowns reported:
  - By source platform (youtube / podcast / podcast_rss / adi17 / fleurs / ...)
  - By audio duration bucket (0-60s, 60-300s, 300-1200s, >1200s)
  - By transcript length bucket in word counts (short <50, medium 50-200, long >=200)

Top-10 lists:
  - Most confidently wrong false positives (highest prob, GT == not_lebanese)
  - Most confidently wrong false negatives (lowest prob, GT == lebanese/mostly_lebanese)
  - Each entry includes item_id, platform, source_status, audio_path, prob, gt_label,
    transcript_word_count, transcript_snippet (first 200 chars).

Read-only on queue.db, the v1 model, transcripts, and annotations.csv.
Only side effects:
  - Writes data/v1_error_analysis.json (overwritten on re-run)
  - Appends Section 9.0.1 to FINDINGS.md (idempotent: if a section with the same
    heading already exists, a new dated re-run section is appended below it; we do
    NOT silently overwrite history, since FINDINGS is a research log)

Run: python scripts/13c_v1_error_analysis.py
     python scripts/13c_v1_error_analysis.py --threshold 0.50
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Windows cp1252 console can't print Arabic; reconfigure to UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import joblib
import numpy as np

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.cfg import Settings
from src.db import DB, QueueItem
from src.dialect.scoring import final_dialect_score
from src.embeddings.engine import embed_text


MODEL_PATH = Path("models/dialect_classifier.joblib")
ANNOTATIONS_CSV = Path("data/annotations.csv")
OUT_JSON = Path("data/v1_error_analysis.json")
FINDINGS_PATH = Path("FINDINGS.md")

GT_BINARY_MAP = {
    "lebanese": 1,
    "mostly_lebanese": 1,
    "not_lebanese": 0,
}
GT_EXCLUDE = {"unclear", "skip", "", None}

DEFAULT_THRESHOLD = 0.70   # best macro-F1 threshold from FINDINGS Section 9.0

DURATION_BUCKETS = [
    ("unknown",   lambda d: d is None),
    ("0-60s",     lambda d: d is not None and d <= 60),
    ("60-300s",   lambda d: d is not None and 60 < d <= 300),
    ("300-1200s", lambda d: d is not None and 300 < d <= 1200),
    (">1200s",    lambda d: d is not None and d > 1200),
]

TRANSCRIPT_LEN_BUCKETS = [
    ("short (<50)",       lambda n: n < 50),
    ("medium (50-200)",   lambda n: 50 <= n < 200),
    ("long (>=200)",      lambda n: n >= 200),
]


def load_transcript_text(item_id: int, transcripts_dir: Path) -> str | None:
    """Mirror of scripts/05_train_dialect_model.py:load_transcript_text."""
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
    """Mirror of scripts/05_train_dialect_model.py:build_feature_vector."""
    lex = diagnostics["lexicon_details"]
    lex_features = np.array(
        [
            lex["lb"],
            lex["msa"],
            lex["strong_lb_hits"],
            lex["msa_ratio_core"],
            diagnostics["final_score"],
        ],
        dtype=float,
    )
    embedding = embed_text(text).astype(float)
    return np.concatenate([lex_features, embedding])


def map_label(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in GT_EXCLUDE:
        return None
    return GT_BINARY_MAP.get(s)


def bucket_for(value, buckets):
    for name, predicate in buckets:
        if predicate(value):
            return name
    return "uncategorized"


def load_annotations() -> list[dict]:
    rows = []
    with open(ANNOTATIONS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def fetch_db_metadata(item_ids: list[int]) -> dict[int, dict]:
    settings = Settings.load()
    db = DB(settings.db_url)
    out: dict[int, dict] = {}
    with Session(db.engine) as session:
        rows = session.scalars(
            select(QueueItem).where(QueueItem.id.in_(item_ids))
        ).all()
        for r in rows:
            out[r.id] = {
                "platform": r.platform,
                "status": r.status,
                "duration_seconds": r.duration_seconds,
                "audio_path": r.audio_path,
                "source_metadata": r.source_metadata or {},
            }
    return out


def classify_outcome(prob: float, y_true: int, threshold: float) -> str:
    pred = 1 if prob >= threshold else 0
    if pred == y_true:
        return "correct_pos" if pred == 1 else "correct_neg"
    return "false_pos" if pred == 1 else "false_neg"


def empty_counts() -> dict:
    return {"total": 0, "correct_pos": 0, "correct_neg": 0,
            "false_pos": 0, "false_neg": 0, "support_pos": 0, "support_neg": 0}


def add_to_counts(c: dict, outcome: str, y_true: int) -> None:
    c["total"] += 1
    c[outcome] = c.get(outcome, 0) + 1
    if y_true == 1:
        c["support_pos"] += 1
    else:
        c["support_neg"] += 1


def finalize_counts(c: dict) -> dict:
    """Add derived rates."""
    out = dict(c)
    sup_pos = max(c["support_pos"], 1)
    sup_neg = max(c["support_neg"], 1)
    out["fp_rate"] = round(c["false_pos"] / sup_neg, 4)   # FP per negative
    out["fn_rate"] = round(c["false_neg"] / sup_pos, 4)   # FN per positive
    out["accuracy"] = round((c["correct_pos"] + c["correct_neg"]) / max(c["total"], 1), 4)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Decision threshold on v1 prob (default: {DEFAULT_THRESHOLD}).")
    args = parser.parse_args(argv[1:])
    threshold = args.threshold

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
    n_features_expected = model.coef_.shape[1] if hasattr(model, "coef_") else None
    print(f"  classes={model.classes_.tolist()} expected_dim={n_features_expected}")

    print(f"\nLoading annotations from {ANNOTATIONS_CSV}...")
    rows = load_annotations()
    item_ids = []
    for r in rows:
        try:
            item_ids.append(int(r["item_id"]))
        except (KeyError, ValueError):
            pass
    print(f"  rows: {len(rows)} | parsable item_ids: {len(item_ids)}")

    print("Fetching queue.db metadata for annotated items (read-only)...")
    db_meta = fetch_db_metadata(item_ids)
    print(f"  matched in DB: {len(db_meta)}")

    # ------------------------------------------------------------------
    # Build per-item record: prediction + true label + metadata + word count
    # ------------------------------------------------------------------
    per_item: list[dict] = []
    skipped_label: list[tuple[int, str | None]] = []
    skipped_no_transcript: list[int] = []
    skipped_feature_error: list[tuple[int, str]] = []

    for r in rows:
        try:
            iid = int(r["item_id"])
        except (KeyError, ValueError):
            continue
        gt_raw = (r.get("ground_truth") or "").strip().lower() or None
        bin_label = map_label(gt_raw)
        if bin_label is None:
            skipped_label.append((iid, gt_raw))
            continue
        text = load_transcript_text(iid, transcripts_dir)
        if not text or not text.strip():
            skipped_no_transcript.append(iid)
            continue
        try:
            diagnostics = final_dialect_score(text)
            fv = build_feature_vector(text, diagnostics)
        except Exception as e:
            skipped_feature_error.append((iid, f"{type(e).__name__}: {e}"))
            continue
        prob = float(model.predict_proba(fv.reshape(1, -1))[0, 1])
        outcome = classify_outcome(prob, bin_label, threshold)
        meta = db_meta.get(iid, {})
        word_count = len(text.split())
        per_item.append({
            "item_id": iid,
            "gt_label_raw": gt_raw,
            "gt_label_binary": bin_label,
            "prob": round(prob, 4),
            "outcome": outcome,
            "platform": meta.get("platform"),
            "source_status": meta.get("status"),
            "duration_seconds": meta.get("duration_seconds"),
            "audio_path": meta.get("audio_path"),
            "source_metadata": meta.get("source_metadata", {}),
            "transcript_word_count": word_count,
            "transcript_snippet": text.strip()[:200],
        })

    print(
        f"\nEvaluable: {len(per_item)} | excluded by label: {len(skipped_label)} "
        f"| missing transcripts: {len(skipped_no_transcript)} "
        f"| feature errors: {len(skipped_feature_error)}"
    )
    if not per_item:
        print("ERROR: nothing evaluable. Aborting.")
        return 1

    # ------------------------------------------------------------------
    # Aggregate breakdowns
    # ------------------------------------------------------------------
    overall = empty_counts()
    by_platform: dict[str, dict] = defaultdict(empty_counts)
    by_status: dict[str, dict] = defaultdict(empty_counts)
    by_duration: dict[str, dict] = defaultdict(empty_counts)
    by_transcript_len: dict[str, dict] = defaultdict(empty_counts)

    for it in per_item:
        outcome = it["outcome"]
        y = it["gt_label_binary"]
        add_to_counts(overall, outcome, y)
        add_to_counts(by_platform[it["platform"] or "unknown"], outcome, y)
        add_to_counts(by_status[it["source_status"] or "unknown"], outcome, y)
        add_to_counts(by_duration[bucket_for(it["duration_seconds"], DURATION_BUCKETS)], outcome, y)
        add_to_counts(by_transcript_len[bucket_for(it["transcript_word_count"], TRANSCRIPT_LEN_BUCKETS)], outcome, y)

    overall_f = finalize_counts(overall)
    by_platform_f = {k: finalize_counts(v) for k, v in by_platform.items()}
    by_status_f = {k: finalize_counts(v) for k, v in by_status.items()}
    by_duration_f = {k: finalize_counts(v) for k, v in by_duration.items()}
    by_transcript_len_f = {k: finalize_counts(v) for k, v in by_transcript_len.items()}

    # Top-10 most-confidently-wrong
    fps = sorted([it for it in per_item if it["outcome"] == "false_pos"],
                 key=lambda x: -x["prob"])[:10]
    fns = sorted([it for it in per_item if it["outcome"] == "false_neg"],
                 key=lambda x: x["prob"])[:10]

    # ------------------------------------------------------------------
    # Print to stdout
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nv1 error analysis @ threshold={threshold}\n{'=' * 60}")
    print(f"\nOverall: {overall_f}")
    print(f"\nBy platform:")
    for k, v in sorted(by_platform_f.items(), key=lambda x: -x[1]["total"]):
        print(f"  {k:<14} total={v['total']:>3} correct={v['correct_pos']+v['correct_neg']:>3} "
              f"FP={v['false_pos']:>3} FN={v['false_neg']:>3} acc={v['accuracy']:.3f}")
    print(f"\nBy source_status (pipeline status when annotated):")
    for k, v in sorted(by_status_f.items(), key=lambda x: -x[1]["total"]):
        print(f"  {k:<16} total={v['total']:>3} FP={v['false_pos']:>3} FN={v['false_neg']:>3} acc={v['accuracy']:.3f}")
    print(f"\nBy duration:")
    for k, v in by_duration_f.items():
        print(f"  {k:<10} total={v['total']:>3} FP={v['false_pos']:>3} FN={v['false_neg']:>3} acc={v['accuracy']:.3f}")
    print(f"\nBy transcript length:")
    for k, v in by_transcript_len_f.items():
        print(f"  {k:<18} total={v['total']:>3} FP={v['false_pos']:>3} FN={v['false_neg']:>3} acc={v['accuracy']:.3f}")
    print(f"\nTop-10 false positives (highest prob, GT=not_lebanese):")
    for it in fps:
        print(f"  id={it['item_id']:>5} platform={it['platform']:<12} "
              f"prob={it['prob']:.3f} gt={it['gt_label_raw']:<15} "
              f"snippet={it['transcript_snippet'][:80]!r}")
    print(f"\nTop-10 false negatives (lowest prob, GT in lebanese/mostly_lebanese):")
    for it in fns:
        print(f"  id={it['item_id']:>5} platform={it['platform']:<12} "
              f"prob={it['prob']:.3f} gt={it['gt_label_raw']:<15} "
              f"snippet={it['transcript_snippet'][:80]!r}")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "threshold": threshold,
        "n_total_annotations": len(rows),
        "n_evaluable": len(per_item),
        "n_excluded_label": len(skipped_label),
        "n_missing_transcripts": len(skipped_no_transcript),
        "n_feature_errors": len(skipped_feature_error),
        "overall": overall_f,
        "by_platform": by_platform_f,
        "by_source_status": by_status_f,
        "by_duration": by_duration_f,
        "by_transcript_length": by_transcript_len_f,
        "top_false_positives": fps,
        "top_false_negatives": fns,
        "skipped_label_examples": [
            {"item_id": iid, "raw_label": lab} for iid, lab in skipped_label[:10]
        ],
        "missing_transcript_item_ids": skipped_no_transcript[:20],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT_JSON} ({OUT_JSON.stat().st_size/1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Append FINDINGS Section 9.0.1
    # ------------------------------------------------------------------
    print(f"\nAppending Section 9.0.1 to {FINDINGS_PATH}...")
    section: list[str] = []
    section.append("\n\n## 9.0.1 V1 Error Patterns\n")
    section.append(f"_Generated {payload['generated_at']} by `scripts/13c_v1_error_analysis.py` "
                   f"(threshold={threshold})._\n\n")
    section.append(
        "Building on Section 9.0, this subsection breaks down v1's errors on the held-out "
        "ground-truth set by source platform, audio duration, transcript length, and identifies "
        "the most confidently wrong items in each direction. The breakdowns are intended to "
        "(a) characterize where v1 fails for the thesis discussion section, and (b) provide a "
        "diagnostic baseline against which v2's error patterns can be compared.\n\n"
    )
    section.append(f"### Setup\n")
    section.append(f"- Model: `{MODEL_PATH}` (v1 LogisticRegression).\n")
    section.append(f"- Decision threshold: **{threshold}** (best macro-F1 in Section 9.0).\n")
    section.append(f"- Evaluable items: {payload['n_evaluable']} of {payload['n_total_annotations']} "
                   f"({payload['n_excluded_label']} excluded by label, "
                   f"{payload['n_missing_transcripts']} missing transcripts, "
                   f"{payload['n_feature_errors']} feature-build errors).\n")

    section.append("\n### Overall\n")
    o = overall_f
    section.append(f"- Total: {o['total']}, correct: {o['correct_pos']+o['correct_neg']}, "
                   f"false-positives: {o['false_pos']}, false-negatives: {o['false_neg']}.\n")
    section.append(f"- Accuracy: {o['accuracy']:.4f} | FP rate (per neg): {o['fp_rate']:.3f} | "
                   f"FN rate (per pos): {o['fn_rate']:.3f}.\n")
    section.append(f"- Support: {o['support_pos']} positives, {o['support_neg']} negatives.\n")

    def render_breakdown(title: str, table: dict[str, dict]) -> None:
        section.append(f"\n### {title}\n")
        section.append("| Bucket | Total | Correct | FP | FN | Accuracy | FP rate | FN rate |\n")
        section.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for k, v in sorted(table.items(), key=lambda x: -x[1]["total"]):
            corr = v["correct_pos"] + v["correct_neg"]
            section.append(
                f"| {k} | {v['total']} | {corr} | {v['false_pos']} | {v['false_neg']} | "
                f"{v['accuracy']:.3f} | {v['fp_rate']:.3f} | {v['fn_rate']:.3f} |\n"
            )

    render_breakdown("By source platform", by_platform_f)
    render_breakdown("By source_status (pipeline status when annotated)", by_status_f)
    render_breakdown("By audio duration", by_duration_f)
    render_breakdown("By transcript word count", by_transcript_len_f)

    section.append("\n### Top-10 most-confidently-wrong false positives\n")
    section.append("(model confidently said Lebanese, ground truth said not_lebanese)\n\n")
    section.append("| item_id | platform | prob | GT | transcript snippet |\n")
    section.append("|---|---|---:|---|---|\n")
    for it in fps:
        snip = it["transcript_snippet"].replace("\n", " ").replace("|", "\\|")[:120]
        section.append(f"| {it['item_id']} | {it['platform']} | {it['prob']:.3f} | "
                       f"{it['gt_label_raw']} | {snip} |\n")

    section.append("\n### Top-10 most-confidently-wrong false negatives\n")
    section.append("(model confidently said not Lebanese, ground truth said lebanese or mostly_lebanese)\n\n")
    section.append("| item_id | platform | prob | GT | transcript snippet |\n")
    section.append("|---|---|---:|---|---|\n")
    for it in fns:
        snip = it["transcript_snippet"].replace("\n", " ").replace("|", "\\|")[:120]
        section.append(f"| {it['item_id']} | {it['platform']} | {it['prob']:.3f} | "
                       f"{it['gt_label_raw']} | {snip} |\n")

    section.append(
        "\n### Interpretation hooks (to fill in when reviewing the table above)\n"
        "- If FP rate is high on `youtube` items in particular, that aligns with the WEAK_POSITIVE "
        "noise quantified in Section 5.2 (Lebanese channels post mixed content; v1 over-trusts the "
        "channel signal via the lexical features).\n"
        "- If FP rate is high on `podcast` / `podcast_rss` items, the lexicon-overlap problem "
        "(pan-Arabic words inflating `lb` count) is the likely cause; v2 acoustic features should "
        "address this.\n"
        "- If FN rate is concentrated in short transcripts, weak lexical signal is the likely "
        "cause; v2 acoustic features (which see audio rather than transcribed text) may help by "
        "using prosodic and phonetic cues that survive short clips.\n"
        "- Compare the v2 numbers in Section 12 against the breakdowns here to characterize where "
        "the acoustic model improves and where it does not.\n"
    )

    with open(FINDINGS_PATH, "a", encoding="utf-8") as f:
        f.write("".join(section))
    print("Appended Section 9.0.1.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
