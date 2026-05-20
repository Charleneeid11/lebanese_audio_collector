#!/usr/bin/env python3
"""
Whisper-LID free-lunch baseline.

Hypothesis: Whisper's built-in language probability (`language_probability` field
in screening transcripts) is NOT a useful Lebanese-vs-other signal because all
items in the corpus are intended to be Arabic, and Whisper does not distinguish
Arabic dialects at the language-ID level.

Method:
  - For each GT item, average `language_probability` across screening chunks
    where Whisper predicted `language == "ar"`. For chunks where Whisper
    predicted a non-Arabic language, treat as 0.0 (not Arabic).
  - Use the resulting prob_arabic as a "Lebanese" probability.
  - Compute the standard benchmark metrics + bootstrap CIs.

Expected outcome: near-random performance. The point of including this baseline
is to demonstrate explicitly that Lebanese detection is not a free byproduct
of Whisper's language ID - it requires dialect-specific modeling.

Adds a single row `whisper_lid_arabic_prob` to data/benchmark_results.csv.

Outputs:
  data/benchmark_predictions/whisper_lid_arabic_prob.json
  data/benchmark_results.csv  (row upserted)
  FINDINGS.md  Section 18 appended
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_harness_utils import (
    PREDS_DIR, RESULTS_CSV, compute_metrics, load_gt_items, save_predictions, upsert_csv_row,
)
from src.cfg import Settings

FINDINGS_PATH = Path("FINDINGS.md")
SYSTEM_NAME = "whisper_lid_arabic_prob"


def main() -> int:
    settings = Settings.load()
    transcripts_dir = Path(settings.transcription.transcripts_dir)

    items = load_gt_items()
    print(f"GT items: {len(items)}", flush=True)

    probs: dict[int, float] = {}
    n_missing = 0
    n_no_arabic = 0
    n_partial = 0
    chunk_lang_distribution: dict[str, int] = {}

    for it in items:
        iid = it["item_id"]
        tpath = transcripts_dir / f"clip_{iid}_screening.json"
        if not tpath.exists():
            n_missing += 1
            continue
        try:
            data = json.loads(tpath.read_text(encoding="utf-8"))
        except Exception:
            n_missing += 1
            continue
        samples = data.get("screening_samples") or []
        if not samples:
            n_missing += 1
            continue
        prob_arabic_per_chunk = []
        for s in samples:
            lang = (s.get("language") or "").strip().lower()
            lp = float(s.get("language_probability") or 0.0)
            chunk_lang_distribution[lang] = chunk_lang_distribution.get(lang, 0) + 1
            if lang == "ar":
                prob_arabic_per_chunk.append(lp)
            else:
                prob_arabic_per_chunk.append(0.0)
        avg = sum(prob_arabic_per_chunk) / len(prob_arabic_per_chunk)
        probs[iid] = float(avg)
        if any(p == 0.0 for p in prob_arabic_per_chunk):
            n_partial += 1
        if avg == 0.0:
            n_no_arabic += 1

    print(f"\nChunk-level language distribution across GT screening clips:")
    for lang, n in sorted(chunk_lang_distribution.items(), key=lambda kv: -kv[1]):
        print(f"  {lang or '(none)':<8} {n}")

    print(f"\nItems with at least one non-Arabic chunk: {n_partial}")
    print(f"Items where Whisper predicted no Arabic at all: {n_no_arabic}")
    print(f"Items missing transcripts: {n_missing}")

    if not probs:
        print("ERROR: no items scored. Aborting.")
        return 1

    save_predictions(PREDS_DIR / f"{SYSTEM_NAME}.json", SYSTEM_NAME, items, probs)
    row = compute_metrics(SYSTEM_NAME, "acoustic_meta", items, probs)
    upsert_csv_row(RESULTS_CSV, row)
    print(
        f"\n[done] {SYSTEM_NAME}: macroF1@0.5={row.get('macro_f1_at_0.5'):.4f}  "
        f"best={row.get('macro_f1_at_best'):.4f}@thr={row.get('best_threshold'):.2f}  "
        f"ROC-AUC={row.get('roc_auc'):.4f}",
        flush=True,
    )

    # FINDINGS append
    section = ["\n\n## 18. Whisper-LID Free-Lunch Baseline\n"]
    section.append(f"_Generated {datetime.now(timezone.utc).isoformat()} by `scripts/23_whisper_lid_baseline.py`. ROADMAP v2 Day 2._\n\n")
    section.append("### 18.1 Question\n")
    section.append(
        "Whisper's built-in language probability is a free byproduct of every transcription. "
        "Is it a useful signal for Lebanese detection? The expected answer is *no* because "
        "Whisper's language ID operates at the language level (Arabic vs French vs English) "
        "rather than the dialect level (Lebanese vs Egyptian). But the experiment is cheap "
        "and the answer is informative either way - it tells future researchers whether "
        "Whisper's existing outputs can be repurposed.\n\n"
    )
    section.append("### 18.2 Method\n")
    section.append(
        "For each GT item, compute the average `language_probability` across its 3 screening "
        "chunks where Whisper predicted `language == \"ar\"`. Chunks predicting a non-Arabic "
        "language contribute 0.0 to the average. The resulting `prob_arabic` is used as the "
        "model's \"Lebanese probability\" and evaluated through the standard benchmark harness.\n\n"
    )
    section.append("### 18.3 Chunk-level language distribution across GT\n")
    section.append("```\n")
    for lang, n in sorted(chunk_lang_distribution.items(), key=lambda kv: -kv[1]):
        section.append(f"  {lang or '(none)':<8} {n}\n")
    section.append("```\n\n")
    section.append(f"- Items with >=1 non-Arabic chunk: {n_partial}\n")
    section.append(f"- Items where Whisper predicted no Arabic at all: {n_no_arabic}\n")
    section.append(f"- Items missing transcripts: {n_missing}\n\n")
    section.append("### 18.4 Results\n")
    section.append(
        f"- macro F1 @ 0.5: {row.get('macro_f1_at_0.5'):.4f} "
        f"(CI95 {row.get('macro_f1_ci95_lo'):.3f}-{row.get('macro_f1_ci95_hi'):.3f})\n"
    )
    section.append(
        f"- best macro F1 (snooped): {row.get('macro_f1_at_best'):.4f} @ thr={row.get('best_threshold'):.2f}\n"
    )
    section.append(
        f"- ROC-AUC: {row.get('roc_auc'):.4f} "
        f"(CI95 {row.get('roc_auc_ci95_lo'):.3f}-{row.get('roc_auc_ci95_hi'):.3f})\n\n"
    )
    auc = row.get("roc_auc") or 0.5
    if 0.45 <= auc <= 0.55:
        verdict = (
            "Near-random, as expected. Whisper's language probability does **not** carry usable "
            "Lebanese-vs-other signal: the corpus is dominated by Arabic chunks across all classes, "
            "and Whisper assigns high `prob_arabic` to both Lebanese and non-Lebanese items "
            "indiscriminately. This baseline confirms that Lebanese detection requires explicit "
            "dialect-specific modeling and is not a free byproduct of upstream ASR."
        )
    elif auc > 0.65:
        verdict = (
            f"Surprisingly above random (ROC-AUC {auc:.4f}). One possible mechanism: Whisper has "
            "lower language-ID confidence on non-Lebanese Arabic items (heavy Egyptian accent in "
            "FLEURS read-prompt; broadcast noise in ADI17), and `prob_arabic` therefore acts as a "
            "weak proxy for clean/clear Lebanese speech vs noisier non-Lebanese audio. This is a "
            "domain-confound signal masquerading as a content signal - similar in spirit to the V2 "
            "shortcut documented in §12.6.3."
        )
    else:
        verdict = (
            f"Below random (ROC-AUC {auc:.4f}). Whisper assigns systematically higher `prob_arabic` to "
            "non-Lebanese items than Lebanese items in this corpus - the opposite of what a dialect "
            "signal would predict. This is consistent with a noise/recording-condition confound where "
            "high `prob_arabic` indicates cleaner audio (FLEURS read-prompt, broadcast) rather than "
            "Lebanese content."
        )
    section.append(verdict + "\n")
    with open(FINDINGS_PATH, "a", encoding="utf-8") as f:
        f.write("".join(section))
    print(f"\nAppended Section 18 to {FINDINGS_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
