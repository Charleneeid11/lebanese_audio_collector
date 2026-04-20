#!/usr/bin/env python3
"""
Assign WEAK_NEGATIVE labels to SCREENED items with no Lebanese dialect markers.

This is a lexicon-based labeling pass that complements 03b (metadata-based labeling).
It processes SCREENED items regardless of whether they have source_metadata, making
it the primary path for podcast_rss items which were discovered without feed_url metadata.

An item is labeled WEAK_NEGATIVE if ALL of the following hold:
  - Whisper detected Arabic (language == "ar") with probability >= MIN_LANG_PROB
  - raw_score < 0 (non-Lebanese signals outweigh Lebanese signals)
    raw_score = lb*1.8 - msa*0.6 - egy*1.0 - gulf*1.0 - sy*0.5
    This means Egyptian/Gulf/MSA vocabulary is dominating the transcript.
  - At least one match in MSA, Egyptian, Gulf, or Syrian lexicons
    (confirms the audio is Arabic, not empty/ambiguous audio)

Note on lb == 0: NOT used as a criterion because LEBANESE_WORDS contains many
pan-Arabic words (يعني, في, بس, مش, تمام, مرحبا) that appear in all dialects.
raw_score < 0 is the correct signal — it shows non-LB signals outweigh LB ones.

These "lexically-verified negatives" serve as the negative class for training
the dialect classifier in 05_train_dialect_model.py.

Run after 03_transcribe_screening.py and before 05_train_dialect_model.py.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB
from src.dialect.scoring import lexicon_score


TRANSCRIPT_DIR = Path("data/transcripts")
MIN_LANG_PROB = 0.70  # minimum Arabic language_probability from Whisper


def is_arabic(samples: list[dict]) -> tuple[bool, float]:
    """Return (is_arabic, avg_probability) based on Whisper language detection."""
    ar_probs = [
        s.get("language_probability", 0.0)
        for s in samples
        if s.get("language") == "ar"
    ]
    if not ar_probs:
        return False, 0.0
    avg = sum(ar_probs) / len(ar_probs)
    return avg >= MIN_LANG_PROB, round(avg, 3)


def get_text(samples: list[dict]) -> str:
    return " ".join(s.get("text", "") for s in samples)


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    items = db.fetch_queue(status="SCREENED", limit=None)
    if not items:
        print("No SCREENED items to process.")
        return

    total = len(items)
    print(f"Processing {total} SCREENED items for lexical negative labeling...")
    print(f"Threshold: lb==0, strong_lb_hits==0, language_prob>={MIN_LANG_PROB}\n")

    neg = 0
    skip_no_transcript = 0
    skip_not_arabic = 0
    skip_has_lb = 0
    skip_no_dialect_signal = 0

    for item in items:
        transcript_path = TRANSCRIPT_DIR / f"clip_{item.id}_screening.json"

        if not transcript_path.exists():
            skip_no_transcript += 1
            continue

        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        samples = data.get("screening_samples", [])

        arabic, lang_prob = is_arabic(samples)
        if not arabic:
            skip_not_arabic += 1
            continue

        text = get_text(samples)
        _, lex = lexicon_score(text)

        # raw_score < 0 means non-Lebanese signals outweigh Lebanese signals.
        # This is the correct criterion — lb==0 is not used because LEBANESE_WORDS
        # contains pan-Arabic words that appear in nearly every Arabic transcript.
        if lex["raw_score"] >= 0:
            skip_has_lb += 1
            continue

        # Must have at least some non-Lebanese Arabic vocabulary to confirm
        # this is actual Arabic content and not empty/unclear audio.
        other_dialect = lex["msa"] + lex["egy"] + lex["gulf"] + lex["sy"]
        if other_dialect == 0:
            skip_no_dialect_signal += 1
            continue

        db.update_status(item.id, "WEAK_NEGATIVE")
        neg += 1

    print("=" * 48)
    print(f"WEAK_NEGATIVE assigned   : {neg}")
    print(f"Skipped — no transcript  : {skip_no_transcript}")
    print(f"Skipped — not Arabic     : {skip_not_arabic}")
    print(f"Skipped — raw_score >= 0 : {skip_has_lb}")
    print(f"Skipped — no dial. signal: {skip_no_dialect_signal}")
    print(f"Total processed          : {total}")
    print("=" * 48)


if __name__ == "__main__":
    main()
