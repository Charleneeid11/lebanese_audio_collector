#!/usr/bin/env python3
# Takes DOWNLOADED audio files, extracts 3×20-second chunks,
# transcribes them using fast Whisper, saves screening transcripts,
# and updates each item’s status to SCREENED.

import sys
import time
from pathlib import Path
import json

# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB
from src.utils.audio import extract_random_chunks
from src.asr.whisper_engine import transcribe_screening


BATCH_SIZE = 5
SLEEP_SECONDS = 1


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    transcripts_dir = Path(settings.transcription.transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    while True:
        items = db.fetch_queue(status="DOWNLOADED", limit=BATCH_SIZE)

        if not items:
            print("No more DOWNLOADED items to screen-transcribe.")
            break

        for item in items:
            if not item.audio_path:
                db.update_status(item.id, "ERROR_SCREEN", "Missing audio_path")
                continue

            audio_path = Path(item.audio_path)
            if not audio_path.exists():
                db.update_status(
                    item.id,
                    "ERROR_SCREEN",
                    f"Audio not found: {audio_path}",
                )
                continue

            print(f"[{item.id}] Extracting chunks from {audio_path} ...")

            try:
                # Step 1: extract 3 × 20s chunks
                chunk_paths = extract_random_chunks(str(audio_path))

                all_results = []
                for chunk in chunk_paths:
                    print(f"   -> Transcribing chunk {chunk} ...")
                    res = transcribe_screening(chunk, settings)
                    all_results.append({
                        "chunk_path": chunk,
                        **res,
                    })

                # Step 2: save JSON
                out_path = transcripts_dir / f"clip_{item.id}_screening.json"

                payload = {
                    "id": item.id,
                    "url": item.url,
                    "platform": item.platform,
                    "screening_samples": all_results,
                }

                out_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                db.update_status(item.id, "SCREENED")
                print(f"  -> Saved screening transcript to {out_path}")

            except Exception as e:
                db.update_status(item.id, "ERROR_SCREEN", str(e))
                print(f"  -> ERROR_SCREEN: {e}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()




# This script performs a lightweight screening transcription on downloaded audio files. It repeatedly fetches items from the database with status="DOWNLOADED" (in batches), 
# verifies that their audio_path exists, and then extracts three random 20-second chunks from each file using extract_random_chunks. Each chunk is transcribed using a fast
# Whisper-based screening function (transcribe_screening), and the results are collected. The script saves a structured JSON file per item (including item ID, URL, platform,
# and all chunk-level transcription results) into the configured transcripts directory. If successful, the item’s status is updated to SCREENED; if any error 
# occurs (missing file, transcription failure, etc.), it is marked as ERROR_SCREEN. This stage provides a quick linguistic preview of the audio before full transcription 
# or further filtering.