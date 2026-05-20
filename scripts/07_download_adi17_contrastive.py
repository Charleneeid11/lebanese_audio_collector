#!/usr/bin/env python3
"""
Download non-Lebanese contrastive audio from ADI17 (QCRI Arabic Dialect Identification).

Dataset: https://huggingface.co/datasets/ArabicSpeech/ADI17
Citation: Ali et al. "The MGB-3 Arabic Dialect Speech Recognition Challenge" / ADI17 benchmark

Strategy:
  - Download dev+test Parquet files one at a time (each ~0.2-0.7 GB)
  - For each file, scan dialect labels
  - Extract audio ONLY for target dialects (EGY, MSA, Gulf, LEB)
  - Save audio as MP3 (96 kbps) in data/raw_audio/ with ADI17 prefix
  - Insert into queue.db with dialect label in source_metadata
  - Delete the Parquet file after extraction to save disk space

Target dialects:
  - EGY: Egyptian (non-Lebanese, target negative class)
  - MSA: Modern Standard Arabic (non-Lebanese, target negative class)
  - Gulf variants: KSA, KWT, UAE, QAT, OMA (non-Lebanese, target negative class)
  - LEB: Lebanese (additional positive training data, pre-labeled, research-grade)

Target size: ~1000 items per dialect (stop early if reached)

Run: python scripts/07_download_adi17_contrastive.py
"""

import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
import requests
import time

from src.cfg import Settings
from src.db import DB


HF_BASE = "https://huggingface.co/datasets/ArabicSpeech/ADI17/resolve/main/"
PARQUET_CACHE = Path("data/adi17_parquet_cache")
PARQUET_CACHE.mkdir(parents=True, exist_ok=True)


def download_parquet_with_resume(remote_path: str, local_path: Path, max_retries: int = 20) -> bool:
    """Download a Parquet file from HF with resume support and progress logging."""
    url = HF_BASE + remote_path
    for attempt in range(max_retries):
        try:
            resume_bytes = local_path.stat().st_size if local_path.exists() else 0
            head = requests.head(url, allow_redirects=True, timeout=30)
            total = int(head.headers.get("content-length", 0))
            if resume_bytes >= total and total > 0:
                return True

            headers = {"Range": f"bytes={resume_bytes}-"} if resume_bytes > 0 else {}
            print(f"    attempt {attempt+1}: resuming from {resume_bytes/1e6:.0f}/{total/1e6:.0f} MB", flush=True)

            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                mode = "ab" if resume_bytes > 0 else "wb"
                got = resume_bytes
                last_report = time.time()
                with open(local_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)
                            if time.time() - last_report >= 30:
                                pct = got / total * 100 if total else 0
                                print(f"    progress: {got/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)", flush=True)
                                last_report = time.time()

            if local_path.stat().st_size >= total and total > 0:
                print(f"    done: {local_path.stat().st_size/1e6:.0f} MB", flush=True)
                return True

        except (requests.exceptions.RequestException, OSError) as e:
            print(f"    attempt {attempt+1} error: {type(e).__name__}: {e}", flush=True)
            time.sleep(min(2 ** attempt, 30))  # exponential backoff, capped at 30s

    print(f"    FAILED after {max_retries} attempts", flush=True)
    return False


REPO_ID = "ArabicSpeech/ADI17"
REPO_TYPE = "dataset"

DEV_FILES = [f"data/dev-{i:05d}-of-00004.parquet" for i in range(4)]
TEST_FILES = [f"data/test-{i:05d}-of-00005.parquet" for i in range(5)]

# Map ADI17 dialect codes to our pipeline categories
# For the contrastive non-Lebanese dataset we want: EGY, MSA, and Gulf variants
TARGET_DIALECTS = {
    "EGY": "egyptian",
    "MSA": "msa",
    "KSA": "gulf",
    "KUW": "gulf",  # ADI17 uses "KUW" not "KWT" for Kuwaiti
    "UAE": "gulf",
    "QAT": "gulf",
    "OMA": "gulf",
    "LEB": "lebanese",  # Additional Lebanese positives
}

TARGET_PER_DIALECT = 1000  # stop early once we have this many per dialect

OUT_DIR = Path("data/raw_audio")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_MANIFEST = Path("data/adi17_processed_files.json")


def load_processed_files() -> set[str]:
    if PROCESSED_MANIFEST.exists():
        return set(json.loads(PROCESSED_MANIFEST.read_text(encoding="utf-8")))
    return set()


def mark_file_processed(fname: str) -> None:
    done = load_processed_files()
    done.add(fname)
    PROCESSED_MANIFEST.write_text(json.dumps(sorted(done), indent=2), encoding="utf-8")


def encode_wav_bytes_to_mp3(audio_bytes: bytes, out_path: Path) -> bool:
    """Convert in-memory WAV bytes to MP3 on disk via ffmpeg stdin."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "wav",
        "-i", "pipe:0",
        "-ac", "1", "-ar", "16000",
        "-af", "loudnorm",
        "-c:a", "libmp3lame", "-b:a", "96k",
        str(out_path),
    ]
    try:
        r = subprocess.run(cmd, input=audio_bytes, capture_output=True, timeout=60)
        return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        print(f"    ffmpeg error: {e}")
        return False


def get_duration_seconds(path: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ], stderr=subprocess.DEVNULL, timeout=30)
        return float(out.strip())
    except Exception:
        return 0.0


def count_existing_per_dialect(db: DB) -> Counter:
    """Count ADI17 items already in DB per dialect."""
    from sqlalchemy.orm import Session
    from src.db import QueueItem
    from sqlalchemy import select

    counts = Counter()
    with Session(db.engine) as session:
        items = session.scalars(
            select(QueueItem).where(QueueItem.platform == "adi17")
        ).all()
        for item in items:
            meta = item.source_metadata or {}
            d = meta.get("adi17_dialect")
            if d:
                counts[d] += 1
    return counts


def process_parquet_file(parquet_path: Path, db: DB, counts: Counter) -> dict:
    """Extract audio for target dialects from a Parquet file. Return stats."""
    print(f"\n  Reading {parquet_path.name}...")
    table = pq.read_table(str(parquet_path))
    rows = table.to_pylist()
    print(f"  Rows: {len(rows)}")

    file_dialects = Counter(r["dialect"] for r in rows)
    print(f"  Dialects in file: {dict(file_dialects)}")

    stats = {"added": 0, "skipped_not_target": 0, "skipped_quota_full": 0, "skipped_failed": 0}

    for row in rows:
        dialect = row["dialect"]

        if dialect not in TARGET_DIALECTS:
            stats["skipped_not_target"] += 1
            continue

        if counts[dialect] >= TARGET_PER_DIALECT:
            stats["skipped_quota_full"] += 1
            continue

        item_id = row["id"]
        audio = row["audio"]
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not audio_bytes:
            stats["skipped_failed"] += 1
            continue

        out_filename = f"adi17_{dialect}_{item_id}.mp3"
        out_path = OUT_DIR / out_filename

        if not out_path.exists():
            if not encode_wav_bytes_to_mp3(audio_bytes, out_path):
                stats["skipped_failed"] += 1
                continue

        duration = get_duration_seconds(out_path)
        if duration <= 0:
            stats["skipped_failed"] += 1
            try:
                out_path.unlink()
            except Exception:
                pass
            continue

        # Insert into DB
        url = f"https://huggingface.co/datasets/{REPO_ID}/{item_id}"
        metadata = {
            "source": "adi17",
            "adi17_id": item_id,
            "adi17_dialect": dialect,
            "contrastive_role": TARGET_DIALECTS[dialect],
        }

        if db.add_to_queue(url=url, platform="adi17", source_metadata=metadata):
            # Newly added — set audio_path and status
            from sqlalchemy.orm import Session
            from src.db import QueueItem
            from sqlalchemy import update

            with Session(db.engine) as session:
                session.execute(
                    update(QueueItem)
                    .where(QueueItem.url == url)
                    .values(
                        audio_path=str(out_path),
                        duration_seconds=int(duration),
                        status="DOWNLOADED",
                    )
                )
                session.commit()

        counts[dialect] += 1
        stats["added"] += 1

    return stats


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    counts = count_existing_per_dialect(db)
    print(f"Existing ADI17 items in DB per dialect: {dict(counts)}")
    print(f"Target per dialect: {TARGET_PER_DIALECT}")
    print(f"Target dialect codes: {list(TARGET_DIALECTS.keys())}\n")

    all_files = DEV_FILES + TEST_FILES
    processed = load_processed_files()
    print(f"Processing {len(all_files)} Parquet files (dev + test splits)", flush=True)
    print(f"Already processed: {len(processed)} files", flush=True)

    for fname in all_files:
        if fname in processed:
            print(f"\nSKIP (already processed): {fname}", flush=True)
            continue

        # Stop if all quotas filled
        all_full = all(counts[d] >= TARGET_PER_DIALECT for d in TARGET_DIALECTS)
        if all_full:
            print("\nAll target dialect quotas filled. Stopping.", flush=True)
            break

        print(f"\n{'='*60}", flush=True)
        print(f"Downloading {fname}...", flush=True)

        local_name = fname.replace("/", "_")
        local_path = PARQUET_CACHE / local_name

        if not download_parquet_with_resume(fname, local_path):
            print(f"  Download failed after retries; moving on.", flush=True)
            continue

        stats = process_parquet_file(local_path, db, counts)
        print(f"  Stats: {stats}", flush=True)
        print(f"  Running totals: {dict(counts)}", flush=True)

        mark_file_processed(fname)

        try:
            local_path.unlink()
            print(f"  Deleted {local_path.name}", flush=True)
        except Exception as e:
            print(f"  Failed to delete Parquet: {e}", flush=True)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"Final counts per dialect: {dict(counts)}")


if __name__ == "__main__":
    main()
