#!/usr/bin/env python3
"""
Targeted MSA download from the ADI17 train split.

MSA is absent from ADI17 dev and test splits (verified 2026-04-21 — see
FINDINGS.md 8.2.1). The full train split is 258 GB. Since Parquet files
are dialect-sorted, MSA should be concentrated in 1-3 specific files.

Strategy (single-pass):
  - Download train files one at a time with resume support
  - For each file, read the dialect column (no audio decode)
  - If MSA is present in the file, extract up to quota and move on
  - If no MSA, delete the file and try the next one
  - Stop when quota of 1,000 MSA items is hit

Persistent state:
  - data/adi17_processed_files.json — files that have been fully inspected
    (shared with script 07; MSA script appends train files here after scan)
  - data/adi17_msa_file_map.json — record of MSA count per scanned file
    (for future reference and thesis documentation)

Run: python scripts/08_download_adi17_msa.py
"""

import io
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
import requests

from src.cfg import Settings
from src.db import DB


REPO_ID = "ArabicSpeech/ADI17"
HF_BASE = "https://huggingface.co/datasets/ArabicSpeech/ADI17/resolve/main/"

TRAIN_FILES = [f"data/train-{i:05d}-of-00040.parquet" for i in range(40)]

TARGET_DIALECT = "MSA"
TARGET_COUNT = 1000

OUT_DIR = Path("data/raw_audio")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_CACHE = Path("data/adi17_parquet_cache")
PARQUET_CACHE.mkdir(parents=True, exist_ok=True)

FILE_MAP_PATH = Path("data/adi17_msa_file_map.json")
PROCESSED_MANIFEST = Path("data/adi17_processed_files.json")


def load_processed() -> set[str]:
    if PROCESSED_MANIFEST.exists():
        return set(json.loads(PROCESSED_MANIFEST.read_text(encoding="utf-8")))
    return set()


def mark_processed(fname: str) -> None:
    done = load_processed()
    done.add(fname)
    PROCESSED_MANIFEST.write_text(json.dumps(sorted(done), indent=2), encoding="utf-8")


def load_msa_map() -> dict[str, int]:
    if FILE_MAP_PATH.exists():
        return json.loads(FILE_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def save_msa_map(m: dict[str, int]) -> None:
    FILE_MAP_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")


def download_parquet_with_resume(remote_path: str, local_path: Path, max_retries: int = 20) -> bool:
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
            time.sleep(min(2 ** attempt, 30))
    print(f"    FAILED after {max_retries} attempts", flush=True)
    return False


def encode_wav_bytes_to_mp3(audio_bytes: bytes, out_path: Path) -> bool:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        "-ac", "1", "-ar", "16000",
        "-af", "loudnorm",
        "-c:a", "libmp3lame", "-b:a", "96k",
        str(out_path),
    ]
    try:
        r = subprocess.run(cmd, input=audio_bytes, capture_output=True, timeout=60)
        return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def get_duration(path: Path) -> float:
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


def count_msa_in_db(db: DB) -> int:
    from sqlalchemy.orm import Session
    from src.db import QueueItem
    from sqlalchemy import select
    with Session(db.engine) as session:
        items = session.scalars(
            select(QueueItem).where(QueueItem.platform == "adi17")
        ).all()
        return sum(1 for i in items if (i.source_metadata or {}).get("adi17_dialect") == "MSA")


def process_file_for_msa(local_path: Path, db: DB, remaining: int) -> tuple[int, int]:
    """Read dialect column of downloaded file. Extract up to `remaining` MSA items.
    Returns (msa_in_file, added)."""
    from sqlalchemy.orm import Session
    from src.db import QueueItem
    from sqlalchemy import update

    print(f"  Reading dialect column from {local_path.name}...", flush=True)
    dialect_col = pq.read_table(str(local_path), columns=["dialect"]).column("dialect").to_pylist()
    total_rows = len(dialect_col)
    msa_indices = [i for i, d in enumerate(dialect_col) if d == TARGET_DIALECT]
    msa_count_in_file = len(msa_indices)
    print(f"  File has {total_rows} rows; {msa_count_in_file} are MSA", flush=True)

    if msa_count_in_file == 0 or remaining <= 0:
        return msa_count_in_file, 0

    # Now read full audio only for the MSA rows
    print(f"  Extracting up to {remaining} MSA items (full audio read)...", flush=True)
    table = pq.read_table(str(local_path))
    added = 0
    for i in msa_indices:
        if added >= remaining:
            break
        row = table.slice(i, 1).to_pylist()[0]
        item_id = row["id"]
        audio = row["audio"]
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not audio_bytes:
            continue

        out_filename = f"adi17_MSA_{item_id}.mp3"
        out_path = OUT_DIR / out_filename
        if not out_path.exists():
            if not encode_wav_bytes_to_mp3(audio_bytes, out_path):
                continue

        duration = get_duration(out_path)
        if duration <= 0:
            try:
                out_path.unlink()
            except Exception:
                pass
            continue

        url = f"https://huggingface.co/datasets/{REPO_ID}/{item_id}"
        metadata = {
            "source": "adi17",
            "adi17_id": item_id,
            "adi17_dialect": "MSA",
            "contrastive_role": "msa",
        }
        if db.add_to_queue(url=url, platform="adi17", source_metadata=metadata):
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
        added += 1
        if added % 100 == 0:
            print(f"    extracted {added}/{remaining}...", flush=True)

    return msa_count_in_file, added


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    current = count_msa_in_db(db)
    print(f"Existing MSA items in DB: {current}", flush=True)
    print(f"Target: {TARGET_COUNT}", flush=True)
    if current >= TARGET_COUNT:
        print("MSA quota met.", flush=True)
        return

    processed = load_processed()
    msa_map = load_msa_map()
    print(f"Train files already inspected: {sum(1 for f in TRAIN_FILES if f in processed)}/{len(TRAIN_FILES)}", flush=True)

    for fname in TRAIN_FILES:
        if fname in processed:
            ct = msa_map.get(fname, 0)
            if ct > 0:
                print(f"\nSKIP (already inspected, {ct} MSA): {fname}", flush=True)
            continue

        current = count_msa_in_db(db)
        if current >= TARGET_COUNT:
            print(f"\nQuota reached: {current} MSA items.", flush=True)
            break

        print(f"\n{'='*60}", flush=True)
        print(f"Trying {fname}...", flush=True)

        local_name = fname.replace("/", "_")
        local_path = PARQUET_CACHE / local_name

        if not download_parquet_with_resume(fname, local_path):
            print(f"  Download failed; moving on.", flush=True)
            continue

        msa_in_file, added = process_file_for_msa(local_path, db, TARGET_COUNT - current)
        msa_map[fname] = msa_in_file
        save_msa_map(msa_map)
        mark_processed(fname)
        print(f"  Added {added} MSA items. MSA in DB now: {count_msa_in_db(db)}", flush=True)

        try:
            local_path.unlink()
            print(f"  Deleted {local_path.name}", flush=True)
        except Exception as e:
            print(f"  Failed to delete: {e}", flush=True)

    final = count_msa_in_db(db)
    print(f"\n{'='*60}", flush=True)
    print(f"DONE. Final MSA count: {final}/{TARGET_COUNT}", flush=True)


if __name__ == "__main__":
    main()
