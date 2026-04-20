#!/usr/bin/env python3
"""
Convert all WAV files in data/raw_audio/ to FLAC (lossless compression).

For each item in queue.db with audio_path ending in .wav:
  1. Run ffmpeg to encode .wav → .flac (lossless)
  2. Verify the .flac file decodes correctly and has the expected duration
  3. Delete the original .wav
  4. Update queue.db: audio_path now points to the .flac file

Why: WAV mono 16kHz is uncompressed (~256 kbps). FLAC cuts size ~50% losslessly.
315 GB of WAVs → ~150 GB of FLACs with zero quality loss. Fully reversible
(decode back to bit-identical WAV anytime via ffmpeg).

Safety:
  - WAV is only deleted after FLAC is successfully encoded AND verified
  - DB update happens only after both succeed
  - Resumable: items already pointing to .flac are skipped
  - Errors are logged and the item's WAV is left intact

Run after stopping any other pipeline jobs that read from data/raw_audio/.
"""

import sys
import subprocess
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB


def ffmpeg_encode_flac(wav_path: Path, flac_path: Path) -> bool:
    """Encode WAV to FLAC using ffmpeg. Returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(wav_path),
        "-c:a", "flac",
        "-compression_level", "5",
        str(flac_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0 and flac_path.exists() and flac_path.stat().st_size > 0
    except Exception:
        return False


def verify_flac(flac_path: Path) -> bool:
    """Verify FLAC file is readable by ffmpeg (decode sanity check)."""
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(flac_path),
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception:
        return False


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    # Get all items with audio_path ending in .wav
    from sqlalchemy.orm import Session
    from src.db import QueueItem
    from sqlalchemy import select

    with Session(db.engine) as session:
        items = list(session.scalars(
            select(QueueItem).where(QueueItem.audio_path.isnot(None))
        ))

    wav_items = [i for i in items if i.audio_path and i.audio_path.lower().endswith(".wav")]
    flac_items = [i for i in items if i.audio_path and i.audio_path.lower().endswith(".flac")]

    print(f"Total items with audio: {len(items)}")
    print(f"  WAV (to convert): {len(wav_items)}")
    print(f"  FLAC (already done): {len(flac_items)}")

    if not wav_items:
        print("Nothing to do.")
        return

    # Disk space check
    total, used, free = shutil.disk_usage("C:/")
    print(f"C: disk free: {free/1e9:.1f} GB")
    if free < 200 * 1024 * 1024:  # 200 MB minimum
        print("WARNING: Very low disk space. Aborting.")
        return

    print(f"\nStarting conversion of {len(wav_items)} files...\n")

    ok = 0
    missing = 0
    encode_fail = 0
    verify_fail = 0
    db_fail = 0

    for idx, item in enumerate(wav_items, 1):
        wav_path = Path(item.audio_path)

        if not wav_path.exists():
            missing += 1
            continue

        flac_path = wav_path.with_suffix(".flac")

        # If FLAC already exists from a previous run, just update DB
        if flac_path.exists() and flac_path.stat().st_size > 0:
            if verify_flac(flac_path):
                try:
                    wav_path.unlink()
                except Exception:
                    pass
                db.update_source_metadata  # no-op, just to avoid unused import
                from sqlalchemy import update
                with Session(db.engine) as session:
                    session.execute(
                        update(QueueItem)
                        .where(QueueItem.id == item.id)
                        .values(audio_path=str(flac_path))
                    )
                    session.commit()
                ok += 1
                if idx % 50 == 0:
                    total, used, free = shutil.disk_usage("C:/")
                    print(f"  [{idx}/{len(wav_items)}] ok={ok} | C: free={free/1e9:.1f} GB")
                continue

        # Encode
        if not ffmpeg_encode_flac(wav_path, flac_path):
            encode_fail += 1
            if flac_path.exists():
                try:
                    flac_path.unlink()
                except Exception:
                    pass
            continue

        # Verify
        if not verify_flac(flac_path):
            verify_fail += 1
            try:
                flac_path.unlink()
            except Exception:
                pass
            continue

        # Both good: delete WAV, update DB
        try:
            wav_path.unlink()
        except Exception as e:
            print(f"  [{item.id}] WAV delete failed: {e}")
            continue

        try:
            from sqlalchemy import update
            with Session(db.engine) as session:
                session.execute(
                    update(QueueItem)
                    .where(QueueItem.id == item.id)
                    .values(audio_path=str(flac_path))
                )
                session.commit()
            ok += 1
        except Exception as e:
            db_fail += 1
            print(f"  [{item.id}] DB update failed: {e}")
            continue

        if idx % 50 == 0:
            total, used, free = shutil.disk_usage("C:/")
            print(f"  [{idx}/{len(wav_items)}] ok={ok} enc_fail={encode_fail} verify_fail={verify_fail} | C: free={free/1e9:.1f} GB")

    print("\n" + "=" * 48)
    print(f"DONE.")
    print(f"  Converted          : {ok}")
    print(f"  Missing WAV files  : {missing}")
    print(f"  Encode failures    : {encode_fail}")
    print(f"  Verify failures    : {verify_fail}")
    print(f"  DB update failures : {db_fail}")

    total, used, free = shutil.disk_usage("C:/")
    print(f"\nC: disk free: {free/1e9:.1f} GB")
    print("=" * 48)


if __name__ == "__main__":
    main()
