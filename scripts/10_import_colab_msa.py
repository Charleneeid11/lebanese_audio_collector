#!/usr/bin/env python3
"""
Import MSA audio extracted by a Colab script into the queue DB.

Generic — handles either:
  - data/adi17_msa.zip   (from scripts/colab_extract_msa.py — ADI17 train split)
  - data/fleurs_msa.zip  (from scripts/colab_extract_msa_fleurs.py — Google FLEURS ar_eg)

Each zip contains:
  - msa_metadata.jsonl  (one JSON item per line; must include id, dialect, audio_filename,
                         duration_seconds, plus source-specific fields)
  - summary.json
  - audio/<source>_MSA_<id>.mp3

The metadata's `source` field (or the zip filename) selects the queue platform:
  - source="adi17" → platform="adi17", url uses HF dataset URL
  - source="fleurs" → platform="fleurs", url uses HF dataset URL with config

Usage:
  python scripts/10_import_colab_msa.py                 # auto-detect first existing zip
  python scripts/10_import_colab_msa.py data/foo.zip    # explicit path
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import update
from sqlalchemy.orm import Session

from src.cfg import Settings
from src.db import DB, QueueItem


CANDIDATE_ZIPS = [
    Path("data/fleurs_msa.zip"),
    Path("data/adi17_msa.zip"),
]
RAW_AUDIO = Path("data/raw_audio")
PROCESSED_MANIFEST = Path("data/adi17_processed_files.json")
MSA_FILE_MAP = Path("data/adi17_msa_file_map.json")


def url_and_metadata_for(item: dict) -> tuple[str, str, dict]:
    """Build (url, platform, metadata) for an MSA metadata entry."""
    source = item.get("source") or "adi17"
    item_id = item["id"]
    if source == "common_voice":
        config = item.get("config", "ar")
        url = f"https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/{config}/{item_id}"
        return url, "common_voice", {
            "source": "common_voice",
            "cv_id": item_id,
            "cv_config": config,
            "cv_split": item.get("split"),
            "cv_client_id": item.get("client_id"),
            "cv_gender": item.get("gender"),
            "cv_age": item.get("age"),
            "cv_accent": item.get("accent"),
            "cv_up_votes": item.get("up_votes"),
            "cv_down_votes": item.get("down_votes"),
            "transcription": item.get("transcription"),
            "adi17_dialect": "MSA",        # keep this field name for downstream consistency
            "contrastive_role": "msa",
            "imported_via": "colab",
        }
    if source == "fleurs":
        config = item.get("fleurs_config") or item.get("config", "ar_eg")
        url = f"https://huggingface.co/datasets/google/fleurs/{config}/{item_id}"
        return url, "fleurs", {
            "source": "fleurs",
            "fleurs_id": item_id,
            "fleurs_config": config,
            "fleurs_split": item.get("fleurs_split") or item.get("split"),
            "fleurs_gender": item.get("gender"),
            "transcription": item.get("transcription"),
            "adi17_dialect": "MSA",
            "contrastive_role": "msa",
            "imported_via": "colab",
        }
    # default: ADI17
    url = f"https://huggingface.co/datasets/ArabicSpeech/ADI17/{item_id}"
    return url, "adi17", {
        "source": "adi17",
        "adi17_id": item_id,
        "adi17_dialect": "MSA",
        "contrastive_role": "msa",
        "imported_via": "colab",
    }


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        zip_path = Path(argv[1])
    else:
        zip_path = next((p for p in CANDIDATE_ZIPS if p.exists()), None)

    if zip_path is None or not zip_path.exists():
        print(f"ERROR: no zip found. Looked for: {[str(p) for p in CANDIDATE_ZIPS]}")
        print("Download from Colab first, or pass an explicit path.")
        return 1

    extract_dir = Path(f"data/{zip_path.stem}_unzip")
    print(f"Extracting {zip_path} -> {extract_dir}/...")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)

    meta_path = extract_dir / "msa_metadata.jsonl"
    summary_path = extract_dir / "summary.json"
    audio_dir = extract_dir / "audio"

    if not meta_path.exists() or not audio_dir.exists():
        print(f"ERROR: zip did not contain expected msa_metadata.jsonl + audio/")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    print(f"Colab summary: {summary}")

    settings = Settings.load()
    db = DB(settings.db_url)

    RAW_AUDIO.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8")]
    print(f"Importing {len(items)} MSA items...")

    added = skipped = failed = 0
    for it in items:
        src = audio_dir / it["audio_filename"]
        dst = RAW_AUDIO / it["audio_filename"]
        if not src.exists():
            failed += 1
            continue
        if not dst.exists():
            shutil.copy2(src, dst)

        url, platform, metadata = url_and_metadata_for(it)
        if db.add_to_queue(url=url, platform=platform, source_metadata=metadata):
            with Session(db.engine) as session:
                session.execute(
                    update(QueueItem)
                    .where(QueueItem.url == url)
                    .values(
                        audio_path=str(dst),
                        duration_seconds=int(it.get("duration_seconds", 0)),
                        status="DOWNLOADED",
                    )
                )
                session.commit()
            added += 1
        else:
            skipped += 1

    print(f"Added: {added}, already-in-db: {skipped}, failed: {failed}")

    # ADI17 only: update train-file scan manifests
    if "per_file" in summary:
        processed = set(json.loads(PROCESSED_MANIFEST.read_text(encoding="utf-8"))) if PROCESSED_MANIFEST.exists() else set()
        msa_map = json.loads(MSA_FILE_MAP.read_text(encoding="utf-8")) if MSA_FILE_MAP.exists() else {}
        for entry in summary["per_file"]:
            if "msa_rows" in entry:
                processed.add(entry["file"])
                msa_map[entry["file"]] = entry["msa_rows"]
        PROCESSED_MANIFEST.write_text(json.dumps(sorted(processed), indent=2), encoding="utf-8")
        MSA_FILE_MAP.write_text(json.dumps(msa_map, indent=2), encoding="utf-8")
        print(f"Updated manifests: {len(processed)} files marked processed.")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
