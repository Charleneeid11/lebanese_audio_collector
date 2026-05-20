"""
Colab notebook script: extract 1,000 MSA audio items from ADI17 train split.

Run this in Google Colab (https://colab.research.google.com — free tier is fine).
Paste this whole file into a single cell and run.

Strategy:
  - hf_transfer (Rust parallel chunked download, 10-50x faster than requests)
  - Iterate train files one at a time
  - Read just the dialect column → if zero MSA, delete file
  - Otherwise extract up to remaining quota (audio + metadata)
  - Encode audio to MP3 (16kHz mono, 96kbps, loudnorm)
  - Bundle results into /content/adi17_msa.zip

After it finishes, download adi17_msa.zip to your local machine, then run
  scripts/10_import_colab_msa.py
to load into the queue DB.
"""

# ============================================================
# Cell 1: setup
# ============================================================
# !pip install -q hf_transfer pyarrow huggingface_hub
# !apt-get -qq install -y ffmpeg

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import io
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "ArabicSpeech/ADI17"
TARGET_DIALECT = "MSA"
TARGET_COUNT = 1000
TRAIN_FILES = [f"data/train-{i:05d}-of-00040.parquet" for i in range(40)]

WORK = Path("/content/adi17_work")
WORK.mkdir(exist_ok=True)
AUDIO_OUT = WORK / "audio"
AUDIO_OUT.mkdir(exist_ok=True)
META_PATH = WORK / "msa_metadata.jsonl"
SUMMARY_PATH = WORK / "summary.json"
ZIP_PATH = Path("/content/adi17_msa.zip")


def encode_wav_to_mp3(audio_bytes: bytes, out_path: Path) -> bool:
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


# ============================================================
# Cell 2: download and process
# ============================================================
total_msa_extracted = 0
file_summaries = []

# Append-mode metadata file so we never lose progress
meta_f = open(META_PATH, "a", encoding="utf-8")

for fname in TRAIN_FILES:
    if total_msa_extracted >= TARGET_COUNT:
        print(f"Quota reached: {total_msa_extracted} MSA items.", flush=True)
        break

    print(f"\n{'='*60}")
    print(f"Downloading {fname} (hf_transfer)...", flush=True)
    t0 = time.time()
    try:
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=str(WORK),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"  download failed: {e}", flush=True)
        file_summaries.append({"file": fname, "error": str(e)})
        continue
    dl_seconds = time.time() - t0
    size_gb = Path(local).stat().st_size / 1e9
    print(f"  downloaded {size_gb:.2f} GB in {dl_seconds:.0f}s ({size_gb*1000/max(dl_seconds,1):.0f} MB/s)", flush=True)

    print(f"  reading dialect column...", flush=True)
    dialect_col = pq.read_table(local, columns=["dialect"]).column("dialect").to_pylist()
    counts = Counter(dialect_col)
    msa_indices = [i for i, d in enumerate(dialect_col) if d == TARGET_DIALECT]
    print(f"  rows={len(dialect_col)}, MSA={len(msa_indices)}, top dialects: {counts.most_common(3)}", flush=True)

    extracted_this_file = 0
    if msa_indices:
        remaining = TARGET_COUNT - total_msa_extracted
        print(f"  extracting up to {remaining} MSA items...", flush=True)
        table = pq.read_table(local)
        for i in msa_indices:
            if extracted_this_file >= remaining:
                break
            row = table.slice(i, 1).to_pylist()[0]
            item_id = row["id"]
            audio = row["audio"]
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if not audio_bytes:
                continue
            out_filename = f"adi17_MSA_{item_id}.mp3"
            out_path = AUDIO_OUT / out_filename
            if not encode_wav_to_mp3(audio_bytes, out_path):
                continue
            duration = get_duration(out_path)
            if duration <= 0:
                try:
                    out_path.unlink()
                except Exception:
                    pass
                continue
            meta_f.write(json.dumps({
                "id": item_id,
                "dialect": "MSA",
                "audio_filename": out_filename,
                "duration_seconds": duration,
                "source_file": fname,
            }) + "\n")
            meta_f.flush()
            extracted_this_file += 1
            total_msa_extracted += 1
            if extracted_this_file % 100 == 0:
                print(f"    extracted {extracted_this_file}...", flush=True)

    file_summaries.append({
        "file": fname,
        "total_rows": len(dialect_col),
        "msa_rows": len(msa_indices),
        "msa_extracted": extracted_this_file,
        "dialect_counts": dict(counts),
        "download_seconds": round(dl_seconds, 1),
    })
    print(f"  extracted {extracted_this_file}; total so far {total_msa_extracted}/{TARGET_COUNT}", flush=True)

    try:
        Path(local).unlink()
    except Exception as e:
        print(f"  delete warning: {e}", flush=True)

meta_f.close()

# ============================================================
# Cell 3: package
# ============================================================
SUMMARY_PATH.write_text(json.dumps({
    "total_msa_extracted": total_msa_extracted,
    "files_scanned": len(file_summaries),
    "per_file": file_summaries,
}, indent=2), encoding="utf-8")

print(f"\nPackaging into {ZIP_PATH}...", flush=True)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as z:
    z.write(META_PATH, arcname="msa_metadata.jsonl")
    z.write(SUMMARY_PATH, arcname="summary.json")
    for mp3 in AUDIO_OUT.glob("*.mp3"):
        z.write(mp3, arcname=f"audio/{mp3.name}")

mb = ZIP_PATH.stat().st_size / 1e6
print(f"Done. {ZIP_PATH} = {mb:.0f} MB. Total MSA: {total_msa_extracted}.")
print(f"Download it from Colab's file panel (left sidebar) to your local machine.")
