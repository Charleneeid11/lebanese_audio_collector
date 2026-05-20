"""
Colab notebook script: extract up to 1,000 MSA audio items.

Run this in Google Colab (https://colab.research.google.com — free tier is fine).
Paste this whole file into a single cell, uncomment the !pip / !apt-get lines, and run.

Sources tried in order (first one that works wins):
  1. google/fleurs config ar_eg (FLEURS — Conneau et al. 2022, multi-speaker MSA, CC-BY-SA)
  2. mozilla-foundation/common_voice_17_0 config ar (Common Voice — multi-speaker, CC0)

Both are multi-speaker (avoids the single-speaker confound that disqualified Halabi 2016)
and citable. Output filenames use the actual source: fleurs_MSA_*.mp3 or cv_MSA_*.mp3.

Audio pipeline matches ADI17:
  - mono, 16 kHz, loudnorm, 96 kbps MP3

Output: /content/fleurs_msa.zip with msa_metadata.jsonl + summary.json + audio/*.mp3
(zip is named fleurs_msa.zip regardless of which source actually worked, so the
local import script auto-detects it.)
"""

# ============================================================
# Cell 1: setup
# ============================================================
# !pip install -q -U datasets soundfile librosa huggingface_hub
# !apt-get -qq install -y ffmpeg

import io
import json
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path

import soundfile as sf
from datasets import load_dataset, Audio


TARGET_COUNT = 1000

WORK = Path("/content/fleurs_work")
WORK.mkdir(exist_ok=True)
AUDIO_OUT = WORK / "audio"
AUDIO_OUT.mkdir(exist_ok=True)
META_PATH = WORK / "msa_metadata.jsonl"
SUMMARY_PATH = WORK / "summary.json"
ZIP_PATH = Path("/content/fleurs_msa.zip")


def array_to_wav_bytes(audio_array, sampling_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_array, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


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
# Cell 2: try sources, extract
# ============================================================
SOURCES = [
    {
        "name": "fleurs",
        "filename_prefix": "fleurs_MSA",
        "loaders": [
            lambda split: load_dataset("google/fleurs", "ar_eg", split=split, trust_remote_code=True),
            lambda split: load_dataset("google/fleurs", "ar_eg", split=split),
            lambda split: load_dataset("google/fleurs", name="ar_eg", split=split, trust_remote_code=True),
        ],
        "splits": ("train", "validation", "test"),
        "audio_field": "audio",
        "id_field": "id",
        "transcript_field": "transcription",
        "gender_field": "gender",
    },
    {
        "name": "common_voice",
        "filename_prefix": "cv_MSA",
        "loaders": [
            lambda split: load_dataset("mozilla-foundation/common_voice_17_0", "ar", split=split, trust_remote_code=True),
            lambda split: load_dataset("mozilla-foundation/common_voice_17_0", "ar", split=split),
        ],
        "splits": ("train", "validation", "test", "other"),
        "audio_field": "audio",
        "id_field": "client_id",        # CV doesn't have a single int id; use client_id as a proxy
        "transcript_field": "sentence",
        "gender_field": "gender",
    },
]


def try_load_split(loaders, split_name):
    last_err = None
    for loader in loaders:
        try:
            ds = loader(split_name)
            return ds, None
        except Exception as e:
            last_err = e
            continue
    return None, last_err


def run_source(src, total_extracted, meta_f, per_split):
    """Process one source until quota hit or out of data. Return new total_extracted."""
    print(f"\n{'='*60}")
    print(f"Source: {src['name']}", flush=True)

    name = src["name"]
    prefix = src["filename_prefix"]

    for split_name in src["splits"]:
        if total_extracted >= TARGET_COUNT:
            break
        print(f"\n  loading split={split_name}...", flush=True)
        ds, err = try_load_split(src["loaders"], split_name)
        if ds is None:
            print(f"    skipped ({type(err).__name__}: {err})", flush=True)
            continue
        try:
            n = len(ds)
            print(f"    loaded: {n} rows", flush=True)
        except Exception:
            print(f"    loaded (streamed; size unknown)", flush=True)

        # Make sure audio column gets decoded into a numpy array
        try:
            ds = ds.cast_column(src["audio_field"], Audio(sampling_rate=16000))
        except Exception as e:
            print(f"    audio cast warning: {e}", flush=True)

        extracted_this_split = 0
        idx = 0
        for sample in ds:
            if total_extracted >= TARGET_COUNT:
                break
            idx += 1
            try:
                audio = sample.get(src["audio_field"]) or {}
                arr = audio.get("array")
                sr = audio.get("sampling_rate")
                item_id = sample.get(src["id_field"])
                transcript = sample.get(src["transcript_field"]) or ""
                gender = sample.get(src["gender_field"])
                if arr is None or sr is None:
                    continue
                wav_bytes = array_to_wav_bytes(arr, sr)
                # Use a row index in the filename to ensure uniqueness for CV
                # (where client_id repeats across rows)
                file_id = f"{item_id}_{idx}" if name == "common_voice" else str(item_id)
                out_filename = f"{prefix}_{file_id}.mp3"
                out_path = AUDIO_OUT / out_filename
                if not encode_wav_to_mp3(wav_bytes, out_path):
                    continue
                duration = get_duration(out_path)
                if duration <= 0:
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                    continue
                meta_f.write(json.dumps({
                    "id": file_id,
                    "dialect": "MSA",
                    "audio_filename": out_filename,
                    "duration_seconds": duration,
                    "source": name,
                    "config": "ar_eg" if name == "fleurs" else "ar",
                    "split": split_name,
                    "gender": gender,
                    "transcription": transcript,
                }) + "\n")
                meta_f.flush()
                extracted_this_split += 1
                total_extracted += 1
                if total_extracted % 100 == 0:
                    print(f"    extracted {total_extracted}/{TARGET_COUNT}...", flush=True)
            except Exception as e:
                print(f"    sample error (idx={idx}): {type(e).__name__}: {e}", flush=True)
                continue

        per_split[f"{name}/{split_name}"] = extracted_this_split
        print(f"    split done: {extracted_this_split} extracted (running total {total_extracted})", flush=True)

    return total_extracted


total_extracted = 0
per_split = {}
meta_f = open(META_PATH, "w", encoding="utf-8")

for src in SOURCES:
    if total_extracted >= TARGET_COUNT:
        break
    try:
        total_extracted = run_source(src, total_extracted, meta_f, per_split)
    except Exception as e:
        print(f"FATAL while processing {src['name']}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        continue

meta_f.close()


# ============================================================
# Cell 3: package
# ============================================================
SUMMARY_PATH.write_text(json.dumps({
    "total_msa_extracted": total_extracted,
    "per_split": per_split,
    "sources_attempted": [s["name"] for s in SOURCES],
}, indent=2), encoding="utf-8")

print(f"\nPackaging into {ZIP_PATH}...", flush=True)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as z:
    z.write(META_PATH, arcname="msa_metadata.jsonl")
    z.write(SUMMARY_PATH, arcname="summary.json")
    for mp3 in AUDIO_OUT.glob("*.mp3"):
        z.write(mp3, arcname=f"audio/{mp3.name}")

mb = ZIP_PATH.stat().st_size / 1e6
print(f"Done. {ZIP_PATH} = {mb:.0f} MB. Total MSA items: {total_extracted}.")
print("Download fleurs_msa.zip from Colab's file panel and run scripts/10_import_colab_msa.py locally.")
