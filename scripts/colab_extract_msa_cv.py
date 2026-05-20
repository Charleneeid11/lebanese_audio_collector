"""
Colab notebook script: extract up to 1,000 MSA audio items from FLEURS Arabic.

Run in Google Colab — paste this whole file in one cell, uncomment !pip / !apt-get
lines, run.

Why FLEURS (`google/fleurs`, config "ar_eg"):
  - MSA register: Arabic FLoRes-101 sentences are written in literary Arabic (Fuṣḥā).
    Speakers read MSA prompts; Egyptian accent variation is the only regional confound.
  - Multi-speaker: many contributors → no single-speaker confound (the disqualifier
    from FINDINGS 8.2.1).
  - Citable: Conneau et al. 2022 ("FLEURS: Few-shot Learning Evaluation of Universal
    Representations of Speech").
  - License: CC-BY-SA 4.0.
  - Still on the HuggingFace Hub (Mozilla Common Voice was withdrawn from HF in
    October 2025, so CV is no longer available without going through Mozilla Data
    Collective; FLEURS is unaffected).

Note on field names: FLEURS uses `id` (int), `transcription`, and an `audio` dict
with `array` and `sampling_rate`. No `client_id`, unlike CV.

Output bundle is named /content/fleurs_msa.zip so the local importer
(scripts/10_import_colab_msa.py) auto-detects it.
"""

# ============================================================
# Cell 1: setup + auth
# ============================================================
# datasets 4.x dropped support for script-based loaders, but FLEURS uses one
# (fleurs.py). Pinning to 2.x keeps FLEURS loadable. soundfile + huggingface_hub
# stay current.
# !pip install -q "datasets==2.21.0" "huggingface_hub" "soundfile" "fsspec<=2024.12.0"
# !apt-get -qq install -y ffmpeg

import io
import json
import os
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path

# --- HF auth: pull HF_TOKEN from Colab secrets and log in (FLEURS doesn't strictly
# need this, but having auth set up unlocks fallback datasets if FLEURS is unhappy)
try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
    if _tok:
        os.environ["HF_TOKEN"] = _tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
        from huggingface_hub import login
        login(token=_tok, add_to_git_credential=False)
        print("HF auth: token loaded from Colab secrets, logged in.", flush=True)
    else:
        print("HF auth: no HF_TOKEN secret in Colab (FLEURS is ungated so this is OK).",
              flush=True)
except Exception as _e:
    print(f"HF auth: skipped ({_e}).", flush=True)

try:
    from huggingface_hub import whoami
    info = whoami()
    print(f"HF auth: logged in as {info.get('name', '?')}", flush=True)
except Exception:
    print("HF auth: anonymous (FLEURS works without login).", flush=True)

import soundfile as sf
from datasets import load_dataset, Audio


HF_DATASET = "google/fleurs"
HF_CONFIG = "ar_eg"
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
# Cell 2: load + extract
# ============================================================
print(f"Loading {HF_DATASET} config={HF_CONFIG}...", flush=True)

# FLEURS has train/validation/test; concatenate for max speaker diversity.
# Use streaming=False because FLEURS is small (~3-7 hours per language) so the
# full download is fast (~hundreds of MB) and avoids streaming-mode quirks.
SPLITS = ["train", "validation", "test"]

total_extracted = 0
per_split = {}
meta_f = open(META_PATH, "w", encoding="utf-8")

LOAD_VARIANTS = [
    {"trust_remote_code": True, "streaming": False},
    {"trust_remote_code": False, "streaming": False},
    {"trust_remote_code": True, "streaming": True},
    {"trust_remote_code": False, "streaming": True},
]

for split_name in SPLITS:
    if total_extracted >= TARGET_COUNT:
        break

    print(f"\n{'='*60}\nSplit: {split_name}", flush=True)
    ds = None
    last_err = None
    for variant in LOAD_VARIANTS:
        try:
            ds = load_dataset(HF_DATASET, HF_CONFIG, split=split_name, **variant)
            print(f"  loaded with variant={variant}", flush=True)
            break
        except Exception as e:
            last_err = e
            print(f"  variant={variant} failed: {type(e).__name__}: {e}", flush=True)
            continue

    if ds is None:
        print(f"  ALL VARIANTS FAILED for split {split_name}", flush=True)
        traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
        per_split[split_name] = 0
        continue

    try:
        n = len(ds)
        print(f"  {n} rows", flush=True)
    except Exception:
        print(f"  (streaming, size unknown)", flush=True)

    try:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        print(f"  audio cast warning: {e}", flush=True)

    extracted_this_split = 0
    idx = 0
    try:
        for sample in ds:
            if total_extracted >= TARGET_COUNT:
                break
            idx += 1
            try:
                audio = sample.get("audio") or {}
                arr = audio.get("array")
                sr = audio.get("sampling_rate")
                item_id = sample.get("id")
                transcript = sample.get("transcription") or sample.get("raw_transcription") or ""
                gender = sample.get("gender")

                if arr is None or sr is None:
                    continue

                wav_bytes = array_to_wav_bytes(arr, sr)
                file_id = f"{split_name}_{item_id}"
                out_filename = f"fleurs_MSA_{file_id}.mp3"
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
                    "source": "fleurs",
                    "config": HF_CONFIG,
                    "split": split_name,
                    "gender": gender,
                    "transcription": transcript,
                }) + "\n")
                meta_f.flush()
                extracted_this_split += 1
                total_extracted += 1
                if total_extracted % 100 == 0:
                    print(f"  extracted {total_extracted}/{TARGET_COUNT}...", flush=True)
            except Exception as e:
                print(f"  sample error (idx={idx}): {type(e).__name__}: {e}", flush=True)
                continue
    except Exception as e:
        print(f"  iteration error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    per_split[split_name] = extracted_this_split
    print(f"  split done: {extracted_this_split} extracted (running total {total_extracted})", flush=True)

meta_f.close()


# ============================================================
# Cell 3: package
# ============================================================
SUMMARY_PATH.write_text(json.dumps({
    "source": HF_DATASET,
    "config": HF_CONFIG,
    "total_msa_extracted": total_extracted,
    "per_split": per_split,
}, indent=2), encoding="utf-8")

print(f"\nPackaging into {ZIP_PATH}...", flush=True)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as z:
    z.write(META_PATH, arcname="msa_metadata.jsonl")
    z.write(SUMMARY_PATH, arcname="summary.json")
    for mp3 in AUDIO_OUT.glob("*.mp3"):
        z.write(mp3, arcname=f"audio/{mp3.name}")

mb = ZIP_PATH.stat().st_size / 1e6
print(f"\nDONE. {ZIP_PATH} = {mb:.0f} MB. Total MSA: {total_extracted}.")
print("Download fleurs_msa.zip from Colab's file panel.")
print("Then locally: .venv/Scripts/python.exe scripts/10_import_colab_msa.py")
