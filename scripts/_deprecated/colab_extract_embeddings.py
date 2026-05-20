"""
Colab notebook script: extract wav2vec2-xls-r-300m embeddings for all audio in
audio_for_embeddings.zip uploaded by the user.

Why wav2vec2-xls-r-300m (`facebook/wav2vec2-xls-r-300m`):
  - 300M-parameter multilingual self-supervised speech model (Babu et al. 2022,
    "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale").
  - Pretrained on 436K hours of speech in 128 languages, including Arabic.
  - Standard backbone for dialect ID, accent ID, and low-resource speech tasks
    in 2022+ literature.
  - Captures pronunciation/phonetic features → directly addresses the "no
    acoustic features" limitation in CLAUDE.md.
  - Mean-pool the last hidden state → 1024-dim utterance embedding.

Hardware: requires Colab GPU (Runtime → Change runtime type → T4 GPU).
Estimated time: ~30-60 min for ~10,000 audio files on T4.

Steps for the user:
  1. Runtime → Change runtime type → GPU (T4 is free)
  2. Upload data/audio_for_embeddings.zip to /content/ via the file panel
  3. Paste this script into a single cell, uncomment !pip / !apt-get, run
  4. When done, download /content/embeddings.zip (contains embeddings.parquet
     and manifest.json) and run scripts/12_load_embeddings.py locally
"""

# ============================================================
# Cell 1: setup
# ============================================================
# !pip install -q -U transformers torch torchaudio librosa pandas pyarrow soundfile
# !apt-get -qq install -y ffmpeg

import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModel


# Verify GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cpu":
    print("WARNING: GPU not available. Embedding extraction will be very slow.", flush=True)
    print("Switch to a GPU runtime: Runtime > Change runtime type > T4 GPU", flush=True)


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
BATCH_SIZE = 8  # T4 fits ~8 with no GC issues; bump on better GPUs

WORK = Path("/content/embed_work")
WORK.mkdir(exist_ok=True)
EXTRACT_DIR = WORK / "extracted"
INPUT_ZIP = Path("/content/audio_for_embeddings.zip")
OUT_PARQUET = WORK / "embeddings.parquet"
OUT_ZIP = Path("/content/embeddings.zip")


# ============================================================
# Cell 2: download audio zip (auto from GitHub Release) + extract
# ============================================================
# Hosted as a public GitHub Release asset. URL is direct, no auth, no rate limit.
ASSET_URL = "https://github.com/Charleneeid116/thesis-data/releases/download/v0.1-embeddings-input/audio_for_embeddings_compact.zip"

if not INPUT_ZIP.exists():
    print(f"Fetching audio zip from {ASSET_URL}...", flush=True)
    import requests
    with requests.get(ASSET_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(INPUT_ZIP, "wb") as f:
            got = 0
            next_report = 100 * 1024 * 1024
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
                if got >= next_report:
                    pct = (100 * got / total) if total else 0
                    print(f"    {got/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)", flush=True)
                    next_report += 100 * 1024 * 1024
    print(f"  done: {INPUT_ZIP.stat().st_size/1e6:.0f} MB", flush=True)

print(f"Extracting {INPUT_ZIP}...", flush=True)
if EXTRACT_DIR.exists():
    import shutil
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ZIP) as z:
    z.extractall(EXTRACT_DIR)

manifest_path = EXTRACT_DIR / "manifest.json"
audio_dir = EXTRACT_DIR / "audio"
if not manifest_path.exists() or not audio_dir.exists():
    raise SystemExit("ERROR: zip missing manifest.json or audio/ directory")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(f"Manifest entries: {len(manifest)}", flush=True)
print(f"Audio files in zip: {len(list(audio_dir.glob('*.mp3')))}", flush=True)


# ============================================================
# Cell 3: load model
# ============================================================
print(f"\nLoading {MODEL_NAME}...", flush=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded.", flush=True)


# ============================================================
# Cell 4: extract embeddings
# ============================================================
def load_audio(path: Path) -> np.ndarray | None:
    """Load audio as mono 16kHz float32 numpy array."""
    try:
        # MP3 → librosa handles it (decodes via audioread/ffmpeg under the hood)
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if y.size == 0:
            return None
        return y.astype(np.float32)
    except Exception as e:
        print(f"    audio load failed for {path.name}: {type(e).__name__}: {e}", flush=True)
        return None


def embed_batch(arrays: list[np.ndarray]) -> np.ndarray:
    """Run a batch through the model, return mean-pooled last hidden states (B, 1024)."""
    inputs = feature_extractor(
        arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # last_hidden_state shape: (B, T, 1024). Mean pool over time using attention_mask.
    hidden = out.last_hidden_state
    mask = inputs.get("attention_mask")
    if mask is not None:
        # wav2vec2 output time dim is downsampled; need to recompute mask in feature space.
        # transformers provides _get_feat_extract_output_lengths for this.
        try:
            feat_lens = model._get_feat_extract_output_lengths(mask.sum(dim=1))
        except Exception:
            feat_lens = torch.full(
                (mask.size(0),), hidden.size(1), dtype=torch.long, device=DEVICE
            )
        T = hidden.size(1)
        feat_mask = (
            torch.arange(T, device=DEVICE).unsqueeze(0) < feat_lens.unsqueeze(1)
        ).unsqueeze(-1)
        summed = (hidden * feat_mask.float()).sum(dim=1)
        denom = feat_mask.float().sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
    else:
        pooled = hidden.mean(dim=1)
    return pooled.cpu().numpy()


print(f"\nExtracting embeddings (batch_size={BATCH_SIZE})...", flush=True)

rows = []
batch_arrays: list[np.ndarray] = []
batch_meta: list[dict] = []
t0 = time.time()
fail_count = 0

for i, entry in enumerate(manifest):
    audio_path = audio_dir / entry["audio_filename"]
    arr = load_audio(audio_path)
    if arr is None:
        fail_count += 1
        continue

    # Cap clip length to avoid OOM on extremely long files (e.g., 20-min podcasts).
    # 60 sec at 16kHz = 960k samples; safe on T4.
    MAX_SAMPLES = SAMPLE_RATE * 60
    if arr.shape[0] > MAX_SAMPLES:
        arr = arr[:MAX_SAMPLES]

    batch_arrays.append(arr)
    batch_meta.append(entry)

    if len(batch_arrays) >= BATCH_SIZE:
        try:
            embs = embed_batch(batch_arrays)
            for m, e in zip(batch_meta, embs):
                rows.append({**m, "embedding": e.astype(np.float32).tolist()})
        except Exception as e:
            print(f"    batch error: {type(e).__name__}: {e}", flush=True)
            fail_count += len(batch_arrays)
        batch_arrays, batch_meta = [], []

    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1)
        eta = (len(manifest) - (i + 1)) / max(rate, 0.01)
        print(f"  {i+1}/{len(manifest)} ({rate:.1f}/s, ETA {eta:.0f}s, fails {fail_count})", flush=True)

# Flush remaining batch
if batch_arrays:
    try:
        embs = embed_batch(batch_arrays)
        for m, e in zip(batch_meta, embs):
            rows.append({**m, "embedding": e.astype(np.float32).tolist()})
    except Exception as e:
        print(f"  final batch error: {e}", flush=True)
        fail_count += len(batch_arrays)

elapsed = time.time() - t0
print(f"\nExtracted {len(rows)} embeddings in {elapsed:.0f}s ({fail_count} failed)", flush=True)


# ============================================================
# Cell 5: save and package
# ============================================================
df = pd.DataFrame(rows)
print(f"DataFrame shape: {df.shape}, embedding dim check: {len(df.iloc[0]['embedding']) if len(df) > 0 else 0}", flush=True)

df.to_parquet(OUT_PARQUET, index=False)
print(f"Wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.0f} MB)", flush=True)

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(OUT_PARQUET, arcname="embeddings.parquet")
    z.write(manifest_path, arcname="manifest.json")
    summary = {
        "model": MODEL_NAME,
        "embedding_dim": int(df.iloc[0]["embedding"].__len__()) if len(df) > 0 else 0,
        "n_rows": len(df),
        "n_failed": fail_count,
        "elapsed_seconds": round(elapsed, 1),
        "device": DEVICE,
        "max_clip_seconds": 60,
        "sample_rate": SAMPLE_RATE,
    }
    z.writestr("summary.json", json.dumps(summary, indent=2))

print(f"\nDONE. Download {OUT_ZIP} ({OUT_ZIP.stat().st_size/1e6:.0f} MB).")
print("Then locally: .venv/Scripts/python.exe scripts/12_load_embeddings.py")
