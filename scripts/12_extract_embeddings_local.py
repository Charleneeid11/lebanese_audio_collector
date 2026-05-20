#!/usr/bin/env python3
"""
Extract wav2vec2-xls-r-300m embeddings locally on CPU.

Reads 10-second MP3 clips from data/audio_clips_compact/ (produced by
scripts/11b_compact_embeddings_zip.py) and writes utterance-level mean-pooled
embeddings to data/embeddings.parquet.

Model: facebook/wav2vec2-xls-r-300m (Babu et al. 2022) — 300M-param multilingual
speech encoder. Same model the Colab script would have used; this is the local
CPU version.

Strategy:
  - PyTorch with set_num_threads(N-1) to use all CPU cores
  - Batch size 4 (kept small to avoid CPU memory pressure)
  - Mean-pool last hidden state over time → 1024-dim embedding
  - Append rows to parquet every 200 items (resumable: a re-run skips items
    already in the parquet)

Output:
  - data/embeddings.parquet — columns: item_id, audio_filename, platform,
    status, duration_seconds, contrastive_role, adi17_dialect, source,
    gt_label, gt_index, embedding (1024-dim float32 list)
  - data/embeddings_summary.json — model name, n_rows, n_failed, elapsed

Run: python scripts/12_extract_embeddings_local.py
Resumable: re-running picks up where it left off.
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModel


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
DEFAULT_BATCH = 4

INPUT_ZIP = Path("data/audio_for_embeddings_compact.zip")
CLIPS_DIR = Path("data/audio_clips_compact")
OUT_PARQUET = Path("data/embeddings.parquet")
SUMMARY_PATH = Path("data/embeddings_summary.json")


def load_manifest() -> list[dict]:
    """Read manifest.json from the compact zip; do not extract the audio
    (audio already lives in CLIPS_DIR locally)."""
    with zipfile.ZipFile(INPUT_ZIP) as z:
        return json.loads(z.read("manifest.json"))


def load_audio(path: Path) -> np.ndarray | None:
    try:
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if y.size == 0:
            return None
        return y.astype(np.float32)
    except Exception as e:
        print(f"    audio load failed for {path.name}: {type(e).__name__}: {e}", flush=True)
        return None


def embed_batch(model, feature_extractor, arrays: list[np.ndarray]) -> np.ndarray:
    inputs = feature_extractor(
        arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        out = model(**inputs)
    hidden = out.last_hidden_state          # (B, T, 1024)
    mask = inputs.get("attention_mask")
    if mask is not None:
        try:
            feat_lens = model._get_feat_extract_output_lengths(mask.sum(dim=1))
        except Exception:
            feat_lens = torch.full((mask.size(0),), hidden.size(1), dtype=torch.long)
        T = hidden.size(1)
        feat_mask = (
            torch.arange(T).unsqueeze(0) < feat_lens.unsqueeze(1)
        ).unsqueeze(-1).float()
        summed = (hidden * feat_mask).sum(dim=1)
        denom = feat_mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
    else:
        pooled = hidden.mean(dim=1)
    return pooled.numpy()


def existing_ids() -> set[int]:
    if not OUT_PARQUET.exists():
        return set()
    df = pd.read_parquet(OUT_PARQUET, columns=["item_id"])
    return set(df["item_id"].tolist())


def append_rows(rows: list[dict]) -> None:
    """Append rows to the parquet (read existing, concat, write back).
    Parquet doesn't have native append, so we do read-concat-write.
    For 14K rows + 1024-dim embeddings, this is fast enough."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if OUT_PARQUET.exists():
        old = pd.read_parquet(OUT_PARQUET)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_parquet(OUT_PARQUET, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--max-clip-seconds", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N items (debugging). 0 = all.")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"PyTorch threads: {args.threads}, batch size: {args.batch}", flush=True)

    if not INPUT_ZIP.exists():
        sys.exit(f"ERROR: {INPUT_ZIP} not found. Run scripts/11b_compact_embeddings_zip.py first.")
    if not CLIPS_DIR.exists():
        sys.exit(f"ERROR: {CLIPS_DIR} not found.")

    print("Loading manifest...", flush=True)
    manifest = load_manifest()
    print(f"  {len(manifest)} items in manifest", flush=True)

    done = existing_ids()
    remaining = [m for m in manifest if m["item_id"] not in done]
    print(f"  already done: {len(done)} | remaining: {len(remaining)}", flush=True)

    if args.limit:
        remaining = remaining[: args.limit]
        print(f"  limited to first {len(remaining)} (debug)", flush=True)
    if not remaining:
        print("Nothing to do.", flush=True)
        return

    print(f"\nLoading {MODEL_NAME} (will download on first run, ~1.2 GB)...", flush=True)
    t0 = time.time()
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  model loaded in {time.time()-t0:.0f}s", flush=True)

    fail_count = 0
    pending_rows: list[dict] = []
    batch_arrays: list[np.ndarray] = []
    batch_meta: list[dict] = []
    t0 = time.time()
    FLUSH_EVERY = 200

    for i, m in enumerate(remaining):
        clip = CLIPS_DIR / m["audio_filename"]
        arr = load_audio(clip)
        if arr is None:
            fail_count += 1
            continue
        max_samples = SAMPLE_RATE * args.max_clip_seconds
        if arr.shape[0] > max_samples:
            arr = arr[:max_samples]

        batch_arrays.append(arr)
        batch_meta.append(m)

        if len(batch_arrays) >= args.batch:
            try:
                embs = embed_batch(model, feature_extractor, batch_arrays)
                for meta, emb in zip(batch_meta, embs):
                    pending_rows.append({
                        **meta,
                        "embedding": emb.astype(np.float32).tolist(),
                    })
            except Exception as e:
                print(f"    batch error: {type(e).__name__}: {e}", flush=True)
                fail_count += len(batch_arrays)
            batch_arrays, batch_meta = [], []

        if len(pending_rows) >= FLUSH_EVERY:
            append_rows(pending_rows)
            pending_rows = []

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(remaining) - (i + 1)) / max(rate, 0.001)
            print(
                f"  {i+1}/{len(remaining)}  "
                f"rate={rate:.2f}/s  "
                f"elapsed={elapsed/60:.1f}min  "
                f"eta={eta/60:.0f}min  "
                f"fails={fail_count}",
                flush=True,
            )

    # Flush trailing batch
    if batch_arrays:
        try:
            embs = embed_batch(model, feature_extractor, batch_arrays)
            for meta, emb in zip(batch_meta, embs):
                pending_rows.append({**meta, "embedding": emb.astype(np.float32).tolist()})
        except Exception as e:
            print(f"  final batch error: {e}", flush=True)
            fail_count += len(batch_arrays)
    append_rows(pending_rows)

    elapsed = time.time() - t0
    n_rows = len(pd.read_parquet(OUT_PARQUET, columns=["item_id"])) if OUT_PARQUET.exists() else 0
    print(f"\nDone. {n_rows} embeddings in {OUT_PARQUET} ({fail_count} failed) in {elapsed/60:.0f}min.", flush=True)

    SUMMARY_PATH.write_text(json.dumps({
        "model": MODEL_NAME,
        "embedding_dim": 1024,
        "n_rows": n_rows,
        "n_failed": fail_count,
        "elapsed_seconds": round(elapsed, 1),
        "device": "cpu",
        "torch_threads": args.threads,
        "batch_size": args.batch,
        "max_clip_seconds": args.max_clip_seconds,
        "sample_rate": SAMPLE_RATE,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
