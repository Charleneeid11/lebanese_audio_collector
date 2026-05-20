#!/usr/bin/env python3
"""
Re-encode the trimmed clips at 10 sec / 64 kbps for a much smaller upload.

Reads from data/audio_clips_for_embed/*.mp3 (already 30s @ 96k) and rewrites
to data/audio_clips_compact/*.mp3 at 10s @ 64k mono 16kHz. Then bundles into
data/audio_for_embeddings_compact.zip.

10 sec is the standard window for utterance-level wav2vec2/XLS-R embeddings in
dialect ID literature (e.g., Babu 2022, Conneau 2022). 64 kbps mono 16 kHz keeps
the audio recognizable while cutting size further.

Run: python scripts/11b_compact_embeddings_zip.py
"""

import json
import multiprocessing as mp
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

SRC_DIR = Path("data/audio_clips_for_embed")
DST_DIR = Path("data/audio_clips_compact")
ORIGINAL_ZIP = Path("data/audio_for_embeddings.zip")
OUT_ZIP = Path("data/audio_for_embeddings_compact.zip")
CLIP_SECONDS = 10


def reencode(args: tuple[str, str]) -> tuple[str, bool, str]:
    src, dst = args
    dst_p = Path(dst)
    if dst_p.exists() and dst_p.stat().st_size > 1024:
        return (src, True, "cached")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", "0", "-t", str(CLIP_SECONDS),
        "-i", src,
        "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "64k",
        dst,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0:
            return (src, False, r.stderr.decode("utf-8", "ignore")[:200])
        return (src, True, "ok")
    except Exception as e:
        return (src, False, f"{type(e).__name__}: {e}")


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the manifest from the existing 30s zip
    print(f"Reading manifest from {ORIGINAL_ZIP}...")
    with zipfile.ZipFile(ORIGINAL_ZIP) as z:
        manifest = json.loads(z.read("manifest.json"))
    print(f"  {len(manifest)} entries")

    jobs = []
    for m in manifest:
        src = SRC_DIR / m["audio_filename"]
        dst = DST_DIR / m["audio_filename"]
        if not src.exists():
            continue
        jobs.append((str(src), str(dst)))

    print(f"\nRe-encoding {len(jobs)} clips to {CLIP_SECONDS}s @ 64k...")
    workers = max(2, mp.cpu_count() - 1)
    t0 = time.time()
    ok = cached = fail = 0
    with mp.Pool(workers) as pool:
        for i, (src, success, msg) in enumerate(pool.imap_unordered(reencode, jobs, chunksize=4)):
            if success:
                if msg == "cached":
                    cached += 1
                else:
                    ok += 1
            else:
                fail += 1
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1)
                print(f"  {i+1}/{len(jobs)}  ok={ok} cached={cached} fail={fail}  ({rate:.0f}/s)", flush=True)
    print(f"\nRe-encode: {ok} new, {cached} cached, {fail} failed in {time.time()-t0:.0f}s")

    # Filter manifest to only items with a valid compact clip
    manifest_out = []
    for m in manifest:
        clip = DST_DIR / m["audio_filename"]
        if clip.exists() and clip.stat().st_size > 1024:
            manifest_out.append(m)

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    print(f"\nZipping {len(manifest_out)} clips to {OUT_ZIP}...")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:
        for i, m in enumerate(manifest_out):
            clip = DST_DIR / m["audio_filename"]
            z.write(clip, arcname=f"audio/{m['audio_filename']}")
            if (i + 1) % 2000 == 0:
                print(f"  zipped {i+1}/{len(manifest_out)}...")
        z.writestr("manifest.json", json.dumps(manifest_out, indent=2))

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"\nDone. {OUT_ZIP} = {size_mb:.0f} MB ({len(manifest_out)} items)")


if __name__ == "__main__":
    main()
