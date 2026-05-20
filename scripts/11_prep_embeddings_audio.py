#!/usr/bin/env python3
"""
Bundle audio for Colab embedding extraction — trimmed and recompressed to keep
the zip uploadable (~700 MB instead of ~91 GB raw).

For each selected item:
  - ffmpeg: trim to first 30 sec, mono, 16 kHz, MP3 96 kbps
  - target dir: data/audio_clips_for_embed/<item_id>.mp3
  - resumable: skips items whose clip already exists at expected size

Selection (same as before):
  - Lebanese candidates: WEAK_POSITIVE, POTENTIAL_LB, BORDERLINE_LB
  - Non-Lebanese contrastive: ADI17 (DOWNLOADED), FLEURS (MSA)
  - Lexical negatives: WEAK_NEGATIVE
  - Ground-truth annotated items (any status)

Output:
  - data/audio_for_embeddings.zip  (audio/ + manifest.json)
  - data/audio_clips_for_embed/    (intermediate trimmed clips, kept for resumability)

Run: python scripts/11_prep_embeddings_audio.py [--workers N]
"""

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.cfg import Settings
from src.db import DB, QueueItem


OUT_ZIP = Path("data/audio_for_embeddings.zip")
CLIPS_DIR = Path("data/audio_clips_for_embed")
ANNOTATIONS_CSV = Path("data/annotations.csv")

INCLUDE_STATUSES = {
    "WEAK_POSITIVE",
    "POTENTIAL_LB",
    "BORDERLINE_LB",
    "WEAK_NEGATIVE",
    "DOWNLOADED",
}

CLIP_SECONDS = 30  # wav2vec2 needs only ~10-30s for utterance-level embeddings


def load_ground_truth() -> dict[str, dict]:
    gt = {}
    if not ANNOTATIONS_CSV.exists():
        print(f"  (no {ANNOTATIONS_CSV} — skipping ground-truth join)")
        return gt
    with open(ANNOTATIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            iid = row.get("item_id") or row.get("id")
            if iid is None:
                continue
            gt[str(iid)] = {
                "gt_label": row.get("ground_truth") or row.get("label"),
                "gt_index": i,
            }
    print(f"  loaded {len(gt)} ground-truth annotations")
    return gt


def trim_one(args: tuple[int, str, str]) -> tuple[int, bool, str]:
    """ffmpeg-trim a single file. Returns (item_id, ok, error_msg)."""
    item_id, src_path, dst_path = args
    dst = Path(dst_path)
    src = Path(src_path)
    if dst.exists() and dst.stat().st_size > 1024:
        return (item_id, True, "skip-cached")
    if not src.exists():
        return (item_id, False, "src-missing")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", "0", "-t", str(CLIP_SECONDS),
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "96k",
        str(dst),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0:
            return (item_id, False, f"ffmpeg: {r.stderr.decode('utf-8', 'ignore')[:200]}")
        if not dst.exists() or dst.stat().st_size < 1024:
            return (item_id, False, "output-too-small")
        return (item_id, True, "ok")
    except subprocess.TimeoutExpired:
        return (item_id, False, "timeout")
    except Exception as e:
        return (item_id, False, f"{type(e).__name__}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(2, mp.cpu_count() - 1))
    args = parser.parse_args()

    settings = Settings.load()
    db = DB(settings.db_url)

    print("Loading ground truth...")
    gt = load_ground_truth()

    print("Querying queue.db...")
    with Session(db.engine) as session:
        items = session.scalars(select(QueueItem)).all()

    selected = []
    for it in items:
        iid_str = str(it.id)
        in_gt = iid_str in gt
        if not in_gt and it.status not in INCLUDE_STATUSES:
            continue
        if not it.audio_path:
            continue
        path = Path(it.audio_path)
        if not path.exists():
            continue
        meta = it.source_metadata or {}
        selected.append({
            "item_id": it.id,
            "audio_path_local": str(path),
            "platform": it.platform,
            "status": it.status,
            "duration_seconds": it.duration_seconds,
            "contrastive_role": meta.get("contrastive_role"),
            "adi17_dialect": meta.get("adi17_dialect"),
            "source": meta.get("source"),
            "gt_label": gt.get(iid_str, {}).get("gt_label"),
            "gt_index": gt.get(iid_str, {}).get("gt_index"),
        })

    print(f"\nSelected: {len(selected)} items")
    from collections import Counter
    by_status = Counter(s["status"] for s in selected)
    print("By status:")
    for s, c in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")
    by_role = Counter(s.get("contrastive_role") or "lebanese_candidate" for s in selected)
    print("By contrastive_role:")
    for r, c in sorted(by_role.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the work list: (item_id, src, dst)
    jobs = []
    for s in selected:
        dst = CLIPS_DIR / f"item_{s['item_id']}.mp3"
        jobs.append((s["item_id"], s["audio_path_local"], str(dst)))

    print(f"\nTrimming {len(jobs)} files to {CLIP_SECONDS}s with {args.workers} workers...")
    t0 = time.time()
    ok = 0
    cached = 0
    fail = 0
    fail_examples = []
    with mp.Pool(args.workers) as pool:
        for i, (iid, success, msg) in enumerate(pool.imap_unordered(trim_one, jobs, chunksize=4)):
            if success:
                if msg == "skip-cached":
                    cached += 1
                else:
                    ok += 1
            else:
                fail += 1
                if len(fail_examples) < 5:
                    fail_examples.append((iid, msg))
            if (i + 1) % 500 == 0:
                rate = (i + 1) / max(time.time() - t0, 1)
                eta = (len(jobs) - (i + 1)) / max(rate, 0.01)
                print(f"  {i+1}/{len(jobs)}  ok={ok} cached={cached} fail={fail}  ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nTrim phase: {ok} new, {cached} cached, {fail} failed in {elapsed:.0f}s")
    if fail_examples:
        print("First failures:")
        for iid, msg in fail_examples:
            print(f"  item {iid}: {msg}")

    # Build manifest only for items where the clip exists
    manifest = []
    missing = 0
    for s in selected:
        clip = CLIPS_DIR / f"item_{s['item_id']}.mp3"
        if not clip.exists() or clip.stat().st_size < 1024:
            missing += 1
            continue
        manifest.append({
            "item_id": s["item_id"],
            "audio_filename": clip.name,
            "platform": s["platform"],
            "status": s["status"],
            "duration_seconds": s["duration_seconds"],
            "contrastive_role": s["contrastive_role"],
            "adi17_dialect": s["adi17_dialect"],
            "source": s["source"],
            "gt_label": s["gt_label"],
            "gt_index": s["gt_index"],
        })
    print(f"\nManifest: {len(manifest)} items ({missing} missing clips excluded)")

    if OUT_ZIP.exists():
        print(f"Deleting existing {OUT_ZIP}...")
        OUT_ZIP.unlink()

    print(f"Writing {OUT_ZIP}...")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:
        for i, m in enumerate(manifest):
            clip = CLIPS_DIR / m["audio_filename"]
            z.write(clip, arcname=f"audio/{m['audio_filename']}")
            if (i + 1) % 2000 == 0:
                print(f"  zipped {i+1}/{len(manifest)}...")
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

    final_size = OUT_ZIP.stat().st_size
    print(f"\nDone. {OUT_ZIP} = {final_size/1e6:.0f} MB ({len(manifest)} items)")
    print(f"\nNext: upload {OUT_ZIP.name} to Colab and run scripts/colab_extract_embeddings.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
