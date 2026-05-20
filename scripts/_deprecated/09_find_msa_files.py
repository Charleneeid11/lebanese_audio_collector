#!/usr/bin/env python3
"""
Identify which ADI17 train Parquet files contain MSA — without downloading audio.

Reads ONLY the `dialect` column from each Parquet file via column projection over
HTTP range requests. PyArrow + fsspec fetches just the column chunks for `dialect`,
which is tiny (strings like "MSA", "EGY") compared to the full file (audio bytes).

Output: data/adi17_msa_candidates.json — for each file: total rows, dialect counts,
MSA row indices.

Run: python scripts/09_find_msa_files.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


REPO_PATH = "datasets/ArabicSpeech/ADI17/data"
TRAIN_FILES = [f"train-{i:05d}-of-00040.parquet" for i in range(40)]
TARGET = "MSA"
OUT_PATH = Path("data/adi17_msa_candidates.json")


def scan_file(fs: HfFileSystem, fname: str) -> dict:
    full = f"{REPO_PATH}/{fname}"
    t0 = time.time()
    with fs.open(full, "rb") as f:
        table = pq.read_table(f, columns=["dialect"])
    dialects = table.column("dialect").to_pylist()
    counts = Counter(dialects)
    msa_indices = [i for i, d in enumerate(dialects) if d == TARGET]
    elapsed = time.time() - t0
    return {
        "file": fname,
        "total_rows": len(dialects),
        "dialect_counts": dict(counts),
        "msa_count": len(msa_indices),
        "msa_indices_first_last": [msa_indices[0], msa_indices[-1]] if msa_indices else None,
        "scan_seconds": round(elapsed, 1),
    }


def main():
    fs = HfFileSystem()
    results = []
    print(f"Scanning {len(TRAIN_FILES)} train files (dialect column only)...\n", flush=True)

    total_msa = 0
    for fname in TRAIN_FILES:
        try:
            r = scan_file(fs, fname)
            results.append(r)
            total_msa += r["msa_count"]
            top = sorted(r["dialect_counts"].items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}={v}" for k, v in top)
            marker = "*MSA*" if r["msa_count"] > 0 else "     "
            print(f"  {marker} {fname}: msa={r['msa_count']:>5}/{r['total_rows']} ({r['scan_seconds']}s) | top: {top_str}", flush=True)
        except Exception as e:
            print(f"  ERR  {fname}: {type(e).__name__}: {e}", flush=True)
            results.append({"file": fname, "error": f"{type(e).__name__}: {e}"})

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved scan results to {OUT_PATH}")
    print(f"Total MSA rows across train split: {total_msa}")
    candidates = [r for r in results if r.get("msa_count", 0) > 0]
    print(f"Files containing MSA: {len(candidates)}")
    for c in candidates:
        print(f"  {c['file']}: {c['msa_count']} MSA rows")


if __name__ == "__main__":
    main()
