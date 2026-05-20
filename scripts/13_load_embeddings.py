#!/usr/bin/env python3
"""
Load wav2vec2-xls-r-300m embeddings and join with queue.db to produce a
single training-ready parquet.

Inputs (in order of preference):
  data/embeddings.parquet — produced by scripts/12_extract_embeddings_local.py.
  data/embeddings.zip — legacy Colab output (extracts embeddings.parquet).

Output:
  data/embeddings_with_labels.parquet — columns:
    item_id, embedding (np.ndarray), lebanese (0/1/None), gt_label, in_ground_truth,
    status, platform, adi17_dialect, contrastive_role

Binary `lebanese` label rules:
  Positive (1):
    - status == 'POTENTIAL_LB'
    - status == 'WEAK_POSITIVE'  (further filtering applied at training time)
    - platform == 'adi17' and adi17_dialect == 'LEB'
  Negative (0):
    - status == 'WEAK_NEGATIVE'
    - platform == 'adi17' and adi17_dialect in {EGY, KSA, KUW, UAE, QAT, OMA}
    - platform == 'fleurs'
  Otherwise:
    - None  (e.g., BORDERLINE_LB; not used for training)

Read-only: does not modify queue.db.

Run: python scripts/13_load_embeddings.py
"""

import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.cfg import Settings
from src.db import DB, QueueItem


DIRECT_PARQUET = Path("data/embeddings.parquet")
ZIP_PATH = Path("data/embeddings.zip")
EXTRACT_DIR = Path("data/embeddings_unzip")
OUT_PARQUET = Path("data/embeddings_with_labels.parquet")

NEGATIVE_ADI17_DIALECTS = {"EGY", "KSA", "KUW", "UAE", "QAT", "OMA"}


def derive_label(row: pd.Series, db_meta: dict | None) -> int | None:
    """Apply the binary 'lebanese' label rules. Returns 1, 0, or None."""
    status = row.get("status")
    platform = row.get("platform")
    adi17_dialect = row.get("adi17_dialect")
    if (db_meta or {}):
        # Prefer DB metadata if available (parquet may be stale on edges)
        adi17_dialect = adi17_dialect or db_meta.get("adi17_dialect")

    # Positive rules
    if status == "POTENTIAL_LB":
        return 1
    if status == "WEAK_POSITIVE":
        return 1
    if platform == "adi17" and adi17_dialect == "LEB":
        return 1
    # Negative rules
    if status == "WEAK_NEGATIVE":
        return 0
    if platform == "adi17" and adi17_dialect in NEGATIVE_ADI17_DIALECTS:
        return 0
    if platform == "fleurs":
        return 0
    return None


def main() -> int:
    # Prefer the local parquet (from 12_extract_embeddings_local.py); fall back
    # to the legacy Colab zip layout.
    if DIRECT_PARQUET.exists():
        parquet_path = DIRECT_PARQUET
        summary_path = Path("data/embeddings_summary.json")
        print(f"Using local extractor output: {parquet_path}")
    elif ZIP_PATH.exists():
        print(f"Extracting {ZIP_PATH} -> {EXTRACT_DIR}/...")
        if EXTRACT_DIR.exists():
            shutil.rmtree(EXTRACT_DIR)
        EXTRACT_DIR.mkdir(parents=True)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(EXTRACT_DIR)
        parquet_path = EXTRACT_DIR / "embeddings.parquet"
        summary_path = EXTRACT_DIR / "summary.json"
        if not parquet_path.exists():
            print(f"ERROR: {parquet_path} not in zip.")
            return 1
    else:
        print(f"ERROR: neither {DIRECT_PARQUET} nor {ZIP_PATH} found.")
        print("Run scripts/12_extract_embeddings_local.py first.")
        return 1

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Embedding extraction summary: {summary}")

    print(f"\nLoading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  rows: {len(df)}, columns: {list(df.columns)}")

    # Sanity-check embedding dim (wav2vec2-xls-r-300m -> 1024)
    if "embedding" not in df.columns:
        print("ERROR: 'embedding' column missing from parquet.")
        return 1
    sample_dim = len(df.iloc[0]["embedding"])
    print(f"  embedding dim: {sample_dim}")
    if sample_dim != 1024:
        print(f"  WARNING: expected 1024 dims (xls-r-300m), got {sample_dim}. Continuing anyway.")

    # ------------------------------------------------------------------
    # Pull current DB state for the items we have, so labels reflect the
    # latest queue.db (parquet was frozen at Colab job start; status may
    # have moved since then).
    # ------------------------------------------------------------------
    print("\nReading queue.db (read-only)...")
    settings = Settings.load()
    db = DB(settings.db_url)
    item_ids = df["item_id"].tolist()
    with Session(db.engine) as session:
        rows = session.scalars(
            select(QueueItem).where(QueueItem.id.in_(item_ids))
        ).all()
    db_by_id: dict[int, dict] = {
        r.id: {
            "status": r.status,
            "platform": r.platform,
            "source_metadata": r.source_metadata or {},
        }
        for r in rows
    }
    missing_in_db = sum(1 for iid in item_ids if iid not in db_by_id)
    if missing_in_db:
        print(f"  WARNING: {missing_in_db} item_ids in parquet not found in queue.db.")

    # ------------------------------------------------------------------
    # Refresh status / platform / adi17_dialect from DB so label rules use
    # the current truth. Fall back to parquet values if DB is missing the row.
    # ------------------------------------------------------------------
    statuses = []
    platforms = []
    adi17_dialects = []
    contrastive_roles = []
    sources = []
    for _, row in df.iterrows():
        rec = db_by_id.get(row["item_id"], {})
        meta = rec.get("source_metadata", {}) if rec else {}
        statuses.append(rec.get("status") or row.get("status"))
        platforms.append(rec.get("platform") or row.get("platform"))
        adi17_dialects.append(meta.get("adi17_dialect") or row.get("adi17_dialect"))
        contrastive_roles.append(meta.get("contrastive_role") or row.get("contrastive_role"))
        sources.append(meta.get("source") or row.get("source"))

    df["status"] = statuses
    df["platform"] = platforms
    df["adi17_dialect"] = adi17_dialects
    df["contrastive_role"] = contrastive_roles
    df["source"] = sources

    # ------------------------------------------------------------------
    # Derive binary labels and the held-out flag
    # ------------------------------------------------------------------
    print("\nDeriving labels...")
    labels: list[int | None] = []
    for _, row in df.iterrows():
        rec = db_by_id.get(row["item_id"], {})
        labels.append(derive_label(row, rec.get("source_metadata", {})))
    df["lebanese"] = labels

    df["in_ground_truth"] = df["gt_label"].notna() & (df["gt_label"].astype(str).str.strip() != "")

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------
    print("\nLabel distribution:")
    by_label = Counter(df["lebanese"])
    for k in [1, 0, None]:
        print(f"  lebanese={k!s:>5}: {by_label.get(k, 0)}")

    print("\nIn ground truth:")
    print(f"  in_ground_truth=True : {df['in_ground_truth'].sum()}")
    print(f"  in_ground_truth=False: {(~df['in_ground_truth']).sum()}")

    print("\nGround-truth label values (raw):")
    gt_counter = Counter(df.loc[df["in_ground_truth"], "gt_label"].fillna("__none__"))
    for k, v in gt_counter.most_common():
        print(f"  {k}: {v}")

    print("\nBy platform:")
    for k, v in Counter(df["platform"]).most_common():
        print(f"  {k}: {v}")

    print("\nBy status:")
    for k, v in Counter(df["status"]).most_common():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Save (keep only the columns the trainer needs)
    # ------------------------------------------------------------------
    out_cols = [
        "item_id",
        "embedding",
        "lebanese",
        "gt_label",
        "in_ground_truth",
        "status",
        "platform",
        "adi17_dialect",
        "contrastive_role",
        "source",
        "audio_filename",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].copy()

    # Convert embedding lists to numpy arrays for downstream speed/correctness
    df_out["embedding"] = df_out["embedding"].apply(
        lambda x: np.asarray(x, dtype=np.float32)
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / 1e6
    print(f"\nWrote {OUT_PARQUET} ({size_mb:.1f} MB, {len(df_out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
