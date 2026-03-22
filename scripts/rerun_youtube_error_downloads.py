#!/usr/bin/env python3
"""
Reset YouTube ERROR_DOWNLOAD items (with metadata) back to DISCOVERED
so 02_download_audio.py will retry them.

Usage:
  1. python scripts/rerun_youtube_error_downloads.py [--dry-run] [--limit N]
  2. python scripts/02_download_audio.py  # run download (repeat as needed)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB, QueueItem
from sqlalchemy import select, update, func
from sqlalchemy.orm import Session


def main():
    ap = argparse.ArgumentParser(
        description="Reset YouTube ERROR_DOWNLOAD items (with metadata) to DISCOVERED for retry"
    )
    ap.add_argument("--dry-run", action="store_true", help="Print count only, no DB changes")
    ap.add_argument("--limit", type=int, default=None, help="Max items to reset (default: all)")
    args = ap.parse_args()

    settings = Settings.load()
    db = DB(settings.db_url)

    with Session(db.engine) as session:
        # Count matching items
        from sqlalchemy import select, func
        count_stmt = (
            select(func.count(QueueItem.id))
            .where(
                QueueItem.platform == "youtube",
                QueueItem.status == "ERROR_DOWNLOAD",
                QueueItem.source_metadata.isnot(None),
            )
        )
        total = session.scalar(count_stmt) or 0

        if total == 0:
            print("No YouTube ERROR_DOWNLOAD items with metadata found.")
            return

        print(f"Found {total} YouTube ERROR_DOWNLOAD items with metadata.")

        if args.dry_run:
            print("Dry-run: no changes made. Run without --dry-run to reset.")
            return

        stmt = (
            update(QueueItem)
            .where(
                QueueItem.platform == "youtube",
                QueueItem.status == "ERROR_DOWNLOAD",
                QueueItem.source_metadata.isnot(None),
            )
            .values(
                status="DISCOVERED",
                audio_path=None,
                duration_seconds=None,
                error_msg=None,
                last_update_at=datetime.utcnow(),
            )
        )
        if args.limit is not None:
            ids_stmt = (
                select(QueueItem.id)
                .where(
                    QueueItem.platform == "youtube",
                    QueueItem.status == "ERROR_DOWNLOAD",
                    QueueItem.source_metadata.isnot(None),
                )
                .order_by(QueueItem.id.asc())
                .limit(args.limit)
            )
            ids = [r[0] for r in session.execute(ids_stmt).fetchall()]
            if not ids:
                print("No items to reset.")
                return
            stmt = stmt.where(QueueItem.id.in_(ids))

        result = session.execute(stmt)
        session.commit()
        n = result.rowcount

    print(f"Reset {n} items to DISCOVERED.")
    print("Run: python scripts/02_download_audio.py")


if __name__ == "__main__":
    main()
