#!/usr/bin/env python3
"""
Reset podcast items that were incorrectly marked as DOWNLOADED
back to DISCOVERED, without violating NOT NULL constraints.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB, QueueItem
from sqlalchemy import update
from sqlalchemy.orm import Session


PLATFORMS = {"podcast", "podcast_rss"}


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    engine = db.engine

    with Session(engine) as session:
        stmt = (
            update(QueueItem)
            .where(
                QueueItem.platform.in_(PLATFORMS),
                QueueItem.status == "DOWNLOADED",
            )
            .values(
                status="DISCOVERED",
                audio_path=None,
                duration_seconds=None,
                error_msg=None,
                last_update_at=datetime.utcnow(),  # 🔑 FIX
            )
        )

        result = session.execute(stmt)
        session.commit()

    print(f"Reset {result.rowcount} podcast items back to DISCOVERED")


if __name__ == "__main__":
    main()
