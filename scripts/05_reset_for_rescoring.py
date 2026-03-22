#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import update
from src.cfg import Settings
from src.db import DB, QueueItem


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    with db.engine.begin() as conn:
        conn.execute(
            update(QueueItem)
            .where(
                QueueItem.status.in_(
                    ["REJECTED", "BORDERLINE_LB", "POTENTIAL_LB", "WEAK_POSITIVE", "WEAK_NEGATIVE"]
                )
            )
            .values(
                status="SCREENED",
                rejection_reason=None,
            )
        )

    print("Reset items back to SCREENED")


if __name__ == "__main__":
    main()
