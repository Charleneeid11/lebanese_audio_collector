#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB
from src.platforms.youtube_rss import discover_candidates


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    results = discover_candidates(settings)
    print(f"Discovered {len(results)} YouTube RSS candidates.")

    inserted = 0
    for item in results:
        meta = item.get("meta", {})
        if db.add_to_queue(
            item["url"],
            item["platform"],
            source_metadata={"channel_id": meta.get("channel_id")},
        ):
            inserted += 1

    print("Done inserting RSS results.")


if __name__ == "__main__":
    main()




# This script performs YouTube discovery using the RSS-based method and inserts the discovered videos into the database queue. 
# It loads configuration via Settings, initializes a database connection, and calls the RSS-based discover_candidates function to retrieve video
# entries from configured channel feeds. It prints how many RSS candidates were found, then iterates through them and adds each video’s URL 
# and platform to the queue using db.add_to_queue. A queue fetch call (db.fetch_queue(status="DISCOVERED", limit=1)) is executed inside the 
# loop but does not affect insertion logic. The script therefore links RSS-based YouTube discovery to the system’s queue, preparing those videos for later processing.