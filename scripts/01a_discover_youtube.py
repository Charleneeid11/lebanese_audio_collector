#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB
from src.platforms.youtube_discovery import discover_candidates


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    results = discover_candidates(settings)
    print(f"Discovered {len(results)} YouTube candidates.")

    for item in results:
        meta = item.get("meta", {})
        db.add_to_queue(
            item["url"],
            item["platform"],
            source_metadata={"channel_id": meta.get("channel"), "query": meta.get("query")},
        )

    print("Done inserting into queue.")


if __name__ == "__main__":
    main()
    
    
    
    
# This script runs the YouTube discovery pipeline and inserts the discovered videos into the system’s processing queue. 
# It loads configuration via Settings, initializes a database connection using the configured db_url, and calls discover_candidates 
# from the YouTube discovery module to retrieve structured video results. It prints the number of discovered candidates, then iterates 
# over them and inserts each item’s URL and platform into the database queue using db.add_to_queue. The script therefore connects API-based
# discovery with the database ingestion layer, preparing newly found YouTube videos for later download and processing.