#01_seed_queue.py
#Seed the queue with a few YouTube URLs.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
#!/usr/bin/env python3
from src.cfg import Settings
from src.db import DB

# For now: manually put a few YouTube URLs here.
SEED_URLS = [
    # replace these with real Lebanese videos you know
    ("https://www.youtube.com/watch?v=7lRJLKb8H3Q", "youtube"),
]

def main():
    s = Settings.load()
    db = DB(s.db_url)

    for url, platform in SEED_URLS:
        db.add_to_queue(url, platform)

    print("Done seeding queue.")

if __name__ == "__main__":
    main()
