#!/usr/bin/env python3
import sys, subprocess, json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB

MAX_RESULTS = 40

def search(query: str) -> list[str]:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        f"ytsearch{MAX_RESULTS}:{query}",
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout:
        return []

    data = json.loads(p.stdout)
    urls = []

    for e in data.get("entries", []):
        u = e.get("url")
        if u and "instagram.com/reel" in u:
            urls.append(u)

    return urls

def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    ig = settings.platforms.instagram
    found = set()

    for u in ig.users or []:
        found.update(search(f"site:instagram.com/reel @{u}"))

    inserted = 0
    for url in found:
        if db.insert_discovered("instagram", url):
            inserted += 1

    print(f"Discovered {len(found)} reels")
    print(f"Inserted {inserted}")

if __name__ == "__main__":
    main()
