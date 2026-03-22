#!/usr/bin/env python3
"""
Discover Facebook videos from public pages (page-based, not search)
"""

import sys, subprocess, json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB

MAX_PER_PAGE = 50

def extract_from_page(page: str) -> list[str]:
    url = f"https://www.facebook.com/{page}/videos"

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-end", str(MAX_PER_PAGE),
        url,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout:
        return []

    try:
        data = json.loads(p.stdout)
    except Exception:
        return []

    urls = []
    for e in data.get("entries", []):
        u = e.get("url")
        if u and "facebook.com" in u:
            urls.append(u)

    return urls

def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    fb = settings.platforms.facebook
    discovered = set()

    for page in fb.pages or []:
        urls = extract_from_page(page)
        discovered.update(urls)
        print(f"[{page}] found {len(urls)} videos")

    inserted = 0
    for url in discovered:
        if db.insert_discovered("facebook", url):
            inserted += 1

    print(f"Discovered {len(discovered)} Facebook videos")
    print(f"Inserted {inserted} new items")

if __name__ == "__main__":
    main()
