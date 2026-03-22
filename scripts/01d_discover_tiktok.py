#!/usr/bin/env python3
"""
Discover TikTok videos using hashtags and users
→ Insert video URLs into queue as DISCOVERED
"""

import sys
import subprocess
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB


MAX_RESULTS_PER_SOURCE = 50


def run_yt_dlp_flat(url: str) -> list[str]:
    """
    Run yt-dlp in flat-playlist mode on a TikTok page
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-end", str(MAX_RESULTS_PER_SOURCE),
        url,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0 or not result.stdout:
        return []

    try:
        data = json.loads(result.stdout)
    except Exception:
        return []

    urls = []
    for entry in data.get("entries", []):
        video_url = entry.get("url")
        if video_url and "tiktok.com" in video_url:
            urls.append(video_url)

    return urls


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    tiktok_cfg = settings.platforms.tiktok
    hashtags = tiktok_cfg.hashtags or []
    users = tiktok_cfg.users or []

    discovered: dict[str, dict] = {}  # url -> meta

    for tag in hashtags:
        tag_url = f"https://www.tiktok.com/tag/{tag}"
        for u in run_yt_dlp_flat(tag_url):
            discovered.setdefault(u, {})["hashtag"] = tag

    for user in users:
        user_url = f"https://www.tiktok.com/@{user}"
        for u in run_yt_dlp_flat(user_url):
            discovered.setdefault(u, {})["user"] = user

    inserted = 0
    for url, meta in discovered.items():
        if db.add_to_queue(url, "tiktok", source_metadata=meta or None):
            inserted += 1

    print(f"Discovered {len(discovered)} TikTok videos")
    print(f"Inserted {inserted} new queue items")


if __name__ == "__main__":
    main()
