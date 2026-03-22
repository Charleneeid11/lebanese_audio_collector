#!/usr/bin/env python3
"""
Backfill source_metadata for existing YouTube items using yt-dlp.
Does NOT touch audio or transcripts. Metadata only.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from yt_dlp import YoutubeDL

from src.cfg import Settings
from src.db import DB


def fetch_video_metadata(url: str) -> dict | None:
    """Extract channel_id, uploader, channel_country from video URL. Returns None on failure."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None
            meta = {
                "channel_id": info.get("channel_id") or info.get("uploader_id"),
                "uploader": info.get("uploader") or info.get("channel"),
            }
            if info.get("channel"):
                meta["channel"] = info["channel"]
            if info.get("location"):
                meta["channel_country"] = info["location"]
            elif info.get("channel_country"):
                meta["channel_country"] = info["channel_country"]
            return meta if meta.get("channel_id") or meta.get("uploader") else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Backfill YouTube source_metadata")
    ap.add_argument("--limit", type=int, default=None, help="Max items to process")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    ap.add_argument("--dry-run", action="store_true", help="Print only, no DB updates")
    args = ap.parse_args()

    settings = Settings.load()
    db = DB(settings.db_url)

    items = db.fetch_youtube_needing_backfill(limit=args.limit)
    if not items:
        print("No YouTube items needing backfill.")
        return

    print(f"Processing {len(items)} YouTube items...")

    ok, fail = 0, 0
    for i, item in enumerate(items):
        meta = fetch_video_metadata(item.url)
        if meta:
            if not args.dry_run:
                db.update_source_metadata(item.id, meta)
            ok += 1
            ch = meta.get("channel_id") or "?"
            up = (meta.get("uploader") or "?")[:40]
            print(f"  [{item.id}] {ch} | {up}")
        else:
            fail += 1
            print(f"  [{item.id}] FAIL: {item.url[:60]}...")

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(items)} done (ok={ok}, fail={fail})")

        time.sleep(args.delay)

    print(f"\nDone. OK: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
