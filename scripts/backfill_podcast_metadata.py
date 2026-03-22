#!/usr/bin/env python3
"""
Backfill source_metadata for existing podcast items by re-parsing RSS feeds.
Matches DB.url to episode enclosure URLs. No re-download, no re-transcription.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import feedparser

from src.cfg import Settings
from src.db import DB


AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".ogg")


def is_audio_url(url: str | None) -> bool:
    return bool(url and url.lower().endswith(AUDIO_EXTS))


def extract_audio_url(entry) -> str | None:
    for enc in entry.get("enclosures", []):
        url = enc.get("url") or enc.get("href")
        if is_audio_url(url):
            return url
    for link in entry.get("links", []):
        url = link.get("href")
        if is_audio_url(url):
            return url
    return None


def extract_feed_meta(feed) -> dict:
    """Extract feed-level metadata (author, country, language)."""
    f = feed.get("feed", {})
    meta = {}
    if f.get("author"):
        meta["feed_author"] = f["author"]
    elif f.get("itunes_author"):
        meta["feed_author"] = f["itunes_author"]
    if f.get("language"):
        meta["feed_language"] = f["language"]
    if f.get("itunes_country"):
        meta["feed_country"] = f["itunes_country"]
    elif f.get("country"):
        meta["feed_country"] = f["country"]
    return meta


def build_audio_url_to_meta_map(feed_urls: list[str]) -> dict[str, dict]:
    """Parse feeds, return map: episode_audio_url -> {feed_url, feed_author, feed_country, feed_language}."""
    url_to_meta: dict[str, dict] = {}
    for feed_url in feed_urls:
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            continue
        feed_meta = extract_feed_meta(feed)
        base = {"feed_url": feed_url, **feed_meta}
        for entry in feed.get("entries", []):
            audio_url = extract_audio_url(entry)
            if audio_url:
                url_to_meta[audio_url] = base.copy()
        time.sleep(0.1)  # gentle rate limit
    return url_to_meta


def main():
    ap = argparse.ArgumentParser(description="Backfill podcast source_metadata from RSS feeds")
    ap.add_argument("--limit", type=int, default=None, help="Max DB items to process")
    ap.add_argument("--dry-run", action="store_true", help="Print only, no DB updates")
    args = ap.parse_args()

    settings = Settings.load()
    db = DB(settings.db_url)

    feeds = settings.platforms.podcasts.rss_feeds
    if not feeds:
        print("No RSS feeds in config.")
        return

    print(f"Parsing {len(feeds)} RSS feeds...")
    url_to_meta = build_audio_url_to_meta_map(feeds)
    print(f"Found {len(url_to_meta)} episode URLs in feeds.")

    items = db.fetch_podcast_needing_backfill(limit=args.limit)
    if not items:
        print("No podcast items needing backfill.")
        return

    print(f"Matching {len(items)} DB rows...")
    ok, fail = 0, 0
    for i, item in enumerate(items):
        meta = url_to_meta.get(item.url)
        if meta:
            if not args.dry_run:
                db.update_source_metadata(item.id, meta)
            ok += 1
            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{len(items)} (matched={ok})")
        else:
            fail += 1

    print(f"\nDone. Matched: {ok}, Unmatched: {fail}")
    if fail > 0:
        print("Unmatched URLs may be from iTunes search (different feed list) or removed feeds.")


if __name__ == "__main__":
    main()
