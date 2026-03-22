#!/usr/bin/env python3

import sys
import feedparser
from pathlib import Path
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB


AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".ogg")


def is_audio_url(url: str | None) -> bool:
    return bool(url and url.lower().endswith(AUDIO_EXTS))


def extract_audio_url(entry) -> str | None:
    # Standard podcast enclosure
    for enc in entry.get("enclosures", []):
        url = enc.get("url")
        if is_audio_url(url):
            return url

    # Fallback: links
    for link in entry.get("links", []):
        url = link.get("href")
        if is_audio_url(url):
            return url

    return None


def normalize_source(feed_url: str) -> str:
    p = urlparse(feed_url)
    return f"{p.scheme}://{p.netloc}"


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    feeds = settings.platforms.podcasts.rss_feeds
    if not feeds:
        print("No podcast RSS feeds configured.")
        return

    discovered = 0

    for feed_url in feeds:
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            print(f"[WARN] Failed to parse feed: {feed_url}")
            continue

        source = normalize_source(feed_url)

        for entry in feed.entries:
            audio_url = extract_audio_url(entry)
            if not audio_url:
                continue

            if db.add_to_queue(
                url=audio_url,
                platform="podcast",
                source_metadata={"feed_url": feed_url},
            ):
                discovered += 1

    print(f"Discovered {discovered} podcast episodes")


if __name__ == "__main__":
    main()




# This script discovers podcast episodes directly from a predefined list of RSS feeds and inserts their audio URLs into the processing queue.
# It loads configured RSS feeds from settings.platforms.podcasts.rss_feeds and parses each feed using feedparser, skipping malformed feeds (feed.bozo). 
# For each entry, it attempts to extract an audio URL by first checking standard podcast enclosures and then falling back to link URLs, accepting only
# files with approved audio extensions (.mp3, .m4a, .wav, .ogg). If a valid audio URL is found, it is inserted into the database queue via db.add_to_queue 
# under the "podcast" platform label. The script counts and reports the number of discovered episodes. It therefore serves as a feed-driven episode ingestion 
# step, transforming configured RSS sources into queued audio items ready for downstream processing.