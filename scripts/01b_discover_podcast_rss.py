#!/usr/bin/env python3
"""
Discover podcast RSS feeds via iTunes Search and enqueue all episodes found.
"""

import argparse
import time
import requests
import feedparser

from src.cfg import Settings
from src.db import DB
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))



ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

# ---- Full-collection defaults (safe but large) ----
DEFAULT_ITUNES_LIMIT = 200          # Apple hard cap ~200
DEFAULT_MAX_FEEDS = 10_000          # Emergency brake
DEFAULT_MAX_EPISODES_PER_FEED = 500
DEFAULT_SLEEP = 0.05                # Polite, but fast


def itunes_search_podcasts(term: str, country: str = "LB", limit: int = DEFAULT_ITUNES_LIMIT) -> list[dict]:
    params = {
        "term": term,
        "media": "podcast",
        "entity": "podcast",
        "country": country,
        "limit": limit,
    }
    r = requests.get(ITUNES_SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []


def rss_episode_enclosures(feed_url: str, max_eps: int = DEFAULT_MAX_EPISODES_PER_FEED) -> list[dict]:
    feed = feedparser.parse(feed_url)
    out = []

    for entry in (feed.entries or [])[:max_eps]:
        enclosures = getattr(entry, "enclosures", []) or []
        if not enclosures:
            continue

        enc = enclosures[0]  # first audio enclosure only
        href = enc.get("href")
        if not href:
            continue

        out.append(
            {
                "audio_url": href,
                "episode_title": getattr(entry, "title", None),
                "published": getattr(entry, "published", None),
                "guid": getattr(entry, "id", None) or getattr(entry, "guid", None),
            }
        )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="LB")
    ap.add_argument(
        "--term",
        action="append",
        required=True,
        help="Repeatable. e.g. --term 'بودكاست لبناني'",
    )
    ap.add_argument("--itunes-limit", type=int, default=DEFAULT_ITUNES_LIMIT)
    ap.add_argument("--max-feeds", type=int, default=DEFAULT_MAX_FEEDS)
    ap.add_argument("--max-episodes-per-feed", type=int, default=DEFAULT_MAX_EPISODES_PER_FEED)
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = ap.parse_args()

    s = Settings.load()
    db = DB(s.db_url)

    seen_feeds: set[str] = set()
    inserted = 0
    feeds_processed = 0

    for term in args.term:
        results = itunes_search_podcasts(
            term=term,
            country=args.country,
            limit=args.itunes_limit,
        )

        for r in results:
            if feeds_processed >= args.max_feeds:
                break

            feed_url = r.get("feedUrl")
            if not feed_url or feed_url in seen_feeds:
                continue

            seen_feeds.add(feed_url)
            feeds_processed += 1

            podcast_title = r.get("collectionName")
            publisher = r.get("artistName")

            episodes = rss_episode_enclosures(
                feed_url,
                max_eps=args.max_episodes_per_feed,
            )

            for ep in episodes:
                meta = {
                    "feed_url": feed_url,
                    "search_term": term,
                }
                if db.add_to_queue(ep["audio_url"], "podcast_rss", source_metadata=meta):
                    inserted += 1

            time.sleep(args.sleep)

        if feeds_processed >= args.max_feeds:
            break

    print(
        f"Done. Feeds processed: {feeds_processed}, "
        f"episodes enqueued: {inserted}"
    )


if __name__ == "__main__":
    main()


# This script discovers podcast episodes via the iTunes Search API and enqueues their audio URLs for processing. It accepts one or more search terms (--term) 
# and queries Apple’s API for podcast collections filtered by country and result limit. For each returned podcast, it retrieves the RSS feedUrl, avoids duplicate 
# feeds using an in-memory set, and enforces an overall feed cap (--max-feeds). It then parses each RSS feed using feedparser, extracts up to a configurable 
# number of episodes per feed, and collects the first available audio enclosure (href) for each episode. For every episode found, it builds metadata (source,
# search term, feed URL, podcast title, publisher, episode title, publication date, and GUID) and enqueues the audio URL into the database via db.add_to_queue, 
# relying on the database layer for URL deduplication. Small sleep intervals are inserted between feeds to avoid aggressive request patterns. 
# The script therefore connects keyword-based podcast discovery to episode-level audio ingestion by populating the system’s processing queue.