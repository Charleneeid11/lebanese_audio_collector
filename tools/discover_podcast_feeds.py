#!/usr/bin/env python3
"""
Discover Podcast RSS feeds via PodcastIndex search/byterm and persist them into config/base.yaml.
"""

import sys
import os
import time
import yaml
import hashlib
import requests
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

load_dotenv()

BASE_YAML = Path(__file__).resolve().parents[1] / "config" / "base.yaml"

API_KEY = (os.getenv("PODCASTINDEX_API_KEY") or "").strip()
API_SECRET = (os.getenv("PODCASTINDEX_API_SECRET") or "").strip()

API_URL = "https://api.podcastindex.org/api/1.0/search/byterm"

# Full-collection settings
PAGE_SIZE = 100           # PodcastIndex supports up to 100 for many endpoints
MAX_PAGES_PER_KEYWORD = None  # Set to an int (e.g., 50) if you want an emergency brake
REQUEST_TIMEOUT = 30
RETRY_COUNT = 3
RETRY_SLEEP_SECONDS = 1.5


def auth_headers() -> dict:
    """
    PodcastIndex expects:
    Authorization = sha1(apiKey + apiSecret + unixTimeSeconds)
    """
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "Missing PODCASTINDEX_API_KEY or PODCASTINDEX_API_SECRET in environment (.env)."
        )

    ts = str(int(time.time()))  # epoch seconds (UTC)
    auth_string = API_KEY + API_SECRET + ts
    auth = hashlib.sha1(auth_string.encode("utf-8")).hexdigest()

    return {
        "User-Agent": "lebanese-audio-collector/0.1 (contact: you@domain.com)",
        "X-Auth-Date": ts,
        "X-Auth-Key": API_KEY,
        "Authorization": auth,
    }


def request_with_retries(url: str, *, headers: dict, params: dict, timeout: int) -> dict:
    last_err = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)  # simple backoff
            else:
                raise last_err


def is_relevant_feed(p: dict) -> bool:
    lang = (p.get("language") or "").lower()
    title = (p.get("title") or "").lower()
    return (
        ("ar" in lang)
        or ("leban" in title)
        or ("لبنان" in title)
    )


def main():
    if not API_KEY or not API_SECRET:
        print("ERROR: Missing API credentials!")
        print("Please set PODCASTINDEX_API_KEY and PODCASTINDEX_API_SECRET in .env file")
        return

    cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")) or {}
    podcasts_cfg = cfg.setdefault("platforms", {}).setdefault("podcasts", {})

    keywords = podcasts_cfg.get("keywords", []) or []
    feeds = set(podcasts_cfg.get("rss_feeds", []) or [])

    total_added = 0

    for kw in keywords:
        offset = 0
        page = 0
        while True:
            page += 1
            if MAX_PAGES_PER_KEYWORD is not None and page > MAX_PAGES_PER_KEYWORD:
                break

            params = {"q": kw, "max": PAGE_SIZE, "offset": offset}

            data = request_with_retries(
                API_URL,
                headers=auth_headers(),
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            batch = data.get("feeds", []) or []
            if not batch:
                break

            before = len(feeds)

            for p in batch:
                if not is_relevant_feed(p):
                    continue
                url = p.get("url")
                if url:
                    feeds.add(url)

            after = len(feeds)
            total_added += max(0, after - before)

            # Next page
            if len(batch) < PAGE_SIZE:
                # No more results
                break
            offset += PAGE_SIZE

            # tiny pause to be polite
            time.sleep(0.05)

    podcasts_cfg["rss_feeds"] = sorted(feeds)

    BASE_YAML.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Saved {len(feeds)} podcast RSS feeds to base.yaml (added {total_added} new)")


if __name__ == "__main__":
    main()




# This script discovers podcast RSS feeds using the PodcastIndex API and persists them into config/base.yaml. It authenticates requests by generating a SHA-1 
# hash of apiKey + apiSecret + currentUnixTimestamp, as required by PodcastIndex, and sends the necessary headers (X-Auth-Date, X-Auth-Key, Authorization). 
# For each keyword defined under platforms.podcasts.keywords in base.yaml, it paginates through search results (PAGE_SIZE batches with offset-based pagination),
# using a retry mechanism with simple exponential backoff for robustness. Each returned feed is filtered through is_relevant_feed, which keeps feeds if their 
# language suggests Arabic or their title references Lebanon (English or Arabic). Valid RSS URLs are added to a deduplicated set, and once all keywords are processed, 
# the updated sorted feed list is written back into base.yaml under rss_feeds. The script therefore expands and maintains the project’s podcast source pool based 
# on configurable search terms while ensuring authentication, filtering, deduplication, and persistence.