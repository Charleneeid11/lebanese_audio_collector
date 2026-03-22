from typing import List, Dict
import feedparser

from src.cfg import Settings


def discover_candidates(settings: Settings) -> List[Dict]:
    """
    Discover YouTube videos using channel RSS feeds.
    Zero quota, safe to rerun.
    """
    yt_cfg = settings.platforms.youtube
    results: List[Dict] = []

    for channel_id in yt_cfg.channels:
        feed_url = (
            "https://www.youtube.com/feeds/videos.xml"
            f"?channel_id={channel_id}"
        )

        feed = feedparser.parse(feed_url)

        if feed.bozo:
            # malformed feed or network issue
            continue

        for entry in feed.entries:
            if not hasattr(entry, "link"):
                continue

            results.append({
                "url": entry.link,
                "platform": "youtube",
                "meta": {
                    "channel_id": channel_id,
                    "published": getattr(entry, "published", None),
                    "title": getattr(entry, "title", None),
                }
            })

    return results




# This script discovers YouTube video candidates using public channel RSS feeds instead of the YouTube Data API, avoiding quota limits
# and allowing safe repeated execution. The discover_candidates function reads configured channel IDs from settings.platforms.youtube.channels, 
# constructs the standard YouTube RSS feed URL for each channel, and parses it using feedparser. If a feed is malformed or a network issue occurs (feed.bozo), 
# it skips that channel. For each valid feed entry, it checks that a video link exists and then appends a structured dictionary containing the video URL,
# platform label ("youtube"), and basic metadata (channel ID, publication date, and title). The function returns a list of these candidate video objects for downstream processing.