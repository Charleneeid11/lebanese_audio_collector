#!/usr/bin/env python3
"""
Assign metadata-based weak labels to SCREENED items.
Uses trusted Lebanese / non-Lebanese source lists from config.
Run after 03_transcribe_screening.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB


def _normalize_feed(url: str | None) -> str:
    if not url:
        return ""
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
    return base.lower()


def _get_trusted_lists(settings: Settings):
    """Merge weak_labels config with platforms (fallback when empty)."""
    wl = settings.weak_labels
    plat = settings.platforms

    lb_channels = wl.trusted_lebanese_channels or plat.youtube.channels
    lb_feeds = wl.trusted_lebanese_feeds or plat.podcasts.rss_feeds
    lb_tiktok = wl.trusted_lebanese_tiktok_users or plat.tiktok.users

    lb_feed_set = {_normalize_feed(f) for f in lb_feeds}
    nlb_feed_set = {_normalize_feed(f) for f in wl.trusted_non_lebanese_feeds}
    nlb_channel_set = set(wl.trusted_non_lebanese_channels)
    nlb_tiktok_set = set(wl.trusted_non_lebanese_tiktok_users)

    return {
        "lb_channels": set(lb_channels),
        "lb_feeds": lb_feed_set,
        "lb_tiktok": set(lb_tiktok),
        "nlb_channels": nlb_channel_set,
        "nlb_feeds": nlb_feed_set,
        "nlb_tiktok": nlb_tiktok_set,
    }


def _classify(meta: dict | None, trusted: dict) -> str | None:
    """Return 'WEAK_POSITIVE', 'WEAK_NEGATIVE', or None."""
    if not meta:
        return None

    ch = meta.get("channel_id")
    if ch:
        if ch in trusted["lb_channels"]:
            return "WEAK_POSITIVE"
        if ch in trusted["nlb_channels"]:
            return "WEAK_NEGATIVE"

    feed = _normalize_feed(meta.get("feed_url"))
    if feed:
        if feed in trusted["lb_feeds"]:
            return "WEAK_POSITIVE"
        if feed in trusted["nlb_feeds"]:
            return "WEAK_NEGATIVE"

    user = meta.get("user")
    if user:
        if user in trusted["lb_tiktok"]:
            return "WEAK_POSITIVE"
        if user in trusted["nlb_tiktok"]:
            return "WEAK_NEGATIVE"

    return None


def main():
    settings = Settings.load()
    db = DB(settings.db_url)
    trusted = _get_trusted_lists(settings)

    print("========================================")
    print("WEAK LABEL ASSIGNMENT — STARTING")
    print("========================================")
    print(f"Trusted Lebanese: {len(trusted['lb_channels'])} channels, {len(trusted['lb_feeds'])} feeds, {len(trusted['lb_tiktok'])} tiktok")
    print(f"Trusted non-Lebanese: {len(trusted['nlb_channels'])} channels, {len(trusted['nlb_feeds'])} feeds, {len(trusted['nlb_tiktok'])} tiktok")
    print("Processing SCREENED items WITH metadata (single pass)...")
    print("(Items without metadata are never fetched; run backfill scripts first)")
    print("(Skipped items → SCREENED_NO_LABEL; never re-processed)")
    print("----------------------------------------")
    sys.stdout.flush()

    items = db.fetch_screened_with_metadata(limit=None)
    if not items:
        print("No SCREENED items with metadata. Done.")
    else:
        pos, neg, skip = 0, 0, 0
        for item in items:
            label = _classify(item.source_metadata, trusted)

            if label == "WEAK_POSITIVE":
                db.update_status(item.id, "WEAK_POSITIVE")
                pos += 1
            elif label == "WEAK_NEGATIVE":
                db.update_status(item.id, "WEAK_NEGATIVE")
                neg += 1
            else:
                db.update_status(item.id, "SCREENED_NO_LABEL")
                skip += 1

        print(f"Processed {len(items)} items → pos={pos}, neg={neg}, skip={skip}")
        print("========================================")
        print("WEAK LABELING COMPLETE")
        print("========================================")
        print(f"WEAK_POSITIVE : {pos}")
        print(f"WEAK_NEGATIVE : {neg}")
        print(f"SKIPPED       : {skip}")
        print("========================================")


if __name__ == "__main__":
    main()