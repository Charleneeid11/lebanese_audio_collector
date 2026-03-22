#!/usr/bin/env python3
"""
02_download_audio.py
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB
from src.platforms.youtube import download_audio as yt_download_audio
from src.platforms.podcast import download_audio as podcast_download_audio
from src.platforms.tiktok import download_audio as tt_download_audio


# BATCH_SIZE = 600
# PLATFORM = "podcast_rss"

BATCH_SIZE = 600
PLATFORM = "youtube"


def is_buzzsprout(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "buzzsprout.com" in host

def is_youtube_short(url: str) -> bool:
    return "/shorts/" in url


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    raw_dir = settings.audio.raw_dir
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    items = db.fetch_queue_by_platform(
        status="DISCOVERED",
        platform=PLATFORM,
        limit=BATCH_SIZE,
    )

    if not items:
        print(f"No DISCOVERED items for platform={PLATFORM}")
        return

    print(f"Processing {len(items)} items for {PLATFORM}")

    for item in items:
        print(f"[{item.id}] Downloading {item.platform} -> {item.url}")

        # ---------- HARD SKIP BUZZSPROUT ----------
        if is_buzzsprout(item.url):
            db.update_status(
                item.id,
                "REJECTED",
                error_msg="host-blocked (buzzsprout)",
            )
            print("  -> Skipped (Buzzsprout host blocked)")
            continue

        # ---------- SKIP YOUTUBE SHORTS ----------
        if PLATFORM == "youtube" and is_youtube_short(item.url):
            db.update_status(
                item.id,
                "REJECTED",
                error_msg="youtube_short_skipped",
            )
            print("  -> Skipped (YouTube Shorts)")
            continue
        try:
            # ---------- YOUTUBE ----------
            if item.platform == "youtube":
                audio_path, duration = yt_download_audio(
                    item.url,
                    raw_dir,
                    max_download_seconds=settings.audio.max_download_seconds,
                )

            # ---------- PODCAST ----------
            elif item.platform in ("podcast", "podcast_rss"):
                audio_path, duration = podcast_download_audio(
                    item.url,
                    raw_dir,
                    max_download_seconds=settings.audio.max_download_seconds,
                )

            # ---------- TIKTOK ----------
            elif item.platform == "tiktok":
                audio_path, duration = tt_download_audio(
                    item.url,
                    raw_dir,
                    max_download_seconds=settings.audio.max_download_seconds,
                )

            else:
                db.update_status(
                    item.id,
                    "UNSUPPORTED_PLATFORM",
                    error_msg=f"Unsupported platform: {item.platform}",
                )
                print("  -> Unsupported platform")
                continue

            print(f"  -> Saved {audio_path} ({duration:.1f}s)")

            # ---------- LENGTH FILTER ----------
            if duration < settings.audio.min_seconds:
                db.update_status(item.id, "SKIPPED_LENGTH")
                print("  -> Skipped (too short)")
                continue

            db.mark_downloaded(item.id, audio_path, duration)
            print("  -> Marked DOWNLOADED")

        except Exception as e:
            db.update_status(item.id, "ERROR_DOWNLOAD", error_msg=str(e))
            print(f"  -> ERROR_DOWNLOAD: {e}")


if __name__ == "__main__":
    main()




# This script processes queued media items and downloads their audio files into the raw audio directory. 
# It fetches up to BATCH_SIZE items from the database queue filtered by status="DISCOVERED" and a specific PLATFORM. 
# For each item, it applies hard filtering rules before download: Buzzsprout-hosted URLs are rejected immediately, and YouTube Shorts (/shorts/) 
# are skipped when processing YouTube content. Depending on the platform (youtube, podcast/podcast_rss, or tiktok), it calls the corresponding 
# platform-specific download_audio function, passing the configured maximum download duration. If the platform is unsupported, the item is marked accordingly. 
# After download, the script enforces a minimum duration threshold (settings.audio.min_seconds); clips that are too short are marked as skipped.
# Valid downloads are recorded in the database via mark_downloaded, storing the local file path and duration. Any exceptions during download result in the 
# item being marked with ERROR_DOWNLOAD. Overall, this script transitions items from the discovery stage to locally stored audio ready for further processing, 
# while enforcing host, format, and length constraints.