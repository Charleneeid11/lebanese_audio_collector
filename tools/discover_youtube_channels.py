#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Dict

import yaml
from googleapiclient.discovery import build

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.cfg import Settings


BASE_YAML_PATH = ROOT / "config" / "base.yaml"


def main():
    # Load settings normally (for API key, region, language)
    settings = Settings.load()
    yt_cfg = settings.platforms.youtube

    if not yt_cfg.api_key:
        raise RuntimeError("YouTube API key missing in base.yaml")

    # Load raw YAML so we can write back safely
    with open(BASE_YAML_PATH, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    youtube_cfg = raw_cfg["platforms"]["youtube"]

    search_queries = youtube_cfg.get("search_queries", [])
    existing_channels = set(youtube_cfg.get("channels", []))

    youtube = build("youtube", "v3", developerKey=yt_cfg.api_key)

    discovered: Dict[str, str] = {}  # channel_id -> title

    for query in search_queries:
        req = youtube.search().list(
            q=query,
            part="snippet",
            type="channel",
            maxResults=10,
            relevanceLanguage=yt_cfg.relevance_language,
            regionCode=yt_cfg.region_code,
        )
        res = req.execute()

        for item in res.get("items", []):
            cid = item["id"]["channelId"]
            title = item["snippet"]["title"]

            if cid not in existing_channels:
                discovered[cid] = title

    if not discovered:
        print("No new channels discovered.")
        return

    # Merge channels
    updated_channels = list(existing_channels) + list(discovered.keys())
    youtube_cfg["channels"] = sorted(set(updated_channels))

    # Write back to base.yaml
    with open(BASE_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(
            raw_cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"Added {len(discovered)} new YouTube channels to base.yaml:\n")
    for cid, title in discovered.items():
        print(f"- {cid}  # {title}")


if __name__ == "__main__":
    main()




# This script discovers new YouTube channels using the YouTube Data API and persists them into config/base.yaml. It loads project 
# settings (to access the API key, region, and language filters) and reads the raw YAML configuration so it can safely update it. 
# For each query listed under platforms.youtube.search_queries, it performs a YouTube search restricted to type="channel", 
# applying the configured relevanceLanguage and regionCode. From the search results, it extracts channel IDs and titles, and keeps only 
# those not already present in the existing channels list. Newly discovered channel IDs are merged with the current set, deduplicated, 
# sorted, and written back to base.yaml. The script therefore expands the YouTube channel source list automatically based on configured 
# search terms while preventing duplicates and preserving the YAML structure.