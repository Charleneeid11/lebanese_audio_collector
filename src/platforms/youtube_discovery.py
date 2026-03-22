from typing import List, Dict
from googleapiclient.discovery import build
from src.cfg import Settings


def _init_yt(settings: Settings):
    cfg = settings.platforms.youtube
    if not cfg.api_key:
        raise ValueError("YouTube API key missing in config.")
    return build("youtube", "v3", developerKey=cfg.api_key)


def search_by_keywords(youtube, settings: Settings) -> List[Dict]:
    cfg = settings.platforms.youtube
    results = []

    for query in cfg.search_queries:
        req = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=20,
            relevanceLanguage=cfg.relevance_language,
            regionCode=cfg.region_code,
            videoDuration="medium",   # avoid 30-second trash, avoid 3-hour long videos
        )
        res = req.execute()

        for item in res.get("items", []):
            vid = item["id"]["videoId"]
            results.append({
                "url": f"https://www.youtube.com/watch?v={vid}",
                "platform": "youtube",
                "meta": {
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelId"],
                    "query": query,
                }
            })

    return results


def search_channel_uploads(youtube, channel_id: str) -> List[Dict]:
    # Step 1: get the channel uploads playlist
    ch_req = youtube.channels().list(
        part="contentDetails",
        id=channel_id,
        maxResults=1
    )
    ch_res = ch_req.execute()

    if not ch_res.get("items"):
        return []

    uploads_playlist = ch_res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Step 2: fetch recent uploads
    pl_req = youtube.playlistItems().list(
        playlistId=uploads_playlist,
        part="snippet",
        maxResults=20
    )
    pl_res = pl_req.execute()

    results = []

    for item in pl_res.get("items", []):
        snippet = item["snippet"]
        vid = snippet["resourceId"]["videoId"]
        results.append({
            "url": f"https://www.youtube.com/watch?v={vid}",
            "platform": "youtube",
            "meta": {
                "title": snippet["title"],
                "channel": channel_id,
                "type": "channel_upload"
            }
        })

    return results


def discover_candidates(settings: Settings) -> List[Dict]:
    youtube = _init_yt(settings)
    cfg = settings.platforms.youtube

    final_results: List[Dict] = []

    # 1. Keyword-based search
    final_results.extend(search_by_keywords(youtube, settings))

    # 2. Channel-based discovery
    for channel_id in cfg.channels:
        final_results.extend(search_channel_uploads(youtube, channel_id))

    # Deduplicate by URL
    seen = set()
    unique_results = []
    for item in final_results:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique_results.append(item)

    return unique_results




# This script performs YouTube content discovery using the official YouTube Data API. It first initializes an authenticated API client via _init_yt,
# which reads the API key from configuration and raises an error if it is missing. Discovery happens through two strategies. The first, search_by_keywords,
# iterates over configured search queries and retrieves up to 20 medium-length videos per query, filtered by relevance language and region code,
# then returns structured results containing the video URL and metadata (title, channel ID, and originating query). The second, search_channel_uploads, 
# retrieves a channel’s uploads playlist via the channel’s contentDetails, then fetches up to 20 recent uploads from that playlist and returns similar 
# structured video entries. The main discover_candidates function combines results from both strategies and performs URL-based deduplication to ensure
# unique candidates before returning the final list.