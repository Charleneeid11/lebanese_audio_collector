# src/platforms/instagram.py
from typing import List, Dict
from src.cfg import Settings


def discover_candidates(settings: Settings) -> List[Dict]:
    """
    Discover Instagram videos/Reels likely to be Lebanese (stub).

    Later: use Meta Graph API with hashtags/accounts from:
    settings.platforms.instagram.hashtags / accounts
    and build:
      {"url": "...", "platform": "instagram", "meta": {...}}
    """
    cfg = settings.platforms.instagram
    # TODO: implement Instagram API-based discovery.
    return []
