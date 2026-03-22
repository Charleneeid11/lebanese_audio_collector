# src/platforms/facebook.py
from typing import List, Dict
from src.cfg import Settings


def discover_candidates(settings: Settings) -> List[Dict]:
    """
    Discover Facebook videos likely to be Lebanese (stub).

    Later: use Meta Graph API with pages/keywords from:
    settings.platforms.facebook.pages / keywords
    and build:
      {"url": "...", "platform": "facebook", "meta": {...}}
    """
    cfg = settings.platforms.facebook
    # TODO: implement Facebook API-based discovery.
    return []
