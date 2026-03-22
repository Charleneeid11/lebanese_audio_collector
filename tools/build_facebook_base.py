#!/usr/bin/env python3
"""
Populate platforms.facebook.pages and platforms.facebook.keywords in base.yaml
"""

import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

BASE_YAML = Path(__file__).resolve().parents[1] / "config" / "base.yaml"

# --- Seed pages (public Lebanese media / creators) ---
PAGE_SEEDS = [
    "mtvlebanon",
    "lbci",
    "aljadeedonline",
    "lebanondebate",
    "sawtbeirut",
    "thisislebanonnews",
    "lebanonfiles",
]

# --- Keyword seeds ---
KEYWORD_SEEDS = [
    "لبنان",
    "لبناني",
    "لبنانية",
    "لهجة لبنانية",
    "كلام لبناني",
    "بيروت",
    "طرابلس",
    "صيدا",
    "Lebanon",
    "Lebanese",
    "Beirut",
]

STOPWORDS = {
    "video", "videos", "news", "official", "page",
    "live", "tv", "channel"
}

def normalize(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace(" ", "")
        .replace("@", "")
        .replace("#", "")
    )

def is_useless(token: str) -> bool:
    return (
        not token
        or len(token) < 3
        or token.isdigit()
        or token in STOPWORDS
    )

def main():
    cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))

    fb = cfg.setdefault("platforms", {}).setdefault("facebook", {})

    pages = set(map(normalize, fb.get("pages") or []))
    keywords = set(map(normalize, fb.get("keywords") or []))

    # --- Pages ---
    for p in PAGE_SEEDS:
        p_norm = normalize(p)
        if not is_useless(p_norm):
            pages.add(p_norm)

    # --- Keywords ---
    for k in KEYWORD_SEEDS:
        k_norm = normalize(k)
        if not is_useless(k_norm):
            keywords.add(k_norm)

    fb["pages"] = sorted(pages)
    fb["keywords"] = sorted(keywords)

    BASE_YAML.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Saved {len(pages)} pages and {len(keywords)} keywords to base.yaml")

if __name__ == "__main__":
    main()
