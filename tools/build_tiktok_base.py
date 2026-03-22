#!/usr/bin/env python3
# Populate platforms.tiktok.hashtags and platforms.tiktok.users in base.yaml

import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

BASE_YAML = Path(__file__).resolve().parents[1] / "config" / "base.yaml"

# Seeds intentionally overlap — script will dedupe intelligently
HASHTAG_SEEDS = [
    "لبنان", "لبناني", "لبنانية", "Lebanon", "Lebanese",
    "لهجة لبنانية", "كلام لبناني", "lebanese dialect", "lebanese arabic",
    "بيروت", "طرابلس", "صيدا", "زحلة", "الجنوب", "الشمال",
    "حياة لبنانية", "ضحك لبناني", "كوميديا لبنانية",
    "مشاكل لبنان", "الوضع بلبنان",
]

USER_SEEDS = [
    # generic patterns — real accounts will be discovered later
    "lebanesecreator",
    "lebanesevoice",
    "lebanesecontent",
    "beirutlife",
    "lebanontalks",
]

STOPWORDS = {
    "video", "videos", "fyp", "foryou", "foryoupage",
    "trending", "viral", "explore", "love", "fun"
}

def normalize(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace(" ", "")
        .replace("#", "")
        .replace("@", "")
    )

def is_useless(token: str) -> bool:
    return (
        len(token) < 4 or
        token.isdigit() or
        token in STOPWORDS
    )

def main():
    cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))

    tiktok = cfg.setdefault("platforms", {}).setdefault("tiktok", {})
    hashtags = set(map(normalize, tiktok.get("hashtags") or []))
    users = set(map(normalize, tiktok.get("users") or []))

    # --- Process hashtags ---
    for h in HASHTAG_SEEDS:
        h_norm = normalize(h)
        if is_useless(h_norm):
            continue
        hashtags.add(h_norm)

    # --- Process users ---
    for u in USER_SEEDS:
        u_norm = normalize(u)
        if is_useless(u_norm):
            continue
        users.add(u_norm)

    # --- Cross-dedupe ---
    # If something appears in both, keep it as hashtag only
    users -= hashtags

    tiktok["hashtags"] = sorted(hashtags)
    tiktok["users"] = sorted(users)

    BASE_YAML.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    print(
        f"Saved {len(hashtags)} hashtags and {len(users)} users to base.yaml"
    )

if __name__ == "__main__":
    main()
