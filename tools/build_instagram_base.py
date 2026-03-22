#!/usr/bin/env python3
import sys, yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

BASE_YAML = Path(__file__).resolve().parents[1] / "config" / "base.yaml"

HASHTAG_SEEDS = [
    "لبنان", "لبناني", "لبنانية",
    "lebanon", "lebanese", "beirut",
    "لهجة_لبنانية", "كلام_لبناني",
    "lebanese_reels", "beirutlife"
]

USER_SEEDS = [
    "lebanesevoice",
    "lebanontalks",
    "beirutlife",
    "lebanesecontent"
]

STOPWORDS = {"reels", "explore", "viral", "love", "instagood"}

def norm(x: str) -> str:
    return x.lower().replace("#", "").replace("@", "").strip()

def useless(x: str) -> bool:
    return len(x) < 4 or x in STOPWORDS or x.isdigit()

def main():
    cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))

    ig = cfg.setdefault("platforms", {}).setdefault("instagram", {})
    hashtags = set(map(norm, ig.get("hashtags") or []))
    users = set(map(norm, ig.get("users") or []))

    for h in HASHTAG_SEEDS:
        h = norm(h)
        if not useless(h):
            hashtags.add(h)

    for u in USER_SEEDS:
        u = norm(u)
        if not useless(u):
            users.add(u)

    users -= hashtags  # cross-dedupe

    ig["hashtags"] = sorted(hashtags)
    ig["users"] = sorted(users)

    BASE_YAML.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    print(f"Saved {len(hashtags)} hashtags and {len(users)} users")

if __name__ == "__main__":
    main()
