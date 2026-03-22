#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cfg import Settings
from src.db import DB


STATUSES = [
    "DISCOVERED",
    "DOWNLOADED",
    "SCREENED",
    "WEAK_POSITIVE",
    "WEAK_NEGATIVE",
    "POTENTIAL_LB",
    "BORDERLINE_LB",
    "REJECTED",
    "ERROR_DOWNLOAD",
    "ERROR_SCREEN",
    "ERROR_SCORING",
]


def main():
    s = Settings.load()
    db = DB(s.db_url)

    status_counts = {}
    platform_counts = {}

    for status in STATUSES:
        items = db.fetch_queue(status=status)
        status_counts[status] = len(items)

        print(f"[{status}] {len(items)}")

        for item in items:
            # count platforms
            platform = item.platform or "UNKNOWN"
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

            # ---- noisy, keep commented unless debugging ----
            # print(
            #     f"{item.id} | {item.platform} | {item.url} | {item.discovered_at}"
            # )
            #
            # if item.error_msg:
            #     print(f"    error_msg: {item.error_msg}")
            #
            # if item.rejection_reason:
            #     print(f"    rejection_reason: {item.rejection_reason}")

    # --------------------
    # STATUS SUMMARY
    # --------------------
    print("\n" + "=" * 40)
    print("QUEUE STATUS SUMMARY")
    print("=" * 40)

    total = 0
    for status, count in status_counts.items():
        print(f"{status:<18} : {count}")
        total += count

    print("-" * 40)
    print(f"{'TOTAL':<18} : {total}")

    # --------------------
    # PLATFORM SUMMARY
    # --------------------
    print("\n" + "=" * 40)
    print("PLATFORM SUMMARY")
    print("=" * 40)

    for platform, count in sorted(platform_counts.items()):
        print(f"{platform:<18} : {count}")


if __name__ == "__main__":
    main()
