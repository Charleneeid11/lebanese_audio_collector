#!/usr/bin/env python3
"""
Check YouTube items with metadata by pipeline stage (status).
Uses sqlite3 directly to avoid dependency issues.
"""
import sqlite3
from pathlib import Path

# DB path relative to project root
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "queue.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("YouTube items WITH metadata, by status:")
    print("-" * 45)
    cur.execute("""
        SELECT status, COUNT(*) as cnt 
        FROM queue 
        WHERE platform = 'youtube' AND source_metadata IS NOT NULL
        GROUP BY status ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    total = 0
    for status, cnt in rows:
        print(f"  {status:<28} : {cnt:>6}")
        total += cnt
    print("-" * 45)
    print(f"  {'TOTAL':<28} : {total:>6}")

    print("\nPipeline funnel (YouTube with metadata):")
    print("  ERROR_DOWNLOAD  → failed at step 02 (download)")
    print("  REJECTED        → passed 02,03,03b; rejected by 04 (model)")
    print("  WEAK_POSITIVE   → passed 02,03; labeled by 03b")
    print("  SCREENED        → passed 02,03; ready for 03b (none left)")

    print("\nYouTube items total (with or without metadata):")
    print("-" * 45)
    cur.execute("""
        SELECT status, COUNT(*) as cnt 
        FROM queue 
        WHERE platform = 'youtube'
        GROUP BY status ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    total_all = 0
    for status, cnt in rows:
        print(f"  {status:<28} : {cnt:>6}")
        total_all += cnt
    print("-" * 45)
    print(f"  {'TOTAL':<28} : {total_all:>6}")

    conn.close()

if __name__ == "__main__":
    main()
