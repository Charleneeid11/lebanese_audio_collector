# Lebanese Audio Collector — Pipeline

## Active platforms

| Platform      | Discovery script              | Download support | Status   |
|---------------|-------------------------------|------------------|----------|
| YouTube       | 01a, 01b                      | ✓                | Active   |
| Podcast (RSS) | 01b, 01c                      | ✓                | Active   |
| TikTok        | 01d                           | ✓                | Active   |
| Instagram     | 01e                           | —                | Disabled |
| Facebook      | 01f                           | —                | Disabled |

Instagram and Facebook are disabled in `config/base.yaml` (empty lists). Scripts and source files are kept for future use.

---

## Pipeline steps

| Step | Script                    | Input status   | Output status      |
|------|---------------------------|----------------|---------------------|
| 0    | `00_show_queue.py`        | —              | Queue summary       |
| 1    | 01a/b/c/d (discover)      | —              | DISCOVERED          |
| 2    | `02_download_audio.py`   | DISCOVERED     | DOWNLOADED          |
| 3    | `03_transcribe_screening.py` | DOWNLOADED | SCREENED            |
| 3b   | `03b_assign_weak_labels.py` | SCREENED     | WEAK_POSITIVE / WEAK_NEGATIVE |
| 4    | `04_score_dialect.py`    | SCREENED       | POTENTIAL_LB / BORDERLINE_LB / REJECTED |
| 5    | `05_train_dialect_model.py` | WEAK_POSITIVE, WEAK_NEGATIVE (fallback: POTENTIAL_LB, REJECTED) | Model file |

---

## Backfill (existing items without metadata)

**YouTube** (fetches metadata from video URL via yt-dlp):
```bash
python scripts/backfill_youtube_metadata.py --limit 100   # test batch
python scripts/backfill_youtube_metadata.py               # full run (~6k items)
```
Options: `--delay 0.5` (rate limit), `--dry-run` (no DB updates).

**Podcast** (re-parses RSS feeds, matches DB.url to enclosure URLs):
```bash
python scripts/backfill_podcast_metadata.py --dry-run     # test
python scripts/backfill_podcast_metadata.py --limit 100  # small batch
python scripts/backfill_podcast_metadata.py               # full run (~28k items)
```
Options: `--limit` (max DB items), `--dry-run` (no DB updates).

---

## Typical run order

```bash
# 1. Discover (run as needed)
python scripts/01a_discover_youtube.py
python scripts/01b_discover_youtube_rss.py
python scripts/01b_discover_podcast_rss.py   # or 01c
python scripts/01d_discover_tiktok.py

# 2. Download (set PLATFORM in script: youtube, podcast, podcast_rss, tiktok)
python scripts/02_download_audio.py

# 3. Transcribe screening chunks
python scripts/03_transcribe_screening.py

# 3b. Assign metadata-based weak labels (run before first training)
python scripts/03b_assign_weak_labels.py

# 4. Score dialect (requires trained model from step 5 first run)
python scripts/04_score_dialect.py

# 5. Train model (uses WEAK_POSITIVE + WEAK_NEGATIVE; fallback: POTENTIAL_LB, REJECTED)
python scripts/05_train_dialect_model.py
```

---

## Current DB state (as of last run)

After running up to step 03:

- **DISCOVERED**: items waiting for download
- **DOWNLOADED**: 0 (all processed by step 3)
- **SCREENED**: ready for step 04 (dialect scoring)
- **POTENTIAL_LB / BORDERLINE_LB / REJECTED**: from previous 04 runs (if any)
