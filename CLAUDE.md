# CLAUDE.md — Lebanese Audio Collector

## Meta — Instructions for Claude

- **Keep this file up to date.** Whenever a change is made to the project (new script added, status flow changed, config modified, incomplete feature completed, etc.), update the relevant section of this file in the same response. Do not wait to be asked.
- **Keep `FINDINGS.md` up to date.** This is the thesis research log. Whenever a new finding is made, a decision is taken, a script produces results, or the pipeline state changes, update the relevant section of `FINDINGS.md` in the same response. Add new subsections as needed. This includes: empirical results, parameter decisions with justifications, error analysis, model performance numbers, and any insight that belongs in the thesis paper. Do not wait to be asked.
- File references use `[file](path)` markdown links so they are clickable in the IDE.

---

## Project Purpose

This is a thesis project to build a dataset of **Lebanese Arabic audio** from open platforms (YouTube, podcasts, TikTok). The goal is to collect, filter, and classify spoken Lebanese dialect audio for linguistic/NLP research.

---

## Architecture Overview

Queue-driven, multi-stage pipeline. Items flow through a SQLite DB (`data/queue.db`) with explicit status transitions. Each pipeline stage is a standalone script that reads items at a given status and writes them to the next.

### Status Flow

```
DISCOVERED → DOWNLOADED → SCREENED ──► 03b (metadata match) ──► WEAK_POSITIVE / WEAK_NEGATIVE
                                   └──► 03c (lexical scoring) ──► WEAK_NEGATIVE
                                   └──► 04  (ML model)        ──► POTENTIAL_LB / BORDERLINE_LB / REJECTED
```

- **03b** labels items whose `source_metadata.channel_id` or `feed_url` matches a trusted list in `config/base.yaml`. Only works for items that have metadata (YouTube items). Produces WEAK_POSITIVE.
- **03c** labels items by running `lexicon_score` on their transcript: `raw_score < 0` → WEAK_NEGATIVE. Works for ALL SCREENED items regardless of metadata. Targets podcast_rss items (no metadata) that contain non-Lebanese dialect signals. See `src/dialect/scoring.py:lexicon_score` for formula.
- **04** requires a trained `models/dialect_classifier.joblib`. Assigns final outcome labels.

Error states: `ERROR_DOWNLOAD`, `ERROR_TRANSCRIBE`, `ERROR_SCORE`, etc.

---

## Directory Structure

```
lebanese_audio_collector/
├── config/base.yaml          # Main config (platforms, models, thresholds, feeds)
├── data/
│   ├── queue.db              # SQLite DB (~21MB, ~1000+ items)
│   ├── raw_audio/            # Downloaded audio as WAV (mono, 16kHz, loudnorm)
│   ├── samples/              # 3×20s chunks per audio for screening
│   ├── transcripts/          # JSON screening transcripts per item
│   └── evaluation/           # (empty) for evaluation results
├── src/
│   ├── cfg.py                # Pydantic config models — validated config access
│   ├── db.py                 # SQLAlchemy ORM + queue operations
│   ├── asr/whisper_engine.py # Faster-Whisper transcription
│   ├── dialect/
│   │   ├── adi_model.py      # CAMeL-Lab BERT dialect classifier wrapper
│   │   ├── lexicons.py       # Curated Lebanese/MSA/Egyptian/Gulf/Syrian word lists
│   │   └── scoring.py        # Hybrid scoring: lexicon + ADI classifier
│   ├── embeddings/engine.py  # paraphrase-multilingual-MiniLM-L12-v2 embeddings
│   ├── platforms/
│   │   ├── youtube.py        # yt-dlp download + ffmpeg normalization
│   │   ├── youtube_rss.py    # RSS-based channel discovery (no API quota)
│   │   ├── youtube_discovery.py # YouTube Data API search
│   │   ├── podcast.py        # HTTP podcast download, SHA-1 filenames
│   │   ├── tiktok.py         # yt-dlp TikTok extraction
│   │   ├── instagram.py      # STUB — not implemented
│   │   └── facebook.py       # STUB — not implemented
│   └── utils/audio.py        # FFmpeg chunk extraction (3×20s random)
├── scripts/                  # Pipeline steps (numbered, run in order)
├── tools/                    # One-time config helpers (channel ID resolution, etc.)
├── models/                   # Trained models saved here (dialect_classifier.joblib)
├── .env                      # PodcastIndex API credentials
├── requirements.txt
└── PIPELINE.md               # Step-by-step pipeline documentation
```

---

## Pipeline Scripts (Run Order)

```bash
# 0. Check queue status
python scripts/00_show_queue.py

# 1. Discover content (run as needed; RSS is free and preferred)
python scripts/01b_discover_youtube_rss.py     # primary: no quota
python scripts/01a_discover_youtube.py         # optional: uses YouTube API quota
python scripts/01c_discover_podcast_rss.py     # podcast feeds
python scripts/01d_discover_tiktok.py          # TikTok (stub/limited)

# 2. Download audio
python scripts/02_download_audio.py            # DISCOVERED → DOWNLOADED

# 3. Transcribe screening chunks
python scripts/03_transcribe_screening.py      # DOWNLOADED → SCREENED

# 3b. Assign weak labels from trusted metadata (YouTube items with channel_id)
python scripts/03b_assign_weak_labels.py       # SCREENED → WEAK_POSITIVE (metadata-based)

# 3c. Assign lexical negatives (podcast_rss and any SCREENED item without metadata)
python scripts/03c_assign_lexical_negatives.py # SCREENED → WEAK_NEGATIVE (lexicon-based)
#     Criterion: raw_score < 0 (non-LB signals outweigh LB) AND Arabic AND has dialect vocab
#     Note: lb==0 is NOT used — LEBANESE_WORDS contains pan-Arabic words that appear everywhere

# 5. Train dialect classifier (needs both WEAK_POSITIVE and WEAK_NEGATIVE)
python scripts/05_train_dialect_model.py       # saves models/dialect_classifier.joblib
#     Trained 2026-04-13: 92% accuracy, ROC-AUC 0.9774 on 648 pos + 1005 neg

# 4. Score dialect with trained model
python scripts/04_score_dialect.py             # SCREENED → POTENTIAL_LB / BORDERLINE_LB / REJECTED
```

### Backfill / Utilities

```bash
python scripts/backfill_youtube_metadata.py --limit 100
python scripts/backfill_podcast_metadata.py --limit 100
python scripts/05_reset_for_rescoring.py       # reset scored items to re-run scoring
python scripts/06_reset_podcast_downloads.py
```

### Additional / Undocumented Scripts

These exist on disk but are not part of the main pipeline flow:

| Script | Purpose |
|--------|---------|
| `scripts/01_seed_queue.py` | Manually seed URLs into the queue |
| `scripts/01e_discover_instagram.py` | Instagram discovery (disabled, stub) |
| `scripts/01f_discover_facebook.py` | Facebook discovery (disabled, stub) |
| `scripts/check_youtube_stages.py` | Inspect YouTube items by pipeline stage |
| `scripts/rerun_youtube_error_downloads.py` | Retry `ERROR_DOWNLOAD` items for YouTube |
| `scripts/reset_weak_labels.py` | Reset `WEAK_POSITIVE`/`WEAK_NEGATIVE` → `SCREENED` |
| `scripts/03c_assign_lexical_negatives.py` | Assign WEAK_NEGATIVE via lexical scoring (added 2026-04-13) |
| `scripts/07_download_adi17_contrastive.py` | Download dev+test splits from ADI17 (EGY, Gulf, LEB) |
| `scripts/08_download_adi17_msa.py` | Local MSA grinder over ADI17 train split — slow on home connection, superseded by Colab approach |
| `scripts/09_find_msa_files.py` | Footer-only Parquet scan to identify MSA-bearing files (column stats unavailable on this dataset, kept for reference) |
| `scripts/colab_extract_msa.py` | **Run in Google Colab.** Iterates ADI17 train split via `hf_transfer`. Empirically confirmed 0 MSA in ADI17 (2026-04-27 — see FINDINGS 8.2.3). Kept for documentation. |
| `scripts/colab_extract_msa_fleurs.py` | **Run in Google Colab.** Tries `google/fleurs` (ar_eg) then `mozilla-foundation/common_voice_17_0` (ar) — first attempt failed silently 2026-04-27, kept for reference. Use `colab_extract_msa_cv.py` instead. |
| `scripts/colab_extract_msa_cv.py` | **Run in Google Colab. PRIMARY MSA SOURCE.** Loads `google/fleurs` config `ar_eg` (Conneau et al. 2022; CC-BY-SA; multi-speaker; MSA register read from FLoRes-101 sentences). Pin `datasets==2.21.0` (newer versions dropped script-based loaders). Bundles into `fleurs_msa.zip` for local import. **2026-04-27: imported 798 unique MSA items.** |
| `scripts/11_prep_embeddings_audio.py` | Bundle audio for v2: trim each item to 30s @ 96k MP3, manifest into `audio_for_embeddings.zip`. |
| `scripts/11b_compact_embeddings_zip.py` | Re-encode to 10s @ 64k for smaller distribution (~950 MB → can host as GitHub Release if needed). |
| `scripts/12_extract_embeddings_local.py` | **PHASE 3 EXTRACTION.** Run wav2vec2-xls-r-300m locally on CPU; outputs `data/embeddings.parquet` (1024-d mean-pooled). Resumable. Replaces the earlier Colab path. |
| `scripts/colab_extract_embeddings.py` | Earlier Colab extraction script — kept for reference. |
| `scripts/13_load_embeddings.py` | Read `data/embeddings.parquet`, join with queue.db, derive binary `lebanese` labels, output `data/embeddings_with_labels.parquet`. |
| `scripts/13b_eval_v1_on_gt.py` | Evaluate v1 (text-only) on the held-out 300-item ground truth; produces v1↔v2 comparison numbers. Reads `models/dialect_classifier.joblib`, `data/annotations.csv`, transcripts. Appends Section 9.0 to FINDINGS.md. Read-only on data and model. |
| `scripts/14_train_classifier_v2.py` | Train LR + MLP on acoustic embeddings; evaluate on held-out 300 GT items; save `models/dialect_classifier_v2_acoustic.joblib`; append Section 12 to FINDINGS.md. `--strict_positive` filters WEAK_POSITIVE by `strong_lb_hits>=1`. **2026-05-01: V2 underperforms V1 on held-out GT (macro F1 0.38 vs 0.74).** |
| `scripts/15_remediation_v2.py` | After V2 underperformed: run threshold tuning, per-source class balancing, and hybrid V1+V2 feature experiments. Appends Section 12.6 to FINDINGS.md. **2026-05-02: confirmed recording-domain confound; no remediation matches V1.** |
| `scripts/colab_finetune_xlsr.py` | **Run in Google Colab on T4 GPU.** End-to-end fine-tunes wav2vec2-xls-r-300m: top 6 transformer layers + classification head trainable, CNN feature extractor + bottom 18 layers frozen. Per-(platform, label) WeightedRandomSampler to defeat recording-domain confound. Pulls audio from public GitHub release. ~30-60 min wall clock. Outputs `xlsr_finetuned.zip`. |
| `scripts/16_eval_finetuned_xlsr.py` | Local: extracts the Colab fine-tuned model, re-evaluates on the canonical 300 GT, and appends Section 13 (V2.5) to FINDINGS.md with full scoreboard against V1, V2, V2-balanced, and hybrid. |
| `scripts/16b_eval_local_v25.py` | **Local-checkpoint adapter for V2.5 eval.** Loads `models/xlsr_finetune_ckpt/latest.pt`, saves to HF format under `models/xlsr_finetuned/`, runs inference on GT, appends FINDINGS Section 13. Used when the local fine-tune (`16_finetune_xlsr_local.py`) did not run to natural epoch-end. |
| `scripts/16c_append_findings_section13.py` | Idempotent FINDINGS Section 13 appender from already-saved `models/v25_eval_local.json`. Used to recover from crashed eval prints (Windows charmap). |
| `scripts/10_import_colab_msa.py` | Local: auto-detects `data/fleurs_msa.zip` or `data/adi17_msa.zip` and imports into queue.db with `contrastive_role=msa`. Generic across sources. |
| `scripts/20_benchmark_harness.py` | **ROADMAP v2 Day 1.** Unified evaluation harness for the Lebanese cross-domain DID benchmark. Standardized predict-then-score interface for every system; per-system pause/resumable via `data/benchmark_predictions/<name>.json`; computes 12-threshold sweep, percentile-bootstrap (n=1000) 95% CIs on macro F1 + ROC-AUC, confusion matrices. Outputs `data/benchmark_results.csv`. |
| `scripts/benchmark_harness_utils.py` | Shared utilities for the benchmark family (load_gt_items, compute_metrics, bootstrap CIs, upsert_csv_row, save/load_predictions). Imported by 22, 23, and future scripts. |
| `scripts/21_platform_probe.py` | **ROADMAP v2 Day 2.** Direct test of the recording-domain confound mechanism (§12.6.3). Trains 4-way LogisticRegression on V2's 1024-d embeddings -> {adi17, youtube, podcast_rss, fleurs}. Reports per-class P/R/F1 + confusion matrix + accuracy. Output: `data/platform_probe_results.json`, `models/platform_probe.joblib`, FINDINGS Section 16. **2026-05-18: accuracy 0.8912 (chance 0.25) — mechanism claim directly supported.** |
| `scripts/22_v1_ablation.py` | **ROADMAP v2 Day 2.** Trains two V1 variants — lex-only (5-d) and embedding-only (384-d) — on the same WEAK_POSITIVE+WEAK_NEGATIVE+REJECTED pool with `strong_lb_hits>=1` filter (mirrors `scripts/05_train_dialect_model.py`). Saves both models, runs benchmark eval, upserts rows to `data/benchmark_results.csv`, appends FINDINGS Section 17. **2026-05-18: V1 embedding-only beats V1 combined.** |
| `scripts/23_whisper_lid_baseline.py` | **ROADMAP v2 Day 2.** Uses Whisper's existing `language_probability` for "ar" as a free-lunch Lebanese-detection baseline. Adds one row to benchmark CSV + FINDINGS Section 18. **2026-05-18: ROC-AUC 0.500 — confirmed not useful (all chunks predicted Arabic uniformly).** |
| `scripts/24_eval_public_systems.py` | **ROADMAP v2 Day 3.** Evaluates 4 publicly released Arabic dialect ID systems on the held-out 296-item Lebanese GT: `badrex/mms-300m-arabic-dialect-identifier` (Badr 2025), `tiantiaf/voxlect-arabic-dialect-mms-lid-256` (Voxlect 2026), `Elyadata/ADI-whisper-ADI20` (Elleuch 2025 — the only country-level `LEB` baseline), `IbrahimAmin/marbertv2-arabic-written-dialect-classifier` (text). Per-system pause/resume via `data/benchmark_predictions/<name>.json`. Idempotent FINDINGS Section 19 write (strips prior block before appending). Loader dispatch via the SYSTEMS dict's `loader` field: HF AutoModelForAudioClassification, HF AutoModelForSequenceClassification, vendored `MMSWrapper` (Voxlect — `scripts/_vendored/voxlect_mms_dialect.py`, requires `loralib`), and vendored `WhisperDialectClassifier` (Elyadata — `scripts/_vendored/elyadata_classifier_attention_pooling.py`, requires `speechbrain` + downloads `openai/whisper-large-v3` on first run). |
| `scripts/_vendored/` | Third-party class definitions vendored from upstream repos so we can load non-standard HF models without installing the full upstream packages. `voxlect_mms_dialect.py` is `MMSWrapper` from github.com/tiantiaf0627/voxlect (used by Voxlect — `PyTorchModelHubMixin`-based; auto-loads config from HF). `elyadata_classifier_attention_pooling.py` is `WhisperDialectClassifier` from github.com/elyadata/ADI-20 (used by Elyadata; SpeechBrain `Pretrained` subclass). |
| `scripts/_deprecated/` | Archived scripts no longer in the active pipeline. See `scripts/_deprecated/README.md`. Reference only; do not run. |

### Error Recovery Map

| Error / stuck state | Recovery script |
|--------------------|----------------|
| `ERROR_DOWNLOAD` (YouTube) | `scripts/rerun_youtube_error_downloads.py` |
| `ERROR_DOWNLOAD` (podcast) | `scripts/06_reset_podcast_downloads.py` |
| `ERROR_SCORE` or any scored state | `scripts/05_reset_for_rescoring.py` |
| `WEAK_POSITIVE` / `WEAK_NEGATIVE` | `scripts/reset_weak_labels.py` then re-run 03b + 03c |

---

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate       # Windows (bash)
pip install -r requirements.txt
```

**Python version:** 3.13

**Run all scripts from the project root** (`lebanese_audio_collector/`) with the venv active. All paths in scripts are relative to the root — running from elsewhere will silently break path resolution.

**External requirements:**
- `ffmpeg` and `ffprobe` must be installed and in PATH
- `.env` file must have `PODCASTINDEX_API_KEY` and `PODCASTINDEX_API_SECRET`
- `config/base.yaml` must have a valid `youtube_api_key` for API-based discovery

---

## Key Technologies

| Purpose | Library / Tool |
|---------|---------------|
| Audio download | `yt-dlp`, `requests` |
| Audio processing | `ffmpeg` / `ffprobe` |
| ASR (screening) | `faster-whisper` (`base` model — switched from `medium` 2026-04-14 after benchmarking: 10x faster, lang_prob=1.00, adequate quality for lexical scoring) |
| ASR (full) | `faster-whisper` (large-v2, not yet scripted) |
| Dialect ID (ML) | `transformers`, `torch` — CAMeL-Lab BERT |
| Embeddings | `sentence-transformers` — MiniLM-L12-v2 |
| DB ORM | `sqlalchemy >= 2.0` (SQLite) |
| Config validation | `pydantic >= 2.0`, `PyYAML` |
| RSS parsing | `feedparser` |
| Classifier training | `scikit-learn` (LogisticRegression), `joblib` |
| YouTube API | `google-api-python-client` |

---

## Data Formats

### Database (`data/queue.db`, table: `queue`)

| Column | Type | Notes |
|--------|------|-------|
| `id` | int PK | auto-increment |
| `url` | str unique | deduplicated on insert |
| `platform` | str | youtube / podcast / tiktok |
| `status` | str | see status flow above |
| `discovered_at` | datetime | |
| `last_update_at` | datetime | |
| `error_msg` | str | populated on ERROR_* statuses |
| `audio_path` | str | path to raw WAV file |
| `duration_seconds` | float | |
| `rejection_reason` | str | populated on REJECTED |
| `source_metadata` | JSON | uploader, channel_id, title, etc. |

### Transcripts (`data/transcripts/clip_<id>_screening.json`)

```json
{
  "id": 123,
  "url": "...",
  "platform": "youtube",
  "screening_samples": [
    {
      "chunk_path": "data/samples/<id>_chunk0.wav",
      "language": "ar",
      "language_probability": 0.97,
      "text": "...",
      "segments": [{"id": 0, "start": 0.0, "end": 5.2, "text": "..."}]
    }
  ]
}
```

### Audio Files

- Raw: `data/raw_audio/<video_id>.wav` — mono, 16 kHz, loudness-normalized, max 1200s
- Chunks: `data/samples/<video_id>_chunk<N>.wav` — 3 chunks × 20 seconds each

---

## Configuration (`config/base.yaml`)

Key sections:
- `audio`: `min_seconds` (currently 6), `max_download_seconds` (1200), output dir
- `transcription`: `screening_model` (medium), `full_model` (large-v2), `device` (cpu), `compute_type` (int8)
- `youtube`: `api_key`, `region_code` (LB), `language` (ar), `search_queries`, `channel_ids`
- `podcast`: list of ~100 RSS feed URLs, discovery keywords
- `tiktok` / `instagram` / `facebook`: mostly empty/disabled
- `weak_labels`: trusted Lebanese/non-Lebanese channel/feed lists for bootstrapping

---

## Dialect Scoring Logic (`src/dialect/scoring.py`)

### Lexicon scoring (used in 03c and as features in 04)

`raw_score = lb*1.8 - msa*0.6 - egy*1.0 - gulf*1.0 - sy*0.5`
`final_score = max(0, min(1, raw_score / 5.0))`

**Important caveat:** `LEBANESE_WORDS` contains pan-Arabic words (يعني، في، بس، مش، تمام، مرحبا) that appear in virtually every Arabic transcript. As a result, `lb > 0` is nearly always true and is NOT a useful standalone signal. The meaningful signal is `raw_score < 0` (non-LB signals outweigh LB ones).

The ADI model (`src/dialect/adi_model.py`) is intentionally disabled (`adi: null` in all outputs). `final_dialect_score()` uses lexicon only.

### ML classifier (04 → POTENTIAL_LB / BORDERLINE_LB / REJECTED)

Features: `[lb, msa, strong_lb_hits, msa_ratio_core, final_score, *embedding_384d]`
Model: `LogisticRegression(class_weight="balanced")` — saved to `models/dialect_classifier.joblib`
Thresholds: `>= 0.75` → POTENTIAL_LB, `>= 0.50` → BORDERLINE_LB, else → REJECTED

**Trained 2026-04-13:**
- Training data: 648 WEAK_POSITIVE + 1,005 WEAK_NEGATIVE
- Validation accuracy: 92% | ROC-AUC: 0.9774
- Negatives came from 03c (lexical scoring on podcast_rss SCREENED items)

---

## Thesis Parameters Needing Justification

These values are currently set but require documented academic justification (tracked in `info.txt`):

| Location | Parameter | Current value | Why it matters |
|----------|-----------|--------------|----------------|
| `config/base.yaml` | `min_seconds` | 6 | Minimum audio length threshold |
| `src/platforms/youtube.py` | `ydl_opts` | various | Download options tuned for speech/dialect analysis |
| `src/cfg.py` | `model_size` | medium | Whisper model size trade-off |
| `src/cfg.py` | `device` | cpu | Inference device |
| `src/cfg.py` | `compute_type` | int8 | Quantization trade-off |
| `scripts/02_download_audio.py` | error handling strategy | — | How download failures are managed |

When touching any of these, flag that the value has thesis significance and needs justification.

---

## What Is Incomplete

| Feature | File | Status |
|---------|------|--------|
| Instagram discovery | `src/platforms/instagram.py` | Stub, TODO |
| Facebook discovery | `src/platforms/facebook.py` | Stub, TODO |
| TikTok discovery | `scripts/01d_discover_tiktok.py` | Partially implemented |
| Full transcription (post-screening) | Not scripted | Step beyond screening not yet added |
| Evaluation pipeline | `data/evaluation/` | Empty, not implemented |
| Thesis parameter justification | `info.txt` (root) | Needs: min_seconds, ydl_opts, model_size, compute_type |
| Non-Lebanese contrastive dataset | Not yet collected | Next thesis phase: MSA, Egyptian, Gulf dialect data |
| WEAK_NEGATIVE via metadata | `config/base.yaml` weak_labels | `trusted_non_lebanese_*` lists are empty — 03c (lexical) is the current source of negatives |

---

## Database Inspection

```bash
sqlite3 data/queue.db
.headers on
.mode column
SELECT status, COUNT(*) FROM queue GROUP BY status;
SELECT platform, status, COUNT(*) FROM queue GROUP BY platform, status;
SELECT * FROM queue WHERE status = 'POTENTIAL_LB' LIMIT 10;
```

---

## Current Dataset State (as of 2026-04-15)

| Status | Count | Meaning |
|--------|-------|---------|
| WEAK_POSITIVE | 3,232 | Trusted Lebanese source (metadata-based, YouTube) |
| POTENTIAL_LB | 1,769 | High-confidence Lebanese (model prob ≥ 0.75, podcast_rss) |
| BORDERLINE_LB | 313 | Plausible Lebanese (model prob 0.50–0.75) |
| **Total likely-Lebanese** | **5,314** | |
| WEAK_NEGATIVE | 1,005 | Confirmed non-Lebanese (lexical scoring on podcast_rss) |
| REJECTED | 2,583 | Scored and rejected (model prob < 0.50) |
| SCREENED | 0 | All screened items have been labeled |
| DISCOVERED | 19,044 | Mostly podcasts — not yet downloaded |
| ERROR_DOWNLOAD | 7,010 | Failed downloads (mostly podcast_rss exceeding 1200s cap) |

### Storage state
- `data/raw_audio/`: ~95 GB (3,251 YouTube FLAC + 3,656 podcast MP3)
- All YouTube audio converted to FLAC (lossless, from 315 GB → ~63 GB for YouTube portion)
- Podcast MP3s left as-is (already compressed)
- 2,163 orphan "too long" podcast MP3s deleted (105 GB freed)
- C: drive: ~240 GB free

### Trained model (2026-04-15)
- Training data: 500 WEAK_POSITIVE + 500 WEAK_NEGATIVE (script caps at 500 each)
- Validation: 91% accuracy, ROC-AUC 0.9741
- Saved to `models/dialect_classifier.joblib`
- **Note:** Training script has a `limit=500` cap in `05_train_dialect_model.py:47-48`. Removing it would let the model train on all 3,232+1,005 items for likely better performance.

### Next pipeline steps (for the thesis dialect ID model phase)

1. **Build a ground truth test set** — manually annotate ~200 clips (stratified across POTENTIAL_LB, BORDERLINE_LB, REJECTED, WEAK_NEGATIVE) as ground truth. Required for reporting real accuracy/F1.
2. **Collect non-Lebanese contrastive dataset** — systematic collection of MSA, Egyptian, Gulf audio. Creates a proper negative class for the dialect identification model.
3. **Train the final dialect classifier** — with balanced Lebanese vs. other-Arabic data, acoustic features + transcript embeddings, full evaluation pipeline.

---

## Notes for Thesis

- The pipeline is designed for **precision over recall**: better to reject uncertain items than include non-Lebanese audio
- RSS-based YouTube discovery is the primary method to avoid API quota limits
- Weak supervision bootstraps training data from trusted source metadata before full ML scoring
- `03c_assign_lexical_negatives.py` was added because `trusted_non_lebanese_*` lists were empty and podcast_rss items have no metadata — the lexical approach is the current mechanism for generating negatives
- Batch sizes are set inside scripts (e.g., `BATCH_SIZE = 600` in `02_download_audio.py`) — adjust as needed
- Scripts are resumable: they skip already-processed items based on DB status
- `04_score_dialect.py` processes 20 items per run — run in a loop (`for i in $(seq 1 N); do ...`) for bulk scoring
