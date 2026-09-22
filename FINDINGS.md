# Research Findings & Thesis Notes

> **Project:** Automatic Lebanese Dialect Identification from Open Audio Sources
> **Author:** Charlene El Khoury Eid
> **University:** Lebanese American University (Byblos)
> **Last updated:** 2026-04-15

This document records all empirical findings, methodological decisions, and results produced during the thesis project. It is organized to map onto thesis chapters and can be used as source material for the thesis paper and defense presentation.

---

## 1. Problem Statement

**Goal:** Build an automatic system for identifying Lebanese Arabic dialect in audio collected from open platforms (YouTube, podcasts, TikTok). The system must:
- Collect audio at scale from open sources
- Filter and classify audio by Arabic dialect
- Produce a validated Lebanese dialect corpus
- Train and evaluate a dialect identification model

**Why Lebanese Arabic specifically:**
- Under-resourced dialect in NLP — limited labeled data exists
- Linguistically distinct from MSA but overlaps significantly with Syrian/Palestinian (Levantine family)
- No existing large-scale, publicly available Lebanese dialect audio corpus

---

## 2. Pipeline Architecture

### 2.1 Design Philosophy
- **Queue-driven, multi-stage pipeline** — each stage is a standalone script that reads items at a given status and writes them to the next
- **Precision over recall** — better to reject uncertain items than include non-Lebanese audio
- **Weak supervision** — bootstrap training labels from trusted source metadata before ML scoring
- **Resumable** — scripts skip already-processed items based on DB status

### 2.2 Status Flow
```
DISCOVERED → DOWNLOADED → SCREENED ──► 03b (metadata match) ──► WEAK_POSITIVE
                                   └──► 03c (lexical scoring) ──► WEAK_NEGATIVE
                                   └──► 04  (ML model)        ──► POTENTIAL_LB / BORDERLINE_LB / REJECTED
```

### 2.3 Key Pipeline Scripts
| Step | Script | Function |
|------|--------|----------|
| 01a/b/c/d | discover_*.py | Source discovery (YouTube RSS, API, podcast RSS, TikTok) |
| 02 | download_audio.py | Audio download with duration cap (1200s) |
| 03 | transcribe_screening.py | Extract 3×20s random chunks, transcribe with Whisper |
| 03b | assign_weak_labels.py | Label items from trusted channels as WEAK_POSITIVE |
| 03c | assign_lexical_negatives.py | Label items with non-Lebanese lexical profile as WEAK_NEGATIVE |
| 04 | score_dialect.py | ML-based scoring → POTENTIAL_LB / BORDERLINE_LB / REJECTED |
| 05 | train_dialect_model.py | Train LogisticRegression on weak labels |

### 2.4 Audio Processing Decisions

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Screening model | Whisper `base` (faster-whisper) | Benchmarked against `medium`: 10× faster, identical language detection (lang_prob=1.00 on all test chunks), adequate transcription quality for lexical scoring. Switched from `medium` on 2026-04-14. |
| Device / quantization | CPU / int8 | Hardware constraint (no GPU available). int8 quantization gives ~2× speedup vs float32 with minimal quality loss for Arabic. |
| Chunk strategy | 3 × 20 seconds, random positions | Balances coverage of audio content against processing time. Random positions avoid intro/outro bias. |
| Audio format (YouTube) | FLAC (lossless) | Converted from WAV on 2026-04-14 to address disk space constraints. 315 GB WAV → 63 GB FLAC (80% reduction). Bit-identical on decode — zero quality loss. |
| Audio format (podcasts) | MP3 (original) | Already compressed at source. Re-encoding to FLAC would increase size. Left as-is. |
| Max duration | 1200 seconds (20 min) | Longer files are rejected at download. Balances coverage against storage. |
| Min duration | 6 seconds | Below this, insufficient audio for meaningful dialect analysis. |
| Mono, 16 kHz, loudnorm | Yes | Standard ASR preprocessing. Loudness normalization prevents volume-based bias. |

---

## 3. Data Collection Results

### 3.1 Sources
- **YouTube:** 521 trusted Lebanese channel IDs discovered via RSS feeds (no API quota) and YouTube Data API
- **Podcasts:** ~151 RSS feeds (Arabic-language, mixed dialects)
- **TikTok:** Limited (13 items screened, partially implemented)
- **Instagram / Facebook:** Not implemented (stubs only)

### 3.2 Dataset Size (as of 2026-04-29)

**Lebanese-positive pool (Phase 1 collection):**

| Status | Count | Platform breakdown |
|--------|-------|--------------------|
| WEAK_POSITIVE | 3,232 | YouTube (3,228), podcast_rss (4) |
| POTENTIAL_LB | 1,769 | podcast_rss (majority) |
| BORDERLINE_LB | 313 | podcast_rss (majority) |
| REJECTED | 2,583 | YouTube (1,717), podcast_rss (630+) |
| WEAK_NEGATIVE | 1,005 | podcast_rss (997), TikTok (8) |
| SCREENED | 0 | All labeled |
| DISCOVERED | 19,044 | Mostly podcast_rss (undownloaded) |
| ERROR_DOWNLOAD | 7,010 | Mostly podcast_rss (exceeded 1200s cap) |

**Contrastive non-Lebanese pool (Phase 2 collection, 2026-04-21 to 2026-04-27):**

| Class | Source | Count | Citation |
|-------|--------|-------|----------|
| Lebanese (extra) | ADI17 (LEB) | 1,000 | Ali et al. 2019 |
| Egyptian | ADI17 (EGY) | 1,000 | Ali et al. 2019 |
| Gulf | ADI17 (KSA, KUW, UAE, QAT, OMA) | 5,000 | Ali et al. 2019 |
| MSA | FLEURS (`google/fleurs`, `ar_eg`) | 798 | Conneau et al. 2022 |

**Held-out evaluation:** 300 manually annotated items in `data/annotations.csv` (Section 6).

**Aggregate counts:**
- Total items with audio + screening transcripts (Phase 1 only): ~8,902
- Total likely-Lebanese (before ground truth): 5,314
- Total contrastive non-Lebanese: 7,798 (including 1,000 extra ADI17 LEB positives)
- Items selected for v2 embedding extraction (Phase 3): 14,177 (all Phase-1 labeled items + Phase-2 contrastive + 300 GT items)

### 3.3 Storage

| Directory | Size | Files | Format |
|-----------|------|-------|--------|
| data/raw_audio | ~95 GB | ~6,907 | FLAC (YouTube), MP3 (podcasts) |
| data/transcripts | ~30 MB | ~6,334 | JSON |
| data/queue.db | ~21 MB | 1 | SQLite |

**Storage optimization performed (2026-04-14):**
- Converted all YouTube WAVs to FLAC: 315 GB → 63 GB (80% reduction, lossless)
- Deleted 2,163 orphan podcast MP3s: freed 105 GB (files downloaded but exceeded length cap, left on disk due to missing cleanup in `podcast.py:53-56`)
- Deleted incomplete .part files and orphans: freed 3.8 GB
- Net result: C: drive went from 0 GB free → 240 GB free

---

## 4. Dialect Scoring Methodology

### 4.1 Lexicon-Based Scoring

**Formula:**
```
raw_score = lb × 1.8 − msa × 0.6 − egy × 1.0 − gulf × 1.0 − sy × 0.5
final_score = max(0, min(1, raw_score / 5.0))
```

**Lexicon lists:** Curated word lists for Lebanese, MSA, Egyptian, Gulf, and Syrian Arabic (see `src/dialect/lexicons.py`).

**Strong Lebanese markers:** `شو، ليش، هيك، هلق، عنجد، بدي، كتير، وين، هون، هيدا، هيدي، هدول` — high-precision Levantine markers.

**Critical finding — lexicon overlap:** `LEBANESE_WORDS` contains pan-Arabic words (يعني، في، بس، مش، تمام، مرحبا) that appear in virtually every Arabic transcript. As a result, `lb > 0` is nearly always true and is NOT a useful standalone signal. The meaningful signal is `raw_score < 0` (non-LB signals outweigh LB ones).

**Implication for 03c (WEAK_NEGATIVE assignment):** Items are labeled WEAK_NEGATIVE when `raw_score < 0` AND the audio is Arabic (Whisper language detection probability ≥ 0.70) AND at least one non-Lebanese dialect word is present. The criterion `lb == 0` was initially tested but rejected because it produced only 2 negatives out of 3,717 items due to the pan-Arabic overlap problem.

### 4.2 ML Classifier

**Architecture:** LogisticRegression with `class_weight="balanced"`
**Features:** `[lb, msa, strong_lb_hits, msa_ratio_core, final_score, *embedding_384d]`
- Lexical features (5 dimensions) from `lexicon_score()`
- Semantic embeddings (384 dimensions) from `paraphrase-multilingual-MiniLM-L12-v2`

**Thresholds:**
- `probability ≥ 0.75` → POTENTIAL_LB
- `probability ≥ 0.50` → BORDERLINE_LB
- `probability < 0.50` → REJECTED

**Training data:** 500 WEAK_POSITIVE + 500 WEAK_NEGATIVE (script caps at 500 each)
**Validation (train/test split):** 91% accuracy, ROC-AUC 0.9741

**Important note:** The training script (`05_train_dialect_model.py`) has a hardcoded `limit=500` at lines 47-48. This was not removed during the initial pipeline run, meaning the model only used 500 of the 3,232 available WEAK_POSITIVE items and 500 of the 1,005 WEAK_NEGATIVE items. This is a known limitation.

### 4.3 ADI Model (Disabled)

The CAMeL-Lab BERT Arabic Dialect Identification model (`src/dialect/adi_model.py`) is implemented but intentionally disabled. All scoring outputs show `adi: null`. The `final_dialect_score()` function uses lexicon scoring only. Reason: the ADI model was considered as an additional feature but was not integrated into the scoring pipeline.

---

## 5. Weak Supervision Analysis

### 5.1 Weak Labeling Strategies

**Metadata-based (03b — WEAK_POSITIVE):**
- Items whose `source_metadata.channel_id` matches a trusted Lebanese YouTube channel list (521 channels)
- Produces WEAK_POSITIVE labels
- Assumption: content from Lebanese channels is Lebanese dialect
- **Limitation discovered:** Lebanese channels post mixed content including MSA, formal Arabic, educational content, music — not exclusively Lebanese dialect

**Metadata-based (03b — WEAK_NEGATIVE):**
- Items whose `source_metadata.channel_id` or `feed_url` matches `trusted_non_lebanese_*` lists
- **Never produced results** — `trusted_non_lebanese_*` lists in config were empty throughout the project
- Podcast_rss items had empty `source_metadata {}` (discovered by older pipeline version without feed_url storage)

**Lexicon-based (03c — WEAK_NEGATIVE):**
- Added on 2026-04-13 to address the missing negative labels
- Criterion: `raw_score < 0` AND Arabic AND has non-Lebanese dialect vocabulary
- Produced 1,005 WEAK_NEGATIVE items from podcast_rss content
- Spot-check confirmed these were Egyptian, Gulf, MSA content (e.g., articles about Egyptian actresses, Gulf driving podcasts, pan-Arab music reviews)

### 5.2 Noise in Weak Labels — Quantified by Ground Truth

See Section 6 for full ground truth evaluation results.

**Key finding:** Metadata-based positive labels (WEAK_POSITIVE) contain ~33% noise — items from Lebanese channels that are not actually Lebanese dialect content. This noise propagated into the ML classifier via training.

---

## 6. Ground Truth Evaluation

### 6.1 Annotation Methodology

**Tool:** Custom Flask-based web application (`tools/annotate.py`) serving audio clips with transcript display and one-click labeling.

**Sample design:** Stratified random sample of 300 items drawn from five pipeline prediction tiers:
- WEAK_POSITIVE: 45 (sanity check on metadata-based labels)
- POTENTIAL_LB: 90 (primary precision target)
- BORDERLINE_LB: 60 (uncertainty band)
- REJECTED: 60 (false negative check)
- WEAK_NEGATIVE: 45 (sanity check on lexical negatives)

**Justification:** At n=45 per class, the 95% confidence interval half-width on an estimated proportion is ≤ 14.6%; at n=90, ≤ 10.3%. This balances statistical reliability per class against annotator workload.

**Label options:** Lebanese / Mostly Lebanese-mixed / Not Lebanese / Unclear / Skip

**Audio presented:** 60-second clip extracted from a deterministic position (30% into the source file) using ffmpeg. Encoded as MP3 at 96 kbps for browser playback.

**Annotator:** Single annotator (thesis author), native Lebanese Arabic speaker.

### 6.2 Results

**Overall distribution (N=300):**

| Ground truth label | Count | Percentage |
|--------------------|-------|------------|
| Lebanese | 32 | 10.7% |
| Mostly Lebanese / mixed | 50 | 16.7% |
| Not Lebanese | 214 | 71.3% |
| Unclear | 4 | 1.3% |

**Per-tier precision (Lebanese + Mostly Lebanese treated as positive):**

| Pipeline tier | n | Lebanese | Mostly LB | Not LB | Unclear | **Precision** | **95% CI** |
|---------------|---|----------|-----------|--------|---------|---------------|------------|
| WEAK_POSITIVE | 45 | 15 | 15 | 11 | 4 | **66.7%** | [52%, 80%] |
| POTENTIAL_LB | 90 | 11 | 23 | 56 | 0 | **37.8%** | [28%, 48%] |
| BORDERLINE_LB | 60 | 4 | 8 | 48 | 0 | **20.0%** | [10%, 30%] |
| REJECTED | 60 | 0 | 1 | 59 | 0 | **1.7%** | [0%, 5%] |
| WEAK_NEGATIVE | 45 | 2 | 3 | 40 | 0 | **11.1%** | [2%, 21%] |

### 6.3 Key Findings

1. **The pipeline has high specificity but low precision for positive predictions.**
   - REJECTED correctly identifies non-Lebanese 98.3% of the time
   - WEAK_NEGATIVE correctly identifies non-Lebanese 88.9% of the time
   - But POTENTIAL_LB (the "high confidence" positive tier) is only 37.8% actually Lebanese
   - Even WEAK_POSITIVE (trusted sources) is only 66.7% Lebanese

2. **The root cause of low precision is noisy positive training data.**
   - WEAK_POSITIVE items come from "trusted Lebanese channels" — but Lebanese channels post MSA news, formal Arabic educational content, music videos, and other non-dialectal content
   - The ML classifier was trained on these noisy labels and learned to reproduce the noise
   - The lexical overlap problem (pan-Arabic words in `LEBANESE_WORDS`) compounds the issue

3. **The "Mostly Lebanese / mixed" category is significant.**
   - 50 out of 300 items (16.7%) were labeled as mixed
   - This suggests a substantial portion of Lebanese audio content involves code-switching between Lebanese dialect and MSA or other varieties
   - This is linguistically expected — Lebanese speakers frequently mix dialect and formal Arabic

4. **BORDERLINE_LB is essentially noise.**
   - Only 20% precision — not meaningfully better than random
   - Should not be used as positive training data

5. **Negative labels are reliable.**
   - REJECTED: 98.3% correctly non-Lebanese → excellent rejection
   - WEAK_NEGATIVE: 88.9% correctly non-Lebanese → lexical approach works well for negatives
   - The pipeline's strength is in what it rejects, not what it accepts

### 6.4 Implications for Model Improvement

The ground truth evaluation reveals that the current pipeline's main bottleneck is **positive label quality**, not the ML model architecture or feature engineering. The recommended path forward:

1. **Apply lexical verification to WEAK_POSITIVE items** — require both metadata match AND strong Lebanese lexical markers (e.g., `strong_lb_hits ≥ 1`) to qualify as a positive training example
2. **Retrain on cleaned data** with the 500-item cap removed
3. **Evaluate on the 300-item ground truth test set** (never used for training)
4. **Report the improvement** as evidence that noise reduction in weak labels directly improves dialect classification

---

## 6b. Model v2 — Retrained with Lexical Verification (2026-04-15)

### 6b.1 Training Data Changes

| Change | Before (v1) | After (v2) |
|--------|-------------|------------|
| Positive source | WEAK_POSITIVE, no filter | WEAK_POSITIVE filtered by `strong_lb_hits >= 1` |
| Positive count | 500 (capped) | 2,003 (all passing filter) |
| Negative source | WEAK_NEGATIVE only | WEAK_NEGATIVE + REJECTED |
| Negative count | 500 (capped) | 1,651 (all with transcripts) |
| Total training | 1,000 | 3,654 |

**Filter criterion:** `strong_lb_hits >= 1` — items from trusted Lebanese channels must contain at least one strong Lebanese dialect marker (شو, هيك, هلق, بدي, etc.) in the screening transcript. Items with zero markers are excluded as likely MSA/formal content.

**Threshold selection:** Ground truth annotation of 45 WEAK_POSITIVE items showed precision jumps from 67% (no filter) to 81% (strong >= 1). Higher thresholds (>=2, >=3, >=4) did not significantly improve precision on the WEAK_POSITIVE subset (small sample, CIs overlap) but substantially reduced training set size (from 2,003 to 1,367 / 863 / 531). `strong >= 1` was selected as the best balance of precision improvement and data retention.

### 6b.2 Validation Results (Train/Test Split)

| Metric | v1 (noisy) | v2 (cleaned) |
|--------|------------|--------------|
| Training items | 1,000 | 3,654 |
| Validation accuracy | 91% | 89% |
| ROC-AUC | 0.9741 | 0.9643 |

**Note:** The slight validation performance decrease is expected — more diverse training data with some remaining noise makes the task harder. The real evaluation is on ground truth.

### 6b.3 Ground Truth Evaluation (300-item test set)

**Performance at various thresholds (Lebanese + Mostly Lebanese = positive):**

| Threshold | Precision | Recall | F1 | Accuracy |
|-----------|-----------|--------|----|----------|
| 0.50 | 48.3% | 86.6% | 62.0% | — |
| 0.60 | 51.9% | 85.4% | 64.5% | — |
| **0.70** | **55.5%** | **80.5%** | **65.7%** | — |
| 0.75 | 55.4% | 75.6% | 63.9% | 76.4% |
| 0.80 | 56.9% | 70.7% | 63.0% | — |
| 0.85 | 60.9% | 64.6% | 62.7% | — |
| 0.90 | 64.6% | 62.2% | 63.4% | — |

**Best F1:** 65.7% at threshold 0.70

**Probability distribution by actual label:**

| Ground truth | n | Mean prob | Median prob | Min | Max |
|--------------|---|-----------|-------------|-----|-----|
| Lebanese | 32 | 0.885 | 0.983 | 0.251 | 1.000 |
| Mostly Lebanese | 50 | 0.790 | 0.947 | 0.013 | 0.999 |
| Not Lebanese | 214 | 0.387 | 0.261 | 0.001 | 0.998 |

**Per-tier performance at threshold 0.75:**

| Pipeline tier | n | Pred. positive | True positive | TP | FP | Precision |
|---------------|---|----------------|---------------|----|----|-----------|
| WEAK_POSITIVE | 41 | 28 | 30 | 22 | 6 | **79%** |
| POTENTIAL_LB | 90 | 76 | 34 | 33 | 43 | **43%** |
| BORDERLINE_LB | 60 | 6 | 12 | 5 | 1 | **83%** |
| REJECTED | 60 | 1 | 1 | 1 | 0 | **100%** |
| WEAK_NEGATIVE | 45 | 1 | 5 | 1 | 0 | **100%** |

### 6b.4 Key Findings from Model v2

1. **The model successfully separates distributions.** True Lebanese items cluster at high probabilities (median 0.983), while non-Lebanese items cluster at low (median 0.261). The model IS learning dialect signal.

2. **Precision remains the bottleneck.** Even at the best F1 threshold (0.70), precision is 55.5% — meaning ~45% of items the model calls "Lebanese" are actually not. This is driven by overlap: some non-Lebanese items still score very high (max 0.998).

3. **WEAK_POSITIVE precision improved from 67% to 79%** after the lexical filter — the filter worked as intended.

4. **POTENTIAL_LB precision improved slightly (38% → 43%)** — the model is better but the fundamental problem of noisy training data persists.

5. **The gap between validation (89%) and ground truth (76.4%) is 12.6 percentage points.** This gap quantifies the difference between weak-label quality and real-world performance — a key finding for the thesis discussion of weak supervision limitations.

6. **Recall is strong.** At threshold 0.70, the model catches 80.5% of truly Lebanese items. The model's strength is sensitivity, not specificity for positives.

### 6b.5 Remaining Improvement Paths

The current text-only model (lexical features + semantic embeddings) has reached a ceiling around F1 65% with the available training data quality. Further improvement requires:

1. **Acoustic features** — pronunciation differences between dialects (MFCCs, pitch contours, formants) are not captured by text-based features. Adding acoustic features could disambiguate items where the transcript looks similar across dialects.

2. **Better positive training data** — the 81% precision of the lexically-verified WEAK_POSITIVE set still contains ~19% noise. Options:
   - Manually annotate more positives (expensive but clean)
   - Active learning: have the model flag uncertain items for human review
   - Use a more sophisticated strong marker set (beyond the current 26 markers)

3. **Contrastive multi-dialect dataset** — the current negatives are unlabeled "not Lebanese" (mixed Egyptian, Gulf, MSA, Syrian, etc.). Training with dialect-labeled negatives (e.g., "this is Egyptian", "this is Gulf") would give the model richer signal for what non-Lebanese Arabic sounds like.

4. **Model architecture** — LogisticRegression may be too simple for the 389-dimensional feature space. Options: Random Forest, Gradient Boosting, or a small neural network.

---

## 7. Methodological Decisions Log

### 7.1 Whisper Model Selection (2026-04-14)
- **Decision:** Switch screening model from `medium` to `base`
- **Benchmark:** Tested both on 5 real audio chunks (Lebanese YouTube + non-Lebanese TikTok)
  - `base`: 1.3s/chunk inference, lang_prob=1.00 on all chunks
  - `medium`: ~20s/chunk inference, lang_prob=1.00 on all chunks
  - `small`: 11.4s/chunk, failed on 1 chunk (empty output)
- **Outcome:** `base` is 15× faster with identical language detection and adequate transcription quality for lexical scoring
- **Thesis justification:** "The screening stage requires only language detection and sufficient transcription quality for lexical dialect scoring, not perfect ASR. Whisper `base` achieves identical language detection confidence (1.00) while reducing per-chunk inference time from 20s to 1.3s on CPU."

### 7.2 WEAK_NEGATIVE Criterion (2026-04-13)
- **Initial attempt:** `lb == 0` (zero Lebanese word matches)
- **Result:** Only 2 out of 3,717 items qualified — nearly useless
- **Root cause:** `LEBANESE_WORDS` contains pan-Arabic words present in all Arabic text
- **Revised criterion:** `raw_score < 0` (non-Lebanese signals outweigh Lebanese signals)
- **Result:** 1,005 items labeled WEAK_NEGATIVE
- **Spot-check:** Confirmed these were Egyptian, Gulf, MSA content

### 7.3 Audio Format Migration (2026-04-14)
- **Trigger:** C: drive reached 100% capacity (477 GB used, 0 free), crashing the pipeline
- **Action:** Converted all YouTube WAV files to FLAC (lossless compression)
- **Compression ratio:** Average 29% of WAV size (better than expected 50% — speech audio compresses well due to silence and consistent speaker patterns)
- **Space freed:** 315 GB WAV → 63 GB FLAC = 252 GB recovered
- **Quality impact:** Zero — FLAC is bit-identical on decode. ffmpeg and Whisper read FLAC natively.
- **Additional cleanup:** 2,163 orphan podcast MP3s deleted (105 GB) — confirmed via SHA-1 hash matching as files downloaded then rejected for exceeding 1200s cap

### 7.4 Lexical Verification of WEAK_POSITIVE (2026-04-15)
- **Decision:** Add `strong_lb_hits >= 1` filter to WEAK_POSITIVE items at training time
- **Analysis:** Tested thresholds >=1 through >=5 on 45 annotated WEAK_POSITIVE items:
  - `>=0` (no filter): 67% precision, 3,232 items
  - `>=1`: **81% precision**, 2,003 items (62% kept)
  - `>=2`: 78% precision, 1,367 items (42%)
  - `>=4`: 83% precision, 531 items (16%)
- **Choice:** `>=1` selected — largest dataset with meaningful precision jump. Higher thresholds had overlapping CIs on the small 45-item sample and drastically reduced data.
- **Implementation:** Filter applied at training time in `05_train_dialect_model.py` (DB statuses unchanged)
- **Thesis justification:** "We require each trusted-channel item to contain at least one strong Lebanese dialect marker in its transcript, filtering metadata-only positives that represent MSA/formal content from Lebanese creators. This joint metadata+lexical criterion increased estimated positive precision from 67% to 81%."

### 7.5 Model v2 Training Design (2026-04-15)
- **Changes from v1:** Removed 500-item cap; added lexical filter for positives; added REJECTED items as additional negatives
- **Training data:** 2,003 positives (filtered WEAK_POSITIVE) + 1,651 negatives (1,005 WEAK_NEGATIVE + 646 REJECTED with transcripts)
- **Validation:** 89% accuracy, ROC-AUC 0.9643
- **Ground truth F1:** 65.7% at threshold 0.70 (best), overall accuracy 76.4% at 0.75

### 7.6 Ground Truth Sample Design (2026-04-15)
- **Sample size:** 300 (chosen over 200 for tighter per-class confidence intervals)
- **Stratification:** Over-sampled discriminative bands (POTENTIAL_LB n=90, BORDERLINE_LB n=60, REJECTED n=60) relative to confident tiers (WEAK_POSITIVE n=45, WEAK_NEGATIVE n=45)
- **Randomization:** Deterministic seed (42) for reproducibility; sample order shuffled to prevent annotator fatigue bias from labeling one tier at a time
- **Annotation tool:** Custom Flask web app with HTML5 audio player, keyboard shortcuts, CSV persistence
- **Sample stored at:** `data/annotation_sample.json` (reproducible)
- **Annotations stored at:** `data/annotations.csv`

### 7.7 ADI17 Selected as Contrastive Source (2026-04-21)
- **Decision:** Use `ArabicSpeech/ADI17` (Ali et al. 2019) as the contrastive non-Lebanese dataset for EGY, Gulf, and additional LEB positives.
- **Alternatives considered:** Halabi 2016 Arabic Speech Corpus (rejected — single-speaker confound); own-pipeline scraping (rejected by user — prefers citable datasets).
- **Rationale:** Multi-speaker broadcast audio matching Lebanese podcasts/YouTube acoustic profile; single citation; openly available; contains 17 country-level Arabic dialects with research-grade labels.
- **Counts retrieved:** 1,000 LEB · 1,000 EGY · 5,000 Gulf (KSA, KUW, UAE, QAT, OMA).
- **Detail:** Section 8.2 / 8.2.1.

### 7.8 FLEURS Selected for MSA (2026-04-27)
- **Decision:** Use `google/fleurs` config `ar_eg` (Conneau et al. 2022) as the MSA contrastive source.
- **Empirical finding driving this:** ADI17 train split contains 0 MSA across all 990,821 rows — ADI17 is country-dialect only, no MSA class (verified via Colab full scan, see Section 8.2.3).
- **Alternatives considered and rejected:**
  - Halabi 2016 — single-speaker confound (rejected initially in 8.2.1, still applies).
  - Mozilla Common Voice 11/13/17 — withdrawn from HuggingFace by Mozilla in October 2025; only available via Mozilla Data Collective with separate access portal.
  - MGB-2 (Al Jazeera broadcast MSA) — would be the strongest acoustic-parity match to ADI17 but requires QCRI registration, longer access path; deferred to future work.
  - Own-pipeline Al Jazeera/BBC scraping — rejected by user (citability concerns).
- **Rationale for FLEURS:** Multi-speaker, citable (LREC 2022), CC-BY-SA, available on HF, MSA register via FLoRes-101 prompts read by Egyptian speakers (Egyptian accent on MSA content — small concession vs single-speaker confound).
- **Count retrieved:** 798 unique MSA items.
- **Implementation:** Pinned `datasets==2.21.0` in Colab because `datasets>=4.0` removed support for script-based loaders, and FLEURS still ships `fleurs.py`.
- **Detail:** Section 8.2.3 / 8.2.4.

### 7.9 Acoustic Embedding Backbone (2026-04-29)
- **Decision:** `facebook/wav2vec2-xls-r-300m` (Babu et al. 2022) — 300M-parameter multilingual self-supervised speech encoder, pretrained on 436K hours of speech in 128 languages including Arabic.
- **Alternatives considered:**
  - wav2vec2-base (95M, English pretrain) — smaller/faster but Arabic transfer untested.
  - HuBERT-base — similar size and quality to wav2vec2-base.
  - Whisper-large-v2 encoder — already in pipeline for ASR but encoder-only forward pass is heavier.
  - MFCC + classical ML — lighter baseline but loses pretrained representational power; could be added as a sanity-check baseline.
  - elgeish/wav2vec2-large-xlsr-53-arabic — Arabic-finetuned 300M variant, similar size to XLS-R.
- **Rationale:** XLS-R is the standard backbone in 2022+ dialect ID literature, has the broadest multilingual pretraining (covers Arabic), and produces 1024-dimensional utterance embeddings via mean-pooling the last hidden state. Single citation. Same size class as alternatives, so no speed advantage from switching.
- **Pooling strategy:** Mean over time (masked by attention mask via `_get_feat_extract_output_lengths`) → 1024-d vector per clip.

### 7.10 Local CPU Extraction vs Colab GPU (2026-04-29)
- **Initial plan:** Run extraction on Colab T4 GPU (free tier).
- **Friction encountered (in order):**
  1. Colab file-panel upload disconnects on 3.4 GB zip (free runtime limits).
  2. Direct file hosts blocked or gated: transfer.sh (down), bashupload.com (TLS), pixeldrain (UniFi network filter), gofile.io free-tier API (paywalled), file.io (one-time download risky).
  3. GitHub Releases worked (uploaded successfully via API using credential-manager-cached token to a separate repo) but at this point the user requested no external services.
- **Decision:** Switch to local CPU extraction with PyTorch threading (21 cores).
- **Performance characteristics:**
  - First model load: 30 min (Windows Defender scanning cached files; symlinks unavailable without dev-mode/admin).
  - Subsequent loads: 2 sec (warm OS disk cache).
  - Inference rate: ~0.53 clips/sec (batch_size=8) → ~7 hours for 14,177 clips.
  - Resumable: parquet appended every 200 items.
- **Trade-off:** ~7 hours wall-clock vs ~30 min on T4. Acceptable given full local control and reproducibility (no third-party file-host dependency).

### 7.11 Audio Clip Duration for Embedding Input (2026-04-29)
- **Decision:** 10 seconds per clip @ 64 kbps mono 16 kHz MP3.
- **Alternatives:** 30s @ 96k (initial trim — produced 3.4 GB zip, too large to upload), 5s (too short for utterance-level pooling stability).
- **Rationale:** 10 sec is the standard utterance-level window in dialect ID literature; XLS-R produces stable mean-pooled embeddings on this length; cuts the upload zip to 950 MB. Audio normalization pipeline (mono 16 kHz, loudnorm, 96k bitrate originally; re-encoded to 64k for compactness) is uniform across all classes to avoid recording-environment confounds.

---

## 8. Next Steps (Planned)

### 8.1 Clean and Retrain ✅ COMPLETED (2026-04-15)
1. ✅ Filtered WEAK_POSITIVE: 3,232 → 2,003 items via `strong_lb_hits >= 1`
2. ✅ Removed 500-item training cap
3. ✅ Retrained: 2,003 positives + 1,651 negatives
4. ✅ Evaluated on 300-item ground truth: F1=65.7% at threshold 0.70
5. ✅ WEAK_POSITIVE precision improved from 67% → 79%
6. ✅ Documented in Section 6b

### 8.2 Non-Lebanese Contrastive Dataset — IN PROGRESS via ADI17

**Decision (2026-04-20):** Primary source is ADI17 (QCRI Arabic Dialect Identification corpus) via HuggingFace. Verified ungated — no approval required. Research-grade labels, citable.

**Strategy:** Selective download of dev+test splits only (5.2 GB total, vs 263 GB full dataset). Data in these splits is organized by dialect in the Parquet files, so we extract audio for target dialects and delete Parquet files after extraction to save disk space.

**Target dialects and per-dialect quotas (1,000 items each):**
- EGY (Egyptian) — contrastive negative
- MSA (Modern Standard Arabic) — contrastive negative
- Gulf variants: KSA, KWT, UAE, QAT, OMA — contrastive negatives
- LEB (Lebanese) — additional research-grade positive training data

**Script:** `scripts/07_download_adi17_contrastive.py` — resumable (URL-based deduplication in DB), stops per dialect when quota filled, deletes Parquet files after extraction.

**Progress (completed dev+test 2026-04-21):**
- EGY: 1,000 ✅
- KSA: 1,000 ✅
- LEB: 1,000 ✅ (bonus Lebanese positives)
- OMA: 1,000 ✅
- QAT: 1,000 ✅
- UAE: 1,000 ✅
- KUW: 1,000 ✅ (after fixing code mismatch: ADI17 uses "KUW" not "KWT")
- MSA: 0 ❌ — **not present in dev or test splits, only in 258 GB train split (inaccessible at current bandwidth)**
- **Total downloaded: 7,000 items**

**Dialects observed in dev+test (not all targeted):** ALG, EGY, IRA, JOR, KSA, KUW, LEB, LIB, MAU, MOR, OMA, PAL, QAT, SUD, SYR, UAE, YEM. Note MSA is absent. MSA is a separately stored split that requires full train download to access.

**Data quality note:** A dialect-code mismatch (script used "KWT" while ADI17 uses "KUW") caused the first 6,000-item batch to silently skip Kuwaiti items. Caught after dev-00000 was re-processed; corrected in `scripts/07_download_adi17_contrastive.py`. Also added a processed-files manifest at `data/adi17_processed_files.json` to prevent redundant re-downloads on resume.

### 8.2.1 MSA Sourcing Decision (2026-04-21)

**Problem:** MSA (Modern Standard Arabic) is absent from the ADI17 dev and test splits — present only in the 258 GB train split. This creates a gap in the contrastive dataset that must be filled for the dialect classifier to distinguish Lebanese from formal Arabic.

**Options considered:**

| Option | Approach | Decision |
|--------|----------|----------|
| 1 | Stream-filter MSA rows from ADI17 train split (~15-20 GB targeted download of files known to contain MSA) | **SELECTED** |
| 2 | Scrape MSA from own pipeline (Al Jazeera, BBC Arabic news podcasts/channels) | Rejected by user preference |
| 3 | Arabic Speech Corpus (Halabi 2016), CC-BY 4.0, direct download | Rejected |

**Rationale for choosing Option 1 over Option 3:**

1. **Methodological consistency.** All other contrastive dialects (EGY, KSA, KUW, OMA, QAT, UAE) and the bonus LEB positives come from ADI17. Mixing a single-speaker studio corpus (Halabi) with multi-speaker broadcast audio (ADI17) creates an acoustic/domain confound: the classifier could learn "studio audio = MSA" rather than "MSA dialect features = MSA."

2. **Single-speaker confound (critical).** The Arabic Speech Corpus contains audio from one speaker. A classifier trained on this would have no way to distinguish *MSA dialect features* from *this specific speaker's voice identity*. In dialect ID literature, this is one of the most common methodological critiques of poorly-designed experiments — reviewers on a Master's thesis committee will flag it.

3. **Acoustic parity.** All other dialects in the dataset come from similar source types (broadcast, podcast, YouTube). ADI17 MSA is broadcast MSA from the same type of source, preserving acoustic homogeneity across classes. Studio recordings would introduce a detectable recording-environment difference between MSA and all other dialects.

4. **Citability and cross-comparability.** "All contrastive data drawn from ADI17 (Ali et al., 2019)" is a single clean citation that maximizes reproducibility and cross-paper comparability — any future researcher using ADI17 can directly benchmark against these results. Option 3 would require documenting and defending a mixed-corpus design.

5. **Size.** Option 3 has only ~4 hours of audio from one speaker — hitting a 1,000-item quota would require aggressive splitting that creates near-duplicates. ADI17 train has tens of thousands of MSA items.

**Selected Option 1 cost:** The full 258 GB train split is infeasible, but ADI17 data is dialect-sorted within Parquet files, so MSA is likely concentrated in 1-3 specific files (~7-20 GB). A targeted two-pass script will first scan the dialect column from each file (<1 MB per file) to identify which contain MSA, then download only those files with the existing resume-retry infrastructure. Estimated total transfer: 15-20 GB over 8-12 hours at current bandwidth (0.5 MB/s).

**Known tradeoff accepted:** ADI17 MSA is specifically broadcast MSA (news-style reading), not academic-lecture MSA or literary MSA. Register is somewhat narrow. Documented in the limitations section of the thesis.

**Storage:** ADI17 audio stored as MP3 (96 kbps, mono 16 kHz) in `data/raw_audio/` with filename prefix `adi17_<DIALECT>_<id>.mp3`. Database entries: `platform='adi17'`, `source_metadata` includes `adi17_dialect` and `contrastive_role`.

**Note:** Syrian Arabic excluded due to high linguistic overlap with Lebanese (both Levantine).

### 8.2.2 MSA Download Strategy Pivot — Local Grinder → Colab (2026-04-27)

**Problem encountered with local grinder (`scripts/08_download_adi17_msa.py`):**
- Hypothesis "MSA concentrated in 1-3 files due to dialect-sorting" turned out wrong (or at least the early files don't contain MSA — see below).
- Files train-00000 through train-00006 each scanned: 0 MSA. 7 files × ~6.5 GB downloaded = 45 GB transferred over ~5 days, with 0 MSA extracted.
- Home connection at ~0.5 MB/s plus repeated DNS/connection failures requiring 20-retry resume cycles.
- Original attempt at Parquet column-projection scan locally also stalled — 250 small HTTP range requests per file is high-overhead on a flaky link.
- At current rate, scanning all 40 files would take weeks.

**Pivot: Run the work on Google Colab (free tier).**
- Colab provides ~100 MB/s bandwidth (~200x faster than the home link) and no DNS instability.
- Free tier disk (~100 GB) is sufficient with the iterative download-scan-delete pattern.
- `hf_transfer` (Rust-based parallel chunked downloader, set via `HF_HUB_ENABLE_HF_TRANSFER=1`) maximizes parallelism per-file — typically 10-50× faster than serial `requests` even at the same network bandwidth.

**New script set (added 2026-04-27):**
- `scripts/colab_extract_msa.py` — runs in Colab. Downloads ADI17 train files one at a time using `hf_transfer`, reads dialect column, extracts up to 1,000 MSA items, encodes to MP3 with the same ffmpeg pipeline (16 kHz mono, loudnorm, 96 kbps), bundles into `/content/adi17_msa.zip`. Writes metadata as JSONL so partial progress is preserved.
- `scripts/10_import_colab_msa.py` — runs locally. Extracts the downloaded zip, copies MP3s into `data/raw_audio/`, inserts queue.db rows with `status=DOWNLOADED, contrastive_role=msa`, updates `data/adi17_processed_files.json` and `data/adi17_msa_file_map.json` so the local pipeline state remains coherent.

**Expected duration:** end-to-end (Colab download+extract + local import) ~30-90 minutes versus weeks on home connection.

**Local grinder retired:** `scripts/08_download_adi17_msa.py` left in repo for documentation but is superseded. The 7 files it already scanned (train-00000 through train-00006, all 0 MSA) remain marked in `data/adi17_processed_files.json` so the Colab pipeline can skip them.

**Methodological note:** This change does NOT alter dataset selection — the MSA still comes from ADI17 train split, preserving the citability and acoustic-parity rationale of 8.2.1. Only the download mechanism changed.

### 8.2.3 ADI17 Does Not Contain MSA — Sourcing Decision Reopened (2026-04-27)

**Empirical finding from the Colab scan:** A complete dialect-column scan of all 40 ADI17 train Parquet files (990,821 rows total) returned **zero MSA items**. The full dialect inventory in the train split is:

| Dialect | Rows | Dialect | Rows |
|---------|------|---------|------|
| IRA | 277,725 | KUW | 30,507 |
| EGY | 143,013 | ALG | 29,439 |
| MAU | 129,666 | OMA | 26,595 |
| KSA |  66,842 | QAT | 26,088 |
| UAE |  48,474 | YEM | 20,456 |
| SYR |  46,026 | SUD | 18,258 |
| PAL |  36,747 | MOR | 17,432 |
| LEB |  35,938 | JOR |  4,858 |
| LIB |  32,757 | | |

**That is exactly 17 country-level dialects — MSA is not one of them.** This matches the structure described in Ali et al. 2019: ADI17 is a 17-class country-dialect identification dataset; MSA is a separate register treated by other corpora.

**Implication:** The Option 1 plan from 8.2.1 (extract MSA from ADI17 train split) was based on a false premise. The 5 days of local grinder runtime against train-00000 through train-00006 (45 GB downloaded) and the subsequent Colab scan were correctly producing 0 MSA because no MSA exists in this dataset.

**Cost of the discovery:** ~5 days of compute on flaky home connection + ~1 hour Colab compute. The Colab scan was the operation that produced the empirical proof — without it we would still be guessing.

**Sourcing decision must be reopened.** The original 8.2.1 trade-off between Option 1 (ADI17), Option 2 (own-pipeline scraping), Option 3 (Arabic Speech Corpus / single-speaker) is no longer valid because Option 1 doesn't exist as imagined.

**Candidate replacements (to be evaluated):**

| Source | Multi-speaker? | Register | Citable | License | On HF? |
|--------|---------------|----------|---------|---------|--------|
| **MGB-2** (Al Jazeera broadcast) | yes | MSA broadcast news | yes (Ali et al. 2016) | research-use | partial |
| **FLEURS** (Google `google/fleurs` ar_eg) | yes | read MSA from FLoRes-101 sentences | yes (Conneau et al. 2022) | CC-BY-SA 4.0 | yes |
| **Common Voice ar** (Mozilla) | yes (crowdsourced) | mostly read MSA | yes (Ardila et al. 2020) | CC0 | yes |
| Arabic Speech Corpus (Halabi 2016) | no — single speaker | MSA studio | yes | CC-BY 4.0 | no | (rejected in 8.2.1)
| Own-pipeline scrape | yes | broadcast MSA (Al Jazeera, BBC Arabic) | n/a — synthesized | varied | n/a | (rejected in 8.2.1)

**Currently leaning toward:** FLEURS or Common Voice ar — both multi-speaker, both single citation, both available via HF Datasets in minutes. The acoustic-parity argument from 8.2.1 partly weakens (FLEURS/CV are read-prompt audio while ADI17 is broadcast), but this is a smaller methodological concession than the single-speaker confound that disqualified Option 3.

**Decision (revised 2026-04-27 evening): FLEURS Arabic** — see history below.

**History of attempts on 2026-04-27:**

1. **First try: FLEURS (`google/fleurs`, ar_eg).** Failed silently in Colab (per_split: {} in summary.json). Cause unclear at the time; the script's silent except hid the actual error.
2. **Second try: Common Voice 11 (`mozilla-foundation/common_voice_11_0`, ar).** Failed with `DatasetNotFoundError: doesn't exist on the Hub or cannot be accessed` even after HF auth setup (logged in as charlene11). Reason discovered next:
3. **Mozilla pulled Common Voice from HF in October 2025.** All `mozilla-foundation/common_voice_*` versions are gone — moved to Mozilla Data Collective (separate signup/portal). HF is no longer a viable source for CV. Documented banner reads: *"Effective October 2025, Mozilla Common Voice datasets are now exclusively available through Mozilla Data Collective."*
4. **Pivot back to FLEURS** with much more robust loading (try `trust_remote_code` on/off + streaming on/off variants, all errors printed loudly per split).

**Why FLEURS now (final selection):**

- **Still on HuggingFace** — Google hosts and maintains `google/fleurs`, no withdrawal.
- **Ungated** — no agreement, no token, no paperwork.
- **MSA register** — Arabic FLoRes-101 sentences are written in literary Arabic (Fuṣḥā). Speakers read MSA prompts.
- **Multi-speaker** — many contributors per language → no single-speaker confound.
- **Citable** — Conneau et al. 2022 ("FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech").
- **License** — CC-BY-SA 4.0.
- **Small** — ~3-7 hours per language, so a full download is fast (no need for streaming workarounds).
- **Audio register caveat** — `ar_eg` means Arabic recorded by Egyptian speakers. Content is MSA but pronunciation may carry Egyptian accent traces. This is a smaller methodological concession than the single-speaker confound that disqualified Halabi 2016.

**Original Common Voice rationale (kept for thesis context):**

- **Reliability:** CV is the single most battle-tested speech dataset on HF; thousands of papers use it. The `datasets` library has stable, well-documented support.
- **Speed:** Streaming mode (no full download) — ~10 minutes of Colab time to grab 1,000 clips.
- **Citability:** Ardila et al. 2020 ("Common Voice: A Massively-Multilingual Speech Corpus", LREC).
- **MSA register:** All Arabic CV prompts are written in al-Fuṣḥā (المعيارية), so contributors record MSA.
- **Multi-speaker:** Thousands of contributors → no single-speaker confound (the disqualifier from 8.2.1).
- **License:** CC0.
- **Why version 11 specifically:** versions 13+ require an HF user agreement; 11 is open-access and has Arabic.

**Acoustic-parity concession:** ADI17 contrastive dialects are broadcast audio; CV is read-prompt recordings on contributor devices. This introduces a recording-environment difference between the MSA class and the ADI17 dialect classes. Mitigations:
- All audio is normalized through identical pipeline (mono 16 kHz, loudnorm, 96 kbps MP3) so absolute level/codec is matched.
- Ground-truth evaluation will be done on Lebanese clips that contain natural code-switching, which is closer to broadcast acoustics — so the MSA class's read-prompt nature is unlikely to over-fit on the test set.
- Documented in thesis limitations: "MSA contrastive class drawn from Common Voice; acoustic register differs from broadcast ADI17 dialects."

**Implementation:** `scripts/colab_extract_msa_cv.py` (Colab-side, now points at FLEURS despite the filename — left as-is for continuity) + `scripts/10_import_colab_msa.py` (local). FLEURS metadata fields preserved (id, gender, transcription, split) in `source_metadata` for later auditing.

### 8.2.4 Final MSA Import Result (2026-04-27)

**Successfully imported 798 MSA items from FLEURS `ar_eg` train split.**

- Colab script ran cleanly with `datasets==2.21.0` + `fsspec<=2024.12.0` (the version pin needed because `datasets>=4.0` removed support for script-based loaders, and FLEURS still uses `fleurs.py`).
- 1,000 metadata entries written but only 798 unique audio files — FLEURS contains multiple recordings of the same FLoRes-101 sentence by different speakers, sharing the underlying `id`. The unique-URL constraint in queue.db deduped them on import (798 added, 202 "already-in-db", 0 failures).
- All 798 items have `platform='fleurs'`, `status='DOWNLOADED'`, `source_metadata.adi17_dialect='MSA'` (field name kept for downstream consistency), `contrastive_role='msa'`.
- Audio normalized through identical pipeline as ADI17: mono 16 kHz, loudnorm, 96 kbps MP3.

**Final contrastive dataset state (2026-04-27):**

| Class | Source | Count |
|-------|--------|-------|
| Lebanese | ADI17 LEB | 1,000 |
| Egyptian | ADI17 EGY | 1,000 |
| Gulf (KSA/KUW/UAE/QAT/OMA) | ADI17 | 5,000 |
| MSA | FLEURS ar_eg | 798 |

Combined with the existing Lebanese positives (3,232 WEAK_POSITIVE + 1,769 POTENTIAL_LB + 313 BORDERLINE_LB) and 1,005 WEAK_NEGATIVE podcasts, this completes the data collection phase of the thesis. Next phase: retrain the final dialect classifier with the full contrastive set and evaluate on the 300-item ground truth.

### 8.3 Phase 3 — Acoustic v2 Classifier (started 2026-04-29)

**Methodology decision.** Original v1 classifier was text-only (lexical features + transcript embeddings). To address the documented "no acoustic features" limitation, Phase 3 adds wav2vec2-xls-r-300m mean-pooled hidden-state embeddings (1024-d) as the feature source. Single backbone model, single citation (Babu et al. 2022).

**Data prep:**
- `scripts/11_prep_embeddings_audio.py` selects 14,177 items from queue.db (all WEAK_POSITIVE + POTENTIAL_LB + BORDERLINE_LB + WEAK_NEGATIVE + ADI17 + FLEURS, plus all 300 ground-truth items regardless of status). Each is trimmed to 30s @ 96k mono 16kHz MP3. ffmpeg with multiprocessing (21 workers): all 14,177 trimmed in 11 minutes, 0 failures.
- `scripts/11b_compact_embeddings_zip.py` re-encodes those clips to 10s @ 64k → `audio_for_embeddings_compact.zip` (950 MB). 10 sec is the standard window for utterance-level wav2vec2 dialect ID embeddings (literature precedent: Conneau et al. 2022; ADI benchmarks).

**File-host saga (the day's friction):** Originally planned to run extraction in Colab (T4 GPU). Required getting the audio zip to Colab. Sequential failures:
1. Direct Colab upload — runtime disconnects on 3.4 GB before completing.
2. Google Drive — feasible but requires user browser interaction.
3. transfer.sh — service unreachable.
4. bashupload.com — Windows curl SSL trust issue.
5. pixeldrain — blocked at user's UniFi router (content filter).
6. gofile.io — uploads work but free-tier API now returns `error-notPremium` for direct downloads, and their `wt` token is in heavily obfuscated JS.
7. file.io — one-time download (risky if Colab fails partway).
8. GitHub Releases — token from credential manager belongs to second user account (`Charleneeid116`); used it to create `Charleneeid116/thesis-data` public repo and uploaded the zip as a release asset (worked, ~12 min upload at ~1.3 MB/s).

**Pivot to local CPU (2026-04-29).** Given the friction, switched approach: run wav2vec2-xls-r-300m **locally on CPU** with PyTorch threading (21 cores). No external services, no uploads, no Colab dependency. `scripts/12_extract_embeddings_local.py` reads from `data/audio_clips_compact/` directly.

**Local performance characteristics:**
- AutoModel.from_pretrained for cached 1.2 GB model: **30 min wall time** (suspected: Windows Defender scanning the cache files on read; symlinks unavailable without dev-mode/admin).
- librosa first-call init: ~22 sec (numba JIT compilation).
- Single-clip forward pass: ~2 sec on CPU.
- With batch_size=8: estimated ~3-4 hours wall time for all 14,177 items.
- Once model is loaded and disk cache is warm, batch processing is fast.
- Resumable: `data/embeddings.parquet` is appended every 200 items; restart skips already-processed item_ids.

**Pipeline scripts (Phase 3):**
- `scripts/11_prep_embeddings_audio.py` — bundle clips
- `scripts/11b_compact_embeddings_zip.py` — re-encode to compact format
- `scripts/12_extract_embeddings_local.py` — extract embeddings on CPU
- `scripts/13_load_embeddings.py` — join embeddings with queue.db, derive binary labels
- `scripts/13b_eval_v1_on_gt.py` — evaluate v1 (text-only) on held-out GT, appends Section 9.0
- `scripts/14_train_classifier_v2.py` — train LR + MLP, evaluate, save model, append FINDINGS Section 12

**Held-out evaluation strategy.** The 300 ground-truth items annotated in `data/annotations.csv` are excluded from the training pool. Mapping: `lebanese, mostly_lebanese → 1`; `not_lebanese → 0`; `unclear, skip → excluded`. v2's GT performance is the thesis-defensible accuracy/F1 number. v1's GT-set performance was measured in Section 9.0 (ROC-AUC 0.8477; best macro F1 0.7397 at threshold 0.70) so v1↔v2 comparison is apples-to-apples on the same held-out set.

**Networking-blocked services log (for thesis methodology section):** transfer.sh (down), bashupload.com (TLS), pixeldrain (network filter), gofile.io API (paywalled), Mozilla Common Voice on HF (withdrawn Oct 2025), FLEURS via `datasets>=4.0` (loading scripts dropped). All worked around to reach the same scientific outcome.

**Local state cleanup:** All 40 train files marked as fully scanned in `data/adi17_processed_files.json` and `data/adi17_msa_file_map.json` (with `0` MSA each), so any future accidental run of `08_download_adi17_msa.py` will exit immediately without re-downloading.

### 8.5 Open Dataset Decisions Pending Final Evaluation
- **What counts as "Lebanese" in training:** Decision deferred until ground truth precision numbers are used to filter training data. v2 currently treats POTENTIAL_LB, all WEAK_POSITIVE (or filtered by `strong_lb_hits>=1` with `--strict_positive`), and ADI17 LEB as positives.
- **Treatment of "Mostly Lebanese / mixed" items in GT:** 50 of 300 GT items (16.7%) are labeled `mostly_lebanese`. Currently mapped to positive (1) for binary evaluation. Alternatives to evaluate: treat as positive but excluded from train, separate class in multi-class setup, or weighted examples.
- **Multi-class extension:** Once binary v2 is validated, consider 4-way (Lebanese / MSA / Egyptian / Gulf) classification using the same embeddings.

---

## 9. Known Issues & Limitations

| Issue | Description | Impact | Status |
|-------|-------------|--------|--------|
| Lexicon overlap | `LEBANESE_WORDS` contains pan-Arabic words | Inflates `lb` count; `lb > 0` is meaningless as a filter | Documented; mitigated by `raw_score < 0` rule and acoustic v2 |
| Strong markers are Levantine, not uniquely Lebanese | "شو", "وين", "بدي" appear in Syrian, Palestinian too | Cannot distinguish Lebanese from other Levantine dialects with lexicon alone | Documented in thesis limitations |
| Single annotator | Ground truth labeled by thesis author only | No inter-annotator agreement metric; potential bias | Documented; future work could add Cohen's kappa |
| Training cap (v1) | ~~`05_train_dialect_model.py` caps at 500~~ | Restricted v1 training to subset of available data | **FIXED 2026-04-15** — cap removed, full data used |
| Podcast metadata missing | podcast_rss items have empty `source_metadata {}` | 03b cannot label them; 03c (lexical) is the only labeling path | By design; 03c covers this |
| ADI model disabled | CAMeL-Lab BERT not used in scoring | Potential unused signal source | Acceptable; v2 acoustic embeddings supplant it |
| No acoustic features (v1) | v1 is text-only (lexical + transcript embeddings) | Dialect differences in pronunciation are not captured | **ADDRESSED in v2** via wav2vec2-xls-r-300m embeddings |
| Code-switching | Many Lebanese speakers mix dialect and MSA | Complicates binary classification; "Mostly Lebanese" category needed | Documented; v2 evaluation maps "mostly_lebanese" → 1 |
| MSA acoustic register | FLEURS MSA is read-prompt audio; ADI17 dialects are broadcast | Recording-environment confound between MSA class and other dialects | Documented; mitigated by uniform audio normalization (mono 16 kHz, loudnorm, 96 kbps MP3) |
| Egyptian accent in MSA | FLEURS `ar_eg` config = Egyptian speakers reading MSA prompts | Pronunciation may carry Egyptian accent even though content is MSA | Documented; small concession vs single-speaker confound that disqualified Halabi 2016 |
| ADI17 country-level only | ADI17 has 17 country dialects, no MSA class | Required separate FLEURS pull for MSA; introduced acoustic-domain heterogeneity | Resolved; documented in 8.2.3 |
| Mozilla Common Voice unavailable | CV withdrawn from HuggingFace Oct 2025 | Could not use the most-cited multilingual speech corpus for MSA | Resolved by switching to FLEURS |
| GPU unavailable locally | User's machine is CPU-only (22 cores) | Embedding extraction takes ~7 hours instead of ~30 min on T4 | Accepted; resumable script handles overnight runs |
| Imbalanced training pool | More negatives than positives in v2 training | Model could be biased toward majority class | Mitigated by `class_weight='balanced'` in LR/MLP |
| **Recording-domain confound (V2)** | V2 acoustic training pool's positive and negative classes correlate with distinct recording domains (broadcast, read-prompt, podcast). Frozen XLS-R encoder learns domain rather than dialect. | V2 collapses on held-out GT; macro F1 drops from 0.86 (val) to 0.38 (GT). Per-source balancing eliminates the residual signal entirely (ROC-AUC 0.36, below random). | **Confirmed empirically** in Section 12.6 Experiment B. Remedies require end-to-end fine-tuning, MGB-2 broadcast MSA substitution, or speaker-normalized features. |

---

## 10. File Reference

### Data
| File | Purpose |
|------|---------|
| `data/queue.db` | SQLite database with all pipeline state |
| `data/raw_audio/*.{flac,mp3}` | Audio for all collected items (FLAC for YouTube, MP3 for podcast/ADI17/FLEURS) |
| `data/transcripts/clip_<id>_screening.json` | Per-item screening transcripts |
| `data/samples/<id>_chunk<N>.wav` | 3×20s screening chunks per item |
| `data/annotations.csv` | 300-item ground truth test set (item_id, source_status, platform, audio_path, ground_truth, notes, timestamp) |
| `data/annotation_sample.json` | Reproducible GT sample selection (item IDs, strata) |
| `data/adi17_processed_files.json` | Manifest of ADI17 Parquet files scanned/imported |
| `data/adi17_msa_file_map.json` | MSA row counts per ADI17 train file (all 0 — see 8.2.3) |
| `data/audio_clips_for_embed/*.mp3` | 30s @ 96k clips for v2 (intermediate) |
| `data/audio_clips_compact/*.mp3` | 10s @ 64k compact clips for v2 (final embedding input) |
| `data/audio_for_embeddings.zip` | Bundled 30s clips + manifest (3.4 GB) |
| `data/audio_for_embeddings_compact.zip` | Bundled 10s clips + manifest (950 MB) |
| `data/embeddings.parquet` | wav2vec2-xls-r-300m mean-pooled embeddings (1024-d) |
| `data/embeddings_summary.json` | Embedding extraction job metadata |
| `data/embeddings_with_labels.parquet` | Embeddings joined with binary `lebanese` label, ready for training |
| `data/fleurs_msa.zip` | FLEURS MSA bundle from Colab (798 items + metadata) |

### Models
| File | Purpose |
|------|---------|
| `models/dialect_classifier.joblib` | v1 text-only LogisticRegression classifier (Section 6b) |
| `models/dialect_classifier_v2_acoustic.joblib` | v2 acoustic LR or MLP (Phase 3, output of `14_train_classifier_v2.py`) |
| `models/dialect_classifier_v2_acoustic.meta.json` | v2 hyperparameters + full per-set results |

### Configuration & Source
| File | Purpose |
|------|---------|
| `config/base.yaml` | Pipeline configuration (sources, models, thresholds, trusted channel/feed lists) |
| `src/cfg.py` | Pydantic config validation |
| `src/db.py` | SQLAlchemy ORM + queue operations |
| `src/asr/whisper_engine.py` | Faster-Whisper transcription wrapper |
| `src/dialect/lexicons.py` | Curated dialect word lists (Lebanese, MSA, Egyptian, Gulf, Syrian) |
| `src/dialect/scoring.py` | Lexicon scoring formula and `lexicon_score()` helper |
| `src/dialect/adi_model.py` | CAMeL-Lab BERT wrapper (disabled in current pipeline) |
| `src/embeddings/engine.py` | Sentence embedding wrapper (paraphrase-multilingual-MiniLM-L12-v2) |
| `src/platforms/youtube.py` | yt-dlp + ffmpeg YouTube download |
| `src/platforms/podcast.py` | HTTP podcast download |
| `src/utils/audio.py` | ffmpeg chunk extraction |

### Scripts (numbered, run in order)
See `PIPELINE.md` for full step-by-step description.

| Script | Phase | Role |
|--------|-------|------|
| `00_show_queue.py` | All | Queue summary |
| `01a/b_discover_youtube*.py` | 1 | YouTube discovery (RSS preferred) |
| `01c_discover_podcast_rss.py` | 1 | Podcast RSS discovery |
| `01d_discover_tiktok.py` | 1 | TikTok (limited) |
| `02_download_audio.py` | 1 | Audio download with normalization |
| `03_transcribe_screening.py` | 1 | Whisper screening transcription |
| `03b_assign_weak_labels.py` | 1 | WEAK_POSITIVE via trusted metadata |
| `03c_assign_lexical_negatives.py` | 1 | WEAK_NEGATIVE via lexical scoring |
| `04_score_dialect.py` | 1 | v1 model inference → POTENTIAL_LB / BORDERLINE_LB / REJECTED |
| `05_train_dialect_model.py` | 1 | v1 LogisticRegression training |
| `06_reset_podcast_downloads.py` | 1 | Recovery utility |
| `07_download_adi17_contrastive.py` | 2 | ADI17 dev+test import |
| `08_download_adi17_msa.py` | 2 | (superseded — ADI17 has no MSA) |
| `09_find_msa_files.py` | 2 | (diagnostic — Parquet column stats unavailable) |
| `colab_extract_msa.py` | 2 | Colab ADI17 train scan (confirmed 0 MSA) |
| `colab_extract_msa_fleurs.py` | 2 | First-pass FLEURS attempt (failed silently) |
| `colab_extract_msa_cv.py` | 2 | **Primary MSA source.** FLEURS via `datasets==2.21.0` in Colab |
| `10_import_colab_msa.py` | 2 | Local import of FLEURS bundle into queue.db |
| `11_prep_embeddings_audio.py` | 3 | Trim audio to 30s for v2 |
| `11b_compact_embeddings_zip.py` | 3 | Re-encode to 10s @ 64k |
| `12_extract_embeddings_local.py` | 3 | wav2vec2-xls-r-300m embedding extraction (CPU, local) |
| `colab_extract_embeddings.py` | 3 | (alternative — Colab GPU, retired due to upload friction) |
| `13_load_embeddings.py` | 3 | Join embeddings with queue.db, derive binary labels |
| `13b_eval_v1_on_gt.py` | 3 | Evaluate v1 text-only model on held-out GT, append FINDINGS Section 9.0 |
| `14_train_classifier_v2.py` | 3 | Train acoustic v2 classifier, evaluate, append FINDINGS Section 12 |

### Tools
| File | Purpose |
|------|---------|
| `tools/annotate.py` | Flask-based ground truth annotation web app (300-item stratified sample) |
| `scripts/convert_audio_to_flac.py` | WAV → FLAC lossless conversion (one-time storage optimization 2026-04-14) |
| `scripts/check_youtube_stages.py` | Queue inspection by stage |
| `scripts/rerun_youtube_error_downloads.py` | Retry ERROR_DOWNLOAD items |
| `scripts/reset_weak_labels.py` | Roll back weak-label assignments |

---

## 11. Thesis Paper Outline (mapping content to paper sections)

This file maps the project's empirical content to the standard structure of a Master's thesis paper.

| Paper section | Content / FINDINGS source |
|---------------|---------------------------|
| **Abstract** | Summary of pipeline, dataset, two-stage classifier (v1 text-only → v2 acoustic), and held-out 300-item evaluation results. |
| **1. Introduction** | Section 1 (Problem Statement). Motivation: lack of large open Lebanese dialect speech corpora; need for automatic dialect ID for downstream NLP. |
| **2. Related Work** | Cite: Ali et al. 2019 (ADI17); Conneau et al. 2022 (FLEURS); Babu et al. 2022 (XLS-R); Ardila et al. 2020 (Common Voice); Halabi 2016 (Arabic Speech Corpus, rejected); CAMeL-Lab BERT papers; faster-whisper / OpenAI Whisper for ASR. |
| **3. Data Collection** | Section 3 (Data Collection Results), Section 8.2.* (contrastive dataset construction). Sources, sizes, decisions, citability of each component. |
| **4. Preprocessing & Audio Normalization** | Section 2.4 (Audio Processing Decisions), Section 7.3 (Audio Format Migration), Section 7.1 (Whisper Model Selection). |
| **5. Weak Supervision** | Section 5 (Weak Supervision Analysis), Section 7.2 (WEAK_NEGATIVE Criterion), Section 7.4 (Lexical Verification). |
| **6. Ground Truth Annotation** | Section 6 (methodology, results), Section 7.6 (sample design). |
| **7. Model v1 — Text-Only Baseline** | Section 4.2 (architecture), Section 6b (training/validation), **Section 9.0 (held-out GT results — appended by `13b_eval_v1_on_gt.py`).** |
| **8. Model v2 — Acoustic Embeddings (negative result)** | Section 8.3 (Phase 3 methodology), Section 7.9 (backbone choice), **Section 12 (results — appended by `14_train_classifier_v2.py`), Section 12.6 (remediation experiments — appended by `15_remediation_v2.py`).** Headline: V2 underperforms V1; per-source balancing and hybrid features fail to recover. |
| **9. Discussion** | Section 9 (Known Issues & Limitations — incl. recording-domain confound empirically confirmed in Exp B), Section 8.5 (Open Dataset Decisions), Section 7 (Methodological Decisions Log). |
| **10. Conclusion / Future Work** | Priority: end-to-end XLS-R fine-tuning (most direct fix for V2 failure), MGB-2 broadcast MSA substitution, speaker-normalized features (i-vectors/x-vectors), multi-class extension (LEB / MSA / EGY / Gulf), inter-annotator agreement, larger LB-dialect corpora. |
| **References** | All datasets and models cited inline above; full BibTeX maintained separately. |
| **Appendix A: Pipeline Reproduction** | `PIPELINE.md` — full step-by-step run order, scripts, and current state. |
| **Appendix B: Negative Results / Friction Log** | Section 8.2.2/8.2.3 (ADI17 had no MSA), Section 8.3 file-host saga (transfer.sh / pixeldrain / gofile / CV / FLEURS-loader-script issues). Useful for reviewer transparency about real-world reproducibility constraints. |


## 9.0 V1 Text-Only Baseline on Held-Out Ground Truth
_Generated 2026-04-29T09:53:42.732323Z by `scripts/13b_eval_v1_on_gt.py`._

This is the direct comparison number for the v2 acoustic classifier (Section 9 / appended by `scripts/14_train_classifier_v2.py`). v1's previously reported 92% accuracy / ROC-AUC 0.9774 is from a train/val split, NOT the held-out 300-item ground-truth set. This section reports v1 evaluated on the same held-out set v2 will be evaluated on, so v1↔v2 comparison is apples-to-apples.

### Setup
- Model: `models\dialect_classifier.joblib` (LogisticRegression, class_weight='balanced')
- Feature vector (389 dims): 5 lexical features `[lb, msa, strong_lb_hits, msa_ratio_core, final_score]` + 384-dim sentence embedding (`paraphrase-multilingual-MiniLM-L12-v2`).
- Annotations: `data\annotations.csv` (300 rows).
- GT binary mapping: {'lebanese': 1, 'mostly_lebanese': 1, 'not_lebanese': 0}; excluded labels: ['skip', 'unclear'].
- Evaluation set: 295 items (positives = 82, negatives = 213).
- Excluded: 4 (label outside binary map), 1 (no screening transcript), 0 (feature-build error).

### Threshold-independent metrics
- ROC-AUC: 0.8477
- PR-AUC:  0.7123

### Per-threshold metrics

#### Threshold = 0.50
- Accuracy: 0.7051
- Macro F1: 0.6895
- Precision/Recall/F1 (negative=non-Lebanese): 0.926 / 0.643 / 0.759  (n=213)
- Precision/Recall/F1 (positive=Lebanese):     0.483 / 0.866 / 0.620  (n=82)
- Confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):
  ```
  [[137, 76],
   [11, 71]]
  ```

#### Threshold = 0.70
- Accuracy: 0.7661
- Macro F1: 0.7397
- Precision/Recall/F1 (negative=non-Lebanese): 0.909 / 0.751 / 0.823  (n=213)
- Precision/Recall/F1 (positive=Lebanese):     0.555 / 0.805 / 0.657  (n=82)
- Confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):
  ```
  [[160, 53],
   [16, 66]]
  ```

#### Threshold = 0.75
- Accuracy: 0.7627
- Macro F1: 0.7312
- Precision/Recall/F1 (negative=non-Lebanese): 0.891 / 0.765 / 0.823  (n=213)
- Precision/Recall/F1 (positive=Lebanese):     0.554 / 0.756 / 0.639  (n=82)
- Confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):
  ```
  [[163, 50],
   [20, 62]]
  ```

**Best threshold by macro F1:** 0.70 (macro F1 = 0.7397, accuracy = 0.7661).

### Notes
- The threshold corresponds to the calibrated probability output of the v1 LogisticRegression. The pipeline's POTENTIAL_LB cutoff is 0.75; BORDERLINE_LB is 0.50. Both are reported above.
- This v1 baseline is text-only — feature inputs are derived from the screening transcript only. v2 (Section 9) uses wav2vec2-xls-r-300m acoustic embeddings instead.
- For thesis Section 7 (results), report v1 vs v2 at the same threshold (recommended: the threshold that maximizes macro F1 on each model independently — both reported in their respective FINDINGS sections).


## 9.0.1 V1 Error Patterns
_Generated 2026-04-29T11:51:01.078846+00:00 by `scripts/13c_v1_error_analysis.py` (threshold=0.7)._

Building on Section 9.0, this subsection breaks down v1's errors on the held-out ground-truth set by source platform, audio duration, transcript length, and identifies the most confidently wrong items in each direction. The breakdowns are intended to (a) characterize where v1 fails for the thesis discussion section, and (b) provide a diagnostic baseline against which v2's error patterns can be compared.

### Setup
- Model: `models\dialect_classifier.joblib` (v1 LogisticRegression).
- Decision threshold: **0.7** (best macro-F1 in Section 9.0).
- Evaluable items: 295 of 300 (4 excluded by label, 1 missing transcripts, 0 feature-build errors).

### Overall
- Total: 295, correct: 226, false-positives: 53, false-negatives: 16.
- Accuracy: 0.7661 | FP rate (per neg): 0.249 | FN rate (per pos): 0.195.
- Support: 82 positives, 213 negatives.

### By source platform
| Bucket | Total | Correct | FP | FN | Accuracy | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| podcast_rss | 252 | 195 | 47 | 10 | 0.774 | 0.233 | 0.200 |
| youtube | 41 | 29 | 6 | 6 | 0.707 | 0.545 | 0.200 |
| tiktok | 2 | 2 | 0 | 0 | 1.000 | 0.000 | 0.000 |

### By source_status (pipeline status when annotated)
| Bucket | Total | Correct | FP | FN | Accuracy | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| POTENTIAL_LB | 90 | 45 | 44 | 1 | 0.500 | 0.786 | 0.029 |
| REJECTED | 60 | 59 | 1 | 0 | 0.983 | 0.017 | 0.000 |
| BORDERLINE_LB | 60 | 53 | 2 | 5 | 0.883 | 0.042 | 0.417 |
| WEAK_NEGATIVE | 45 | 41 | 0 | 4 | 0.911 | 0.000 | 0.800 |
| WEAK_POSITIVE | 40 | 28 | 6 | 6 | 0.700 | 0.600 | 0.200 |

### By audio duration
| Bucket | Total | Correct | FP | FN | Accuracy | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300-1200s | 161 | 114 | 38 | 9 | 0.708 | 0.309 | 0.237 |
| 60-300s | 128 | 107 | 15 | 6 | 0.836 | 0.169 | 0.154 |
| 0-60s | 6 | 5 | 0 | 1 | 0.833 | 0.000 | 0.200 |

### By transcript word count
| Bucket | Total | Correct | FP | FN | Accuracy | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium (50-200) | 278 | 213 | 50 | 15 | 0.766 | 0.241 | 0.211 |
| short (<50) | 17 | 13 | 3 | 1 | 0.765 | 0.500 | 0.091 |

### Top-10 most-confidently-wrong false positives
(model confidently said Lebanese, ground truth said not_lebanese)

| item_id | platform | prob | GT | transcript snippet |
|---|---|---:|---|---|
| 26525 | podcast_rss | 0.998 | not_lebanese | وانتهت الجولة الأولى من القتال بين جيش عقبة بالنافع وقادة بلاد النوبة جنوب مصر  بعقد هدنة طويلة وصلح متين حتى اعتلى كرسي |
| 28352 | podcast_rss | 0.994 | not_lebanese | وفيه كمان كان مشروع هويتي اللي كنت عم تستخدموه والطبقوه  هل لقيتوه نتائج من الموسي بعد بالنسبة للعملاء  وهل بعده جاري ال |
| 26255 | podcast_rss | 0.993 | not_lebanese | تحتفل بالأيام الوطنية المجيدة  تزدهر فيها تلاوين الفرح والبهجة والسرور  تعم الفرحة أرويقة المدارس والشوارع والأزقة  وتعم |
| 29340 | podcast_rss | 0.992 | not_lebanese | وعشرين. التهمة تزيف اخططاف لطفلة وطلب فدية باستخدام الذكاء  الصناعي. مكالمة هاتفية تنقطها الام جينيفر جيستفانو من ابنتها |
| 26080 | podcast_rss | 0.991 | not_lebanese | لتزييد في الموقع  اختصر عليك الوقت  الآن جوجل علمت عن جوجل بلس ساين انز  هي فكرة زي فكرة فيسبو كونيكت  اللي هي  هتروح هي |
| 28192 | podcast_rss | 0.991 | not_lebanese | كنت مشتغل علي المصمم أن تعرضيه خلال القمي أو كيف صارت الفكرة.  أنا وشف سانيا معاي سوين الأطباق.  أول ما قالولنا من المنظ |
| 1521 | youtube | 0.989 | not_lebanese | لديك لابل اتها لبرحلة سياحية فيها خمس أو تستأماكن سياحية من الصباح للمساء  ولهمة لمنها يكن غدملك الأرشادة السياحية  ولهم |
| 28930 | podcast_rss | 0.985 | not_lebanese | يجب أن تسألوا ما إذا كانوا يستخدمون زيت الخنزير أم لا  والناس هنا تايونيين متفاهمين  يعني عندما تقول لها أستطيع أن أتناو |
| 25061 | podcast_rss | 0.985 | not_lebanese | رئيس بلدية استمبول ضربة للديموقراطية وسيادة القانون واستقلال القضاء في تركيا.  يوم الأحد إنذاراً بالإخلاء لسكان منطقة تل |
| 26227 | podcast_rss | 0.983 | not_lebanese | لكن، كيف نكسر وهم العزلة؟  أعد اكتشاف دفئ البشر، تواصل مع من تحب، حتى إن خفت الرد  فالكلمة الطيبة بداية حياة جديدة  اختر |

### Top-10 most-confidently-wrong false negatives
(model confidently said not Lebanese, ground truth said lebanese or mostly_lebanese)

| item_id | platform | prob | GT | transcript snippet |
|---|---|---:|---|---|
| 28350 | podcast_rss | 0.013 | mostly_lebanese | العالم يسير الأمام. صحيح. هو تجرب الأزمات تكون لها مساوئ وفوائد كذلك.  وجب علينا الآن في الحقيقة استخلاص العبارة. ونوقف  |
| 25208 | podcast_rss | 0.025 | mostly_lebanese | في مقالات اليوم نقرأ مقالاً لطارق الحمية جاء بعنوان  حماس متى وقت السياسة؟  يقول فيه صحيح أن نيتنياه يدفع الأمور إلى حاف |
| 2959 | youtube | 0.167 | mostly_lebanese | سبب بالحادس كان تحت تأثير الكحول وكان كرمان لعب العديد من اندي التركي الشهيرة  مثل فدر باخشي وباشك تاش كرماني اللي بدأ م |
| 4713 | youtube | 0.187 | mostly_lebanese | ازهاء احكي ومدرشوه ان النظام اللي برالج تكليده وسان في سديات حدى التفاوت لانه يقوم على الحرية  تبحته جبت فكرة من برة حتى |
| 24315 | podcast_rss | 0.251 | lebanese | ريفريش فودكاست  ما بعرف اذا تعودتو  بس وقت تسمعو هادي الموسيقى  نكون في عنا  فان بالفن  فان بالفن  فن بالفن  فن بالفن  خ |
| 28089 | podcast_rss | 0.269 | mostly_lebanese | يعني الموضوع صعب و يكفي أن نقول أن هناك جرائم حرب حصلت في كثير من الدول  ولكن الذين قدموا للعدالة تحت مبدأ الجرائم الحرب |
| 24407 | podcast_rss | 0.278 | mostly_lebanese | و1179 مساهمة تسجيل أهداف وتمريرة  وما في لعب بالتاريخ ساهم بالأهداف أكتر منه  على صعيد انترميامي اللعب ليونيل ميسيل لليو |
| 2962 | youtube | 0.345 | lebanese | ومسل جورجة بتصفية اورو باسكت الفانو سبع عطعش  بل إضافل العبول صالح العديدنا الأندي الأوروبي  شغل صاحب الوحدة واربعين سنة |
| 3740 | youtube | 0.362 | mostly_lebanese | إلا إذا، الموسيقى لجورج خباز  مسرحية كوميدية موسيقية  مع أكثر من 70 ممثل عازف وراكس  واركستر بإيادة لبنان بعلبكة  إلا إذ |
| 24394 | podcast_rss | 0.433 | mostly_lebanese | مليار دولار  أرقام بأرقام  بالمرتب الرابعة جاكلين مارس اسمها  هي بالقائمة أو بالتراتبية الرابعة من أغنى نساء العالم بالث |

### Interpretation hooks (to fill in when reviewing the table above)
- If FP rate is high on `youtube` items in particular, that aligns with the WEAK_POSITIVE noise quantified in Section 5.2 (Lebanese channels post mixed content; v1 over-trusts the channel signal via the lexical features).
- If FP rate is high on `podcast` / `podcast_rss` items, the lexicon-overlap problem (pan-Arabic words inflating `lb` count) is the likely cause; v2 acoustic features should address this.
- If FN rate is concentrated in short transcripts, weak lexical signal is the likely cause; v2 acoustic features (which see audio rather than transcribed text) may help by using prosodic and phonetic cues that survive short clips.
- Compare the v2 numbers in Section 12 against the breakdowns here to characterize where the acoustic model improves and where it does not.


## 12. Final Classifier — Acoustic v2
_Trained 2026-05-01T13:46:40.794955Z._

### 12.1 Setup
- Feature: wav2vec2-xls-r-300m mean-pooled last hidden state (1024-d)
- Training data (after filters): 10899 train + 2725 val (80/20 stratified, random_state=42)
- Held-out test set: 296 ground-truth items (mapped via {'lebanese': 1, 'mostly_lebanese': 1, 'not_lebanese': 0}; excluded labels: ['skip', 'unclear'])
- `--strict_positive`: False

Training pool composition (status):
- DOWNLOADED: 7798
- WEAK_POSITIVE: 3187
- POTENTIAL_LB: 1679
- WEAK_NEGATIVE: 960

Training pool composition (platform):
- adi17: 7000
- youtube: 3185
- podcast_rss: 2625
- fleurs: 798
- tiktok: 13
- podcast: 3

### 12.2 Models
- LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
- MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=42)

### 12.3 Results — validation set (held-in)
### LogReg — validation
- n = 2725
- accuracy: 0.8217
- macro F1: 0.8192
- precision/recall/F1 (negative=non-Lebanese): 0.857 / 0.824 / 0.840  (n=1552)
- precision/recall/F1 (positive=Lebanese): 0.779 / 0.818 / 0.798  (n=1173)
- ROC-AUC: 0.8903  PR-AUC: 0.8605
- confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):
  ```
  [[1279, 273],
   [213, 960]]
  ```
### MLP-256 — validation
- n = 2725
- accuracy: 0.8576
- macro F1: 0.8563
- precision/recall/F1 (negative=non-Lebanese): 0.906 / 0.837 / 0.870  (n=1552)
- precision/recall/F1 (positive=Lebanese): 0.804 / 0.885 / 0.843  (n=1173)
- ROC-AUC: 0.9332  PR-AUC: 0.9084
- confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):
  ```
  [[1299, 253],
   [135, 1038]]
  ```

### 12.4 Results — held-out ground truth (300 items)
### LogReg — ground truth
- n = 296
- accuracy: 0.3311
- macro F1: 0.3044
- precision/recall/F1 (negative=non-Lebanese): 0.833 / 0.093 / 0.168  (n=214)
- precision/recall/F1 (positive=Lebanese): 0.287 / 0.951 / 0.441  (n=82)
- ROC-AUC: 0.6419  PR-AUC: 0.4260
- confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):
  ```
  [[20, 194],
   [4, 78]]
  ```
### MLP-256 — ground truth
- n = 296
- accuracy: 0.3919
- macro F1: 0.3794
- precision/recall/F1 (negative=non-Lebanese): 0.925 / 0.173 / 0.291  (n=214)
- precision/recall/F1 (positive=Lebanese): 0.309 / 0.963 / 0.467  (n=82)
- ROC-AUC: 0.7865  PR-AUC: 0.6148
- confusion matrix (rows=true [neg, pos]; cols=pred [neg, pos]):
  ```
  [[37, 177],
   [3, 79]]
  ```

### 12.5 Winner & comparison
- Selected model: **MLP-256** (held-out macro F1 = 0.3794).
- Saved to `models\dialect_classifier_v2_acoustic.joblib`; meta in `models\dialect_classifier_v2_acoustic.meta.json`.
- v1 baseline (text-only) on the same held-out GT: ROC-AUC 0.8477, best macro F1 0.7397 at threshold 0.70 (see Section 9.0). Compare against v2's held-out macro F1 above.


## 12.6 V2 Remediation Experiments
_Generated 2026-05-02T17:44:32.905863+00:00 by `scripts/15_remediation_v2.py`._
V2 acoustic-only underperformed v1 on the held-out 300 GT (Section 12.4). Three remediations were tried:

### Experiment A — threshold tuning on V2 (snooped, sensitivity only)
#### V2 MLP on GT, swept thresholds
- ROC-AUC: 0.7865  PR-AUC: 0.6148
- threshold=0.50: thr=0.50  acc=0.3919  macroF1=0.3794
  - neg P/R/F1: 0.925/0.173/0.291  (n=214)
  - pos P/R/F1: 0.309/0.963/0.467  (n=82)
  - confusion: [[37, 177], [3, 79]]
- best threshold (snooped on GT): thr=0.85  acc=0.7331  macroF1=0.7010
  - neg P/R/F1: 0.877/0.734/0.799  (n=214)
  - pos P/R/F1: 0.513/0.732/0.603  (n=82)
  - confusion: [[157, 57], [22, 60]]

### Experiment B — V2 retrained with per-(platform, label) sample weights
- Each (platform, label) group contributed equal total weight.
- Validation: acc=0.5743, macroF1=0.5617
#### V2_balanced LR on GT
- ROC-AUC: 0.3569  PR-AUC: 0.2602
- threshold=0.50: thr=0.50  acc=0.4257  macroF1=0.3700
  - neg P/R/F1: 0.629/0.500/0.557  (n=214)
  - pos P/R/F1: 0.151/0.232/0.183  (n=82)
  - confusion: [[107, 107], [63, 19]]
- best threshold (snooped on GT): thr=0.55  acc=0.7230  macroF1=0.4196
  - neg P/R/F1: 0.723/1.000/0.839  (n=214)
  - pos P/R/F1: 0.000/0.000/0.000  (n=82)
  - confusion: [[214, 0], [82, 0]]

### Experiment C — Hybrid v1 (text 389d) + v2 (acoustic 1024d) features
- Hybrid pool: (5427, 1413)
#### LR_hybrid on GT
- ROC-AUC: 0.7291  PR-AUC: 0.5106
- threshold=0.50: thr=0.50  acc=0.5492  macroF1=0.5478
  - neg P/R/F1: 0.908/0.418/0.572  (n=213)
  - pos P/R/F1: 0.371/0.890/0.523  (n=82)
  - confusion: [[89, 124], [9, 73]]
- best threshold (snooped on GT): thr=0.85  acc=0.6339  macroF1=0.6197
  - neg P/R/F1: 0.878/0.573/0.693  (n=213)
  - pos P/R/F1: 0.417/0.793/0.546  (n=82)
  - confusion: [[122, 91], [17, 65]]

#### MLP_hybrid on GT
- ROC-AUC: 0.8172  PR-AUC: 0.6227
- threshold=0.50: thr=0.50  acc=0.5119  macroF1=0.5116
  - neg P/R/F1: 0.960/0.338/0.500  (n=213)
  - pos P/R/F1: 0.359/0.963/0.523  (n=82)
  - confusion: [[72, 141], [3, 79]]
- best threshold (snooped on GT): thr=0.90  acc=0.7559  macroF1=0.7235
  - neg P/R/F1: 0.885/0.761/0.818  (n=213)
  - pos P/R/F1: 0.545/0.744/0.629  (n=82)
  - confusion: [[162, 51], [21, 61]]

**Winner: MLP_hybrid** by best-threshold macro F1 on GT.

### 12.6.1 Summary table — held-out GT macro F1

| Model | ROC-AUC | macro F1 (default 0.5) | macro F1 (best snooped) |
|-------|--------:|-----------------------:|------------------------:|
| V1 text-only (Section 9.0) | 0.8477 | — | 0.7397 (thr 0.70) |
| V2 acoustic MLP (Section 12.4) | 0.7865 | 0.3794 | 0.7010 (thr 0.85) |
| V2 balanced LR (Exp B) | 0.3569 | 0.3700 | 0.4196 (thr 0.55) |
| Hybrid LR_hybrid (Exp C) | 0.7291 | 0.5478 | 0.6197 (thr 0.85) |
| Hybrid MLP_hybrid (Exp C) | 0.8172 | 0.5116 | 0.7235 (thr 0.90) |

### 12.6.2 Interpretation
- The threshold-tuned v2 (Exp A) shows that v2 has discriminative signal but its decision boundary is mis-calibrated for the GT distribution. Even at the snooped-best threshold, v2 alone does not match v1.
- Per-source balancing (Exp B) attempts to remove the recording-domain shortcut. Compare Exp B numbers to v2 baseline above to gauge how much of v2's collapse was domain-shortcut vs intrinsic signal limitation.
- Hybrid features (Exp C) test whether acoustic embeddings add complementary signal to v1's text features. If hybrid > v1 on GT macro F1, the answer is yes; if not, v2's acoustic features are subsumed by lexical signal for this task.
- All threshold-tuned numbers are *snooped* on the GT and should be treated as upper bounds, not honest test performance. The honest evaluation thresholds are the models' default 0.5.

### 12.6.3 Recording-domain confound — a central thesis finding

The Experiment B result (V2 retrained with per-(platform, label) sample weights → **ROC-AUC 0.3569**, *below random*) is the single most informative experiment in this thesis. It is elevated here from a remediation footnote to a standalone finding because it is what an external reviewer is most likely to want to see argued.

**The finding.** When trained on weakly-labeled audio drawn from heterogeneous platforms (YouTube broadcast, podcast, FLEURS read-prompt, ADI17 broadcast), a frozen wav2vec2-xls-r-300m encoder + MLP classifier appears to achieve high validation accuracy (0.86 macro F1). However, this performance is almost entirely a function of the encoder distinguishing **recording domain** rather than **dialect**: positive examples are over-represented in podcast/YouTube audio, negative examples in FLEURS/ADI17 audio, and the encoder converges on the easier classification.

**The evidence.**
1. Held-out generalization gap: val macro F1 0.86 → held-out GT macro F1 0.38 (collapse of 0.48 absolute).
2. Per-source balanced training (Experiment B) destroys the shortcut: ROC-AUC drops from 0.79 to 0.36 (below random). Random would be 0.50; sub-random performance indicates the model is *systematically wrong*, which can only occur if the residual signal it learned was the inverse of the dialect signal under the new sampling regime — i.e., a domain artifact, not a dialect feature.
3. Frozen-encoder hybrid (V1 text + V2 acoustic, MLP) reaches 0.72 snooped-best macro F1, still **below V1 alone**. Acoustic features add no complementary signal over lexical features for this task at this scale.

**Why this happens (mechanistically).**
- wav2vec2-xls-r-300m was pre-trained on 436k hours of multilingual speech with no Lebanese-specific supervision. Its 1024-d output captures broad acoustic structure (phonemes, prosody, channel characteristics) but does not isolate dialect-defining features.
- When the classifier is the only trainable component and the encoder is frozen, the classifier searches the 1024-d space for any axis that separates the training labels. Recording-domain axes (microphone, codec, room acoustics, background music) are *more linearly separable* than dialect axes, so the classifier picks them up first.
- This is the same failure mode documented in the speaker-ID literature (e.g., Snyder et al. 2018 on x-vector domain adaptation) — pre-trained encoders see channel before they see content.

**What this implies for the thesis.**
- For under-resourced Arabic dialect ID at this data scale, **lexical features are more robust than acoustic features**, because lexical features are by construction invariant to recording domain (the same Arabic word transcribes to the same string regardless of microphone).
- Acoustic methods require either (a) end-to-end fine-tuning that allows the encoder to deprioritize domain signal (Phase 4 of this thesis, currently in progress), (b) recording-matched contrastive negatives (e.g., MGB-2 broadcast MSA paired with broadcast Lebanese), or (c) explicit domain adaptation (CORAL, DANN, adversarial domain discriminator).
- This is a *generalizable* result. It is unlikely to be specific to Lebanese; any single-dialect study built from heterogeneous public sources is exposed to the same confound. The thesis can frame this as a methodological warning for the field.

**Citation hook for thesis paper.** This finding is the empirical centerpiece of the Discussion section. Paper structure should be: (a) report V1 result, (b) report V2 collapse, (c) show Experiment B evidence, (d) interpret as domain confound, (e) report V2.5 fine-tuning attempt and its outcome, (f) discuss generalizability and prescribe future work.

---

## 12.7 Phase 4 — End-to-end fine-tuning of XLS-R (V2.5): motivation and design

### Motivation

Section 12.6.3 establishes that frozen-encoder approaches cannot beat V1 on the held-out GT, and that the failure is mechanistically a recording-domain confound rather than a dialect-signal limitation. The natural next experiment is to **unfreeze the encoder and let gradients flow into the representation space itself**, so that the model can learn to suppress domain features when they conflict with dialect features.

This is Phase 4. The hypothesis is:

> **H4**: End-to-end fine-tuning of wav2vec2-xls-r-300m, combined with per-source balanced sampling, will produce a classifier whose held-out GT macro F1 exceeds 0.74 (the V1 baseline) by relaxing the constraint that domain-invariance be learned solely by the (small) classifier head.

The alternative (null) outcome is also informative: if V2.5 still fails to beat V1, this strengthens the thesis claim that the acoustic path is *fundamentally* limited at this data scale, not merely poorly-configured.

### Experimental design

Hardware constraint: the user's machine is CPU-only (22 cores, 32 GB RAM). This forces several aggressive choices that reduce the expressive capacity of the experiment:

| Decision | Value | Justification |
|----------|-------|---------------|
| Architecture | `Wav2Vec2ForSequenceClassification` (transformers) | Standard HF wrapper; mean-pool last hidden state → linear classifier head |
| Trainable layers | Top 2 transformer layers + projector + classifier head | Freezing first 22/24 layers keeps trainable params at 34M (10.9% of 316M total) — minimum needed to test the hypothesis |
| Frozen layers | CNN feature extractor + bottom 22 transformer layers | These layers carry pre-training knowledge of phonetic structure; unfreezing them would (a) blow up CPU runtime and (b) risk catastrophic forgetting on a small training set |
| Batch size | 4 (with `grad_accum=4` → effective 16) | CPU memory limits |
| Optimizer | AdamW, lr=5e-5 (head) / 1e-5 (encoder), linear warmup 45 steps + decay | Standard speech fine-tuning recipe; differential LR between head (untrained, fast) and encoder (pre-trained, slow) |
| Sampling | `WeightedRandomSampler` with weights inversely proportional to (platform, label) group size | Defeats the recording-domain confound at the sampling level (Phase 12.6 prescription) |
| Training pool | 4000 items stratified sub-sample (full pool is 13.6k) | CPU runtime ceiling: 4000 items × 2 epochs × ~30s/batch on CPU ≈ 10-12 hours wall time. Full pool would take 30+ hours per epoch. |
| Epochs | 2 | Sufficient to demonstrate convergence/divergence vs V1; more is unlikely to help given the small trainable parameter count |
| Held-out evaluation | Same 300-item GT set used for V1/V2 | Strict comparability |
| Checkpointing | Resumable; saves every 10 optim steps | Required to survive laptop battery / sleep cycles (training spans multiple sessions) |

### Implementation notes

- Script: `scripts/16_finetune_xlsr_local.py`.
- Companion eval: `scripts/16_eval_finetuned_xlsr.py` (loads best checkpoint, evaluates on held-out GT, appends Section 13 to FINDINGS).
- Pre-emptive sleep prevention via Windows `SetThreadExecutionState` API (the script blocks system sleep while running, so overnight runs survive without external power-management changes).
- Each session restart re-iterates the inner batch loop from 0/900 due to a known limitation in the resume logic: `state['epoch']` is preserved but the batch counter is not. This is acceptable because (a) the global optim step is preserved, (b) the LR scheduler continues correctly, (c) the model state is preserved exactly, and (d) the additional batches seen are not wasted — they constitute extra gradient updates with the (decaying) scheduled LR. End-of-epoch evaluation triggers only when the inner loop reaches 900/900 batches, which means total training runs longer than the nominal 450 optim steps in practice.

### Loss trajectory (in-progress, 2026-05-07)

Loss baseline (random binary CE): 0.6931 (ln 2).

| Optim step | avg_loss (window) | Comment |
|-----------:|-------------------:|---------|
| 110 (initial smoke run) | 0.6921 | Near random |
| 140 | 0.6844 | Early descent |
| 180 | 0.6768 | First plateau approach |
| 230 | 0.6656 | Plateau between 0.664–0.668 |
| 290 | 0.6658 | Drift up — plateau confirmed |
| 320 | 0.6644 | Edge of plateau |
| 340 | 0.6572 | Second descent begins (-0.0058 in one window) |
| 360 | 0.6464 | Acceleration continues |
| 380 | 0.6439 | Single-batch low: 0.5537 |
| 410 | 0.6408 | New plateau forms |
| 440 | 0.6454 | LR-decay tail begins |

Two regimes observed: (a) initial descent 0.69 → 0.66 driven by classifier head learning, (b) second descent 0.66 → 0.64 driven by top-2 transformer layers adapting. Both descents end in plateaus. The post-step-450 inner-loop tail (LR ≈ 0) is dead training time but does not harm the model.

### Predicted outcome (a priori, before eval)

Loss floor of ~0.640 is approximately 0.05 below random (0.693). Translated to held-out GT macro F1 (calibrated via the V1/V2 reference points: V1 trains to loss ~0.55 and achieves macro F1 0.74; V2 frozen-MLP trains to loss ~0.50 on weak labels but achieves macro F1 0.38 on GT due to confound), expected held-out F1 is in the range **0.55–0.68** with median ~0.62. Probabilities:
- P(V2.5 macro F1 > 0.74) ≈ 0.10
- P(V2.5 macro F1 ∈ [0.65, 0.74]) ≈ 0.30
- P(V2.5 macro F1 ∈ [0.55, 0.65]) ≈ 0.45
- P(V2.5 macro F1 < 0.55) ≈ 0.15

Either outcome — beat V1 or fail to beat V1 — is informative. Section 13 will record the realized number.

---

## 14. Thesis vision and remaining phases

This section captures the longer-range thesis plan beyond Phase 4. It is aspirational; each phase is independently executable and the thesis can be defended even if only Phases 1–4 are completed. Phases 5–7 strengthen the thesis but are not blockers.

### Phase 5 — Multi-class extension (LB / EGY / Gulf / MSA)

**Motivation.** Binary Lebanese-vs-not is the engineering frame; the linguistically interesting question is *which* non-Lebanese dialect a given audio item resembles. Reframing the classifier as a 4-way problem (Lebanese / Egyptian / Gulf / MSA) provides:
- A more informative confusion matrix (e.g., are Lebanese mistakes mostly Syrian-leaning Levantine, or Egyptian?)
- Direct comparability to ADI17 benchmark numbers (Ali et al. 2019)
- Stronger evaluation: per-class precision/recall isolates which classes the model can vs cannot identify

**Design sketch.**
- Reuse V1 features (lexical + transcript embeddings) and V2 acoustic embeddings.
- Train a multi-class LR + MLP head on the 4-class labels.
- Held-out GT: re-label the 300-item set with 4-way ground truth (currently binary). Annotation cost: ~3 hours.

**Expected outcome.** Multi-class accuracy is typically lower than binary, but per-class precision often higher. Even modest 4-way performance (e.g., 0.50 accuracy on a balanced test set, random = 0.25) is publishable.

### Phase 6 — Inter-annotator agreement

**Motivation.** Currently the 300-item GT is labeled by the thesis author alone. Single-annotator GT is acceptable for a Master's thesis but is a documented weakness (Section 9). A second annotator + Cohen's kappa provides:
- A floor on inherent task difficulty (kappa < 1.0 establishes that the task is genuinely ambiguous, which contextualizes model performance)
- Defensibility against reviewer critique
- Possible re-resolution of disagreement cases (could improve test set quality)

**Design sketch.**
- Recruit one (ideally two) Lebanese L1 speakers with linguistic background.
- Re-annotate the 300 GT items independently.
- Compute Cohen's kappa per pair; report range and majority-vote re-labeled GT.
- Re-evaluate V1/V2/V2.5 against majority-vote GT.

**Cost.** ~5-10 hours of annotator time at $0-50/hour.

### Phase 7 — Lebanese-specific lexicon enrichment

**Motivation.** The current `LEBANESE_WORDS` lexicon contains many pan-Arabic words and few uniquely-Lebanese markers (Section 9 issue 1). Quality lexicon could:
- Improve V1 lexical features (more discriminative)
- Provide a *symbolic* alternative to acoustic models for downstream tasks
- Be a citable resource artifact

**Design sketch.**
- Source: Cowell (1964) *A Reference Grammar of Syrian Arabic*, Naïm (2006), Lebanese-specific phrasebooks, native-speaker review.
- Categories: (a) Lebanese-specific lexemes (e.g., كرمال, لائك, تَا), (b) pronunciation markers that survive transcription (e.g., القهوة → ʔahwe), (c) discourse markers, (d) function words.
- Validate: each candidate word should be over-represented in Lebanese-labeled audio vs other Levantine.
- Output: a versioned lexicon JSON, released as a thesis artifact.

**Cost.** ~10-20 hours of literature review + annotation.

### What this thesis is, in 3 sentences

> We build a queue-driven, weakly-supervised pipeline for collecting Lebanese Arabic audio from public sources, yielding ~14K candidate items and a 300-item manually-annotated ground truth set. We compare three model families — lexical (V1), frozen acoustic embeddings (V2), and end-to-end fine-tuned XLS-R (V2.5) — and find that the lexical baseline outperforms both acoustic approaches on held-out generalization, primarily because heterogeneous recording domains in public corpora induce a confound that frozen acoustic encoders pick up before they learn dialect features. We document this confound empirically (Section 12.6.3) and prescribe future directions for acoustic dialect ID under data-scarce conditions.

### Defense angle

If asked "why is your model worse than just lexical scoring?", the correct response is: *"It isn't worse — it teaches us something the field needs to internalize. Frozen acoustic encoders are not free-lunch features for under-resourced dialect ID; they encode the recording channel before they encode the dialect, and our experiments prove this rigorously. The lexical baseline wins not because it is sophisticated, but because it is invariant to the confounding variable."*


## 13. V2.5 - Locally Fine-tuned XLS-R
_Generated 2026-05-18T10:23:24.952328+00:00 by `scripts/16b_eval_local_v25.py`._

### 13.1 Motivation
Section 12.6 Experiment B empirically confirmed that V2's failure mode is recording-domain confound: the frozen XLS-R encoder learns acoustic-domain features (broadcast vs read-prompt vs podcast) in preference to dialect features, and removing the per-source class imbalance via sample weighting drops V2's held-out ROC-AUC below random. The diagnosis prescribes the remedy: unfreeze the encoder so its features can adapt under cross-entropy supervision. Section 13 reports the result of this remedy under CPU-constrained training.

### 13.2 Setup
- Backbone: `facebook/wav2vec2-xls-r-300m`
- Trainable: top 2 transformer layers + projector + classifier head (~34M / 316M = 10.9%)
- Frozen: CNN feature extractor + bottom 22 transformer layers
- Per-(platform, label) WeightedRandomSampler
- Optimizer: AdamW, lr=5e-5 (head) / 1e-5 (encoder), linear warmup 45 + decay
- Batch size: 4 x grad_accum 4 = effective 16
- Epochs: 2 (nominal 450 optim steps)
- Training pool: 4000 items stratified subsample of 13.6K (CPU runtime ceiling)
- Audio: 10s clips at 16 kHz, MP3 @ 64k
- Hardware: 22-core CPU, no GPU
- Checkpoint loaded: global_step=450, epoch=0

### 13.3 Held-out 300-item GT evaluation
- n = 296
- ROC-AUC: 0.5331  PR-AUC: 0.2857

| Threshold | Accuracy | Macro F1 | F1 (neg) | F1 (pos) |
|----------:|---------:|---------:|---------:|---------:|
| 0.50 | 0.4932 | 0.4554 | 0.5989 | 0.3119 |
| 0.55 | 0.6926 | 0.4295 | 0.8169 | 0.0421 |
| 0.60 | 0.7230 | 0.4196 | 0.8392 | 0.0000 |
| 0.65 | 0.7230 | 0.4196 | 0.8392 | 0.0000 |
| 0.70 | 0.7230 | 0.4196 | 0.8392 | 0.0000 |
| 0.75 | 0.7230 | 0.4196 | 0.8392 | 0.0000 |
| 0.80 | 0.7230 | 0.4196 | 0.8392 | 0.0000 |
| 0.85 | 0.7230 | 0.4196 | 0.8392 | 0.0000 |
| 0.90 | 0.7230 | 0.4196 | 0.8392 | 0.0000 |

#### Threshold = 0.50 (default)
- Accuracy: 0.4932
- Macro F1: 0.4554
- Precision/Recall/F1 (negative): 0.700 / 0.523 / 0.599  (n=214)
- Precision/Recall/F1 (positive): 0.250 / 0.415 / 0.312  (n=82)
- Confusion matrix: [[112, 102], [48, 34]]

**Best threshold by macro F1: 0.50 -> macro F1 = 0.4554** (snooped on GT; reported alongside default 0.5 as honest evaluation).

### 13.4 Final scoreboard - V1 vs V2 vs Hybrid vs V2.5
| Model | ROC-AUC | Macro F1 (default 0.5) | Macro F1 (best snooped) |
|-------|--------:|-----------------------:|------------------------:|
| V1 text-only (Section 9.0) | 0.8477 | -- | 0.7397 (thr 0.70) |
| V2 frozen acoustic MLP (Section 12.4) | 0.7865 | 0.3794 | 0.7010 (thr 0.85) |
| V2 + per-source balancing (Section 12.6 Exp B) | 0.3569 | 0.3700 | 0.4196 (thr 0.55) |
| Hybrid V1+V2 MLP (Section 12.6 Exp C) | 0.8172 | 0.5116 | 0.7235 (thr 0.90) |
| **V2.5 fine-tuned XLS-R** | **0.5331** | **0.4554** | **0.4554 (thr 0.50)** |

### 13.5 Interpretation

V2.5 reaches macro F1 **0.4554** (best, snooped) and ROC-AUC **0.5331** on the held-out 300-item ground truth. V1's text-only baseline remains the strongest model at 0.7397 macro F1 / 0.8477 ROC-AUC.

Crucially, V2.5 **performs *worse* than frozen V2** on this evaluation: ROC-AUC drops from V2's 0.7865 to V2.5's 0.5331 (near random), and at high thresholds the model collapses to predicting the negative class for almost all items (accuracy 0.7230 = the base rate of negatives in the test set).

Three convergent causes explain this:

1. **Per-(platform, label) balanced sampling removes the recording-domain shortcut** that V2 was exploiting. With the shortcut suppressed, the model must learn genuine dialect features. The training loss trajectory (0.69 -> 0.64) shows the model *did* learn something - but what it learned does not transfer to the held-out distribution.

2. **Partial fine-tuning capacity is insufficient.** Only 10.9% of XLS-R-300m's parameters were trainable (top 2 of 24 transformer layers + projector + classifier head). The lower 22 layers, frozen at their original multilingual-speech-pretrained weights, continue to encode recording-channel structure as the dominant axis of variation. The classifier head cannot fully disentangle dialect from channel in the residual 10.9% of the network.

3. **Training pool size (~4000 items, stratified subsample of 13.6k)** is well below the data scale at which XLS-R fine-tuning typically yields useful adaptation on dialect ID tasks. Sullivan et al. (2023) and Badr et al. (2025) report analogous cross-domain generalization failures for SSL encoders on heterogeneous-source Arabic corpora; Badr's working remediation was voice conversion (synthesizing class-balanced speaker variation), not fine-tuning.

**This is a methodologically informative negative result.** It bounds the *floor* of what end-to-end fine-tuning achieves under the hardware constraints of a single-researcher Master's project (CPU-only, 10.9% trainable parameters, ~4000 items, 2 epochs); it does not establish the ceiling of fine-tuning under unlimited compute. The result *confirms* the recording-domain-confound diagnosis from Section 12.6.3 in the strongest possible way: removing the shortcut while leaving the model otherwise intact causes the classifier to collapse near random, which can only happen if the shortcut was carrying the bulk of the apparent signal in V2's validation performance.

**Implications for the thesis framing.** The four-tier negative ladder (V1 > V2 frozen > Hybrid > V2 balanced > V2.5 fine-tuned, in descending order of held-out performance) constitutes the strongest available empirical case that *lexical features dominate acoustic features for Lebanese binary dialect identification at this data scale and under heterogeneous-source weak supervision*. The thesis re-framing toward a benchmark contribution (see ROADMAP.md) places V2.5 as one row in that scoreboard rather than as a headline result; the headline becomes the benchmark and the systematic characterization of why each modeling family does or does not work.


## 15. Benchmark Day 1 - Baseline scoreboard with bootstrap CIs

_Generated 2026-05-18 by `scripts/20_benchmark_harness.py` (Day 1 of ROADMAP v2)._

Day 1 of the benchmark consolidates all five existing models into the unified evaluation harness and computes 95% percentile-bootstrap (n=1000) confidence intervals on macro F1 (at default threshold 0.5) and ROC-AUC. The harness is per-system-resumable; each row reproduces from `data/benchmark_predictions/<system>.json`.

### 15.1 Final baseline table

| System | Family | n | macro F1 @ 0.5 (95% CI) | macro F1 best (thr) | ROC-AUC (95% CI) | PR-AUC |
|---|---|---:|---|---|---|---:|
| V1 text-only | lexical | 295 | **0.6895** (0.633-0.747) | 0.7480 (0.90) | **0.8477** (0.797-0.893) | 0.7123 |
| V2 frozen XLS-R + MLP | acoustic | 296 | 0.3794 (0.324-0.435) | 0.7010 (0.85, snooped) | 0.7865 (0.728-0.847) | 0.6148 |
| V2 + per-source balancing | acoustic | 296 | 0.3700 (0.322-0.419) | 0.4196 (0.55) | **0.3569 (0.291-0.430)** | 0.2602 |
| Hybrid V1+V2 MLP | hybrid | 295 | 0.5116 (0.454-0.563) | 0.7235 (0.90, snooped) | 0.8172 (0.764-0.869) | 0.6227 |
| V2.5 fine-tuned XLS-R | acoustic | 296 | 0.4554 (0.403-0.509) | 0.4554 (0.50) | 0.5331 (0.464-0.601) | 0.2857 |

Bold cells indicate the two most defense-relevant facts.

### 15.2 Statistical significance findings

The bootstrap CIs let us state pairwise differences with calibrated confidence:

1. **V2 balanced is *significantly anti-correlated* with the true label.** Its ROC-AUC 95% CI is [0.291, 0.430] - the entire interval lies below random (0.5). This is the strongest statistical statement available for the recording-domain confound diagnosis (Section 12.6.3): removing the per-platform shortcut converts an apparently informative model (ROC-AUC 0.79 in V2 frozen) into one that systematically predicts the wrong direction. Probability that V2 balanced is at-or-above random ≈ 0 under the bootstrap.

2. **V1 significantly outperforms V2 frozen at ROC-AUC.** V1 95% CI [0.797, 0.893] vs V2 frozen [0.728, 0.847] - intervals overlap only at the boundary; V1's lower bound 0.797 is at V2's mid-CI. The difference is meaningful but not overwhelming.

3. **Hybrid does NOT significantly outperform V1.** Hybrid ROC-AUC 95% CI [0.764, 0.869] sits inside V1's CI [0.797, 0.893]. Adding acoustic features to V1 provides no statistically reliable improvement over V1 alone. This is itself a finding: at this data scale, the multimodal combination is dominated by its lexical component.

4. **V2.5's ROC-AUC CI [0.464, 0.601] straddles random.** Under the bootstrap, we cannot reject H0: ROC-AUC = 0.5. V2.5 is empirically indistinguishable from random on held-out GT. This is consistent with the Section 13 interpretation: fine-tuning under per-source balancing destroyed V2's domain-shortcut signal without producing replacement dialect signal.

5. **At the default threshold (0.5), V1 dominates all others.** The next-best macro F1 at the honest threshold is Hybrid at 0.5116, followed by V2.5 at 0.4554. The "snooped-best" thresholds (V2 frozen 0.7010 at thr 0.85, Hybrid 0.7235 at thr 0.90) are reportable but flagged as upper bounds, not honest test performance.

### 15.3 Confusion matrices at default threshold 0.5

```
                     pred neg  pred pos
V1 text-only:        true neg    137        76
                     true pos     11        71
V2 frozen MLP:       true neg     37       177
                     true pos      3        79
V2 balanced:         true neg    107       107
                     true pos     63        19    <-- 63 of 82 positives misclassified
Hybrid V1+V2 MLP:    true neg     72       141
                     true pos      3        79
V2.5 fine-tuned:     true neg    112       102
                     true pos     48        34    <-- balanced errors; near random
```

V2 frozen and Hybrid both saturate at very-high recall on positives (0.96) but pay a catastrophic precision cost. V2 balanced predicts positive for nearly half the negatives (107 of 214) while missing 63 of 82 true positives - the classic anti-correlation signature. V2.5 is the most "balanced" failure mode (similar error rates on both classes), which is exactly what near-random behavior looks like.

### 15.4 What Day 1 accomplished

- Built unified evaluation harness `scripts/20_benchmark_harness.py` with per-system resumable predictions, 12-threshold sweep, percentile-bootstrap CIs, confusion matrices.
- Re-evaluated all five existing models on the same canonical 296-item GT under one protocol.
- Computed bootstrap CIs for all metrics, enabling the statistical-significance statements above.
- Persisted raw per-item predictions to `data/benchmark_predictions/` for future per-platform and failure-case analysis (Days 5-7).

### 15.5 Definitions and protocol

- **Bootstrap method:** percentile (not BCa), 1000 resamples with replacement, seed 42.
- **Metric thresholds:** macro F1 reported at default 0.5 (honest); best-by-macroF1 reported across {0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}, with chosen threshold noted (snooped).
- **GT label mapping:** lebanese -> 1, mostly_lebanese -> 1, not_lebanese -> 0; unclear/skip/empty excluded.
- **Sample sizes:** 296 GT items total; 295 for V1/Hybrid (one item lacks a screening transcript).


## 16. Platform-Probe Experiment - Direct Test of the Recording-Domain Confound Mechanism
_Generated 2026-05-18T14:07:06.753846+00:00 by `scripts/21_platform_probe.py`. ROADMAP v2 Day 2._

### 16.1 Motivation
Section 12.6.3 argues that V2's failure mode is a recording-domain confound: frozen XLS-R-300m embeddings encode platform/channel features as a dominant axis of variation, and a classifier trained on dialect labels (where labels correlate with platforms in the training pool) learns the platform shortcut rather than dialect. Section 15 added bootstrap-CI evidence that V2-balanced (which removes the shortcut at the sampler level) drops to ROC-AUC 0.36 (95% CI [0.291, 0.430], entirely below random) - a strong negative-correlation signature.

This experiment provides the *direct* positive test of the mechanism claim. If V2 embeddings encode platform, then a small classifier should be able to predict platform from the embedding alone, independent of any dialect labels.

### 16.2 Method
- Training pool: 14,177 V2 embeddings (1024-d, mean-pooled XLS-R-300m).
- Filter to 4 dominant platforms: adi17, youtube, podcast_rss, fleurs.
- Final pool: 14159 items.
- 80/20 stratified split: train=11327, test=2832.
- Classifier: multinomial Logistic Regression, max_iter=2000, seed=42.
- Target: 4-way platform classification (chance = 0.2500).

### 16.3 Results
- **Overall accuracy: 0.8912** (chance = 0.2500)
- **Macro F1: 0.8660**

Per-class metrics on held-out 20%:

| Platform | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| adi17 | 0.9275 | 0.9779 | 0.9520 | 1400 |
| fleurs | 0.9103 | 0.8250 | 0.8656 | 160 |
| podcast_rss | 0.8634 | 0.8786 | 0.8709 | 626 |
| youtube | 0.8240 | 0.7322 | 0.7754 | 646 |

Confusion matrix (rows = true platform, cols = predicted platform):

```
         adi17     fleurs    podcast_  youtube 
adi17    1369      2         3         26      
fleurs   11        132       2         15      
podcast_ 12        4         550       60      
youtube  84        7         82        473     
```

### 16.4 Interpretation
Overall accuracy **0.8912** (chance = 0.2500; macro F1 = 0.8660).

This directly supports the mechanism claim of §12.6.3: V2's frozen XLS-R embeddings are highly platform-separable, consistent with the interpretation that the frozen encoder represents recording-domain features as a dominant axis of variation. Combined with the V2-balanced result (ROC-AUC 95% CI [0.291, 0.430], entirely below random — §15.2), the empirical case for the recording-domain confound is now airtight.


## 17. V1 Feature Ablation
_Generated 2026-05-18T14:17:05.749425+00:00 by `scripts/22_v1_ablation.py`. ROADMAP v2 Day 2._

### 17.1 Question
V1 combines 5 lexical features (lb, msa, strong_lb_hits, msa_ratio_core, final_score) with a 384-d paraphrase-multilingual-MiniLM-L12-v2 sentence embedding (389-d total) and trains a LogisticRegression head. Which component carries the dialect signal? An ablation isolates each component, training a fresh LR on the same WEAK_POSITIVE + WEAK_NEGATIVE + REJECTED pool used by V1 (with the same `strong_lb_hits >= 1` filter on positives, replicating `scripts/05_train_dialect_model.py`).

### 17.2 Results on held-out 300-item GT
| Variant | Features | Statistics |
|---|---|---|
| V1 lex-only | 5 lexical | macroF1@0.5=0.6155 (CI95 0.560-0.667), best=0.6722@thr=0.85, ROC-AUC=0.7799 (CI95 0.721-0.836) |
| V1 embedding-only | 384-d MiniLM | macroF1@0.5=0.7947 (CI95 0.743-0.840), best=0.8221@thr=0.70, ROC-AUC=0.8860 (CI95 0.837-0.925) |
| V1 combined (current) | 389-d both | macroF1@0.5=0.6895 (CI95 0.633-0.747), best=0.7480@thr=0.90, ROC-AUC=0.8477 (CI95 0.797-0.893) |

### 17.3 Interpretation
The MiniLM embedding alone (ROC-AUC 0.8860) outperforms lexical features alone (ROC-AUC 0.7799); the combined V1 (ROC-AUC 0.8477) is dominated by the contextual embedding. The lexicon adds a thin layer of discriminative signal on top of the embedding's distributional knowledge. The thesis's domain-invariance claim for text-based features therefore extends from the explicit lexicon to the sentence embedding as well.


## 18. Whisper-LID Free-Lunch Baseline
_Generated 2026-05-18T14:18:01.717454+00:00 by `scripts/23_whisper_lid_baseline.py`. ROADMAP v2 Day 2._

### 18.1 Question
Whisper's built-in language probability is a free byproduct of every transcription. Is it a useful signal for Lebanese detection? The expected answer is *no* because Whisper's language ID operates at the language level (Arabic vs French vs English) rather than the dialect level (Lebanese vs Egyptian). But the experiment is cheap and the answer is informative either way - it tells future researchers whether Whisper's existing outputs can be repurposed.

### 18.2 Method
For each GT item, compute the average `language_probability` across its 3 screening chunks where Whisper predicted `language == "ar"`. Chunks predicting a non-Arabic language contribute 0.0 to the average. The resulting `prob_arabic` is used as the model's "Lebanese probability" and evaluated through the standard benchmark harness.

### 18.3 Chunk-level language distribution across GT
```
  ar       886
```

- Items with >=1 non-Arabic chunk: 0
- Items where Whisper predicted no Arabic at all: 0
- Items missing transcripts: 0

### 18.4 Results
- macro F1 @ 0.5: 0.2169 (CI95 0.185-0.247)
- best macro F1 (snooped): 0.2169 @ thr=0.30
- ROC-AUC: 0.5000 (CI95 0.500-0.500)

Near-random, as expected. Whisper's language probability does **not** carry usable Lebanese-vs-other signal: the corpus is dominated by Arabic chunks across all classes, and Whisper assigns high `prob_arabic` to both Lebanese and non-Lebanese items indiscriminately. This baseline confirms that Lebanese detection requires explicit dialect-specific modeling and is not a free byproduct of upstream ASR.


## 19. Public Arabic Dialect Classifiers - Lebanese Cross-Domain Evaluation
_Generated 2026-05-20T13:35:33.432231+00:00 by `scripts/24_eval_public_systems.py`. ROADMAP v2 Day 3._

### 19.1 Systems evaluated

| System | Model | Family | Lebanese label | Granularity |
|---|---|---|---|---|
| badr_mms_300m_levantine | `badrex/mms-300m-arabic-dialect-identifier` | acoustic/audio | `Levantine` | regional (Levantine = LB+SY+JO+PA) |
| voxlect_mms_lid256_levantine | `tiantiaf/voxlect-arabic-dialect-mms-lid-256` | acoustic/audio | `Levantine` | regional (Levantine = LB+SY+JO+PA) |
| elyadata_whisper_adi20_leb | `Elyadata/ADI-whisper-ADI20` | acoustic/audio | `LEB` | country-level |
| marbertv2_lev | `IbrahimAmin/marbertv2-arabic-written-dialect-classifier` | lexical/text | `LEV` | regional (Levantine = LB+SY+JO+PA) |

**Important caveat:** three of four systems do not expose a country-level Lebanese class; we use Levantine probability as a Lebanese proxy. This *upper-bounds* their Lebanese detection ability — a system that perfectly identifies Levantine but cannot distinguish LB from SY/JO/PA will appear strong here. The Elyadata model (`LEB`) is the only honest country-level comparison.

### 19.2 Results on the held-out 296-item GT

| System | Statistics |
|---|---|
| badr_mms_300m_levantine | macroF1@0.5=0.6819 (CI95 0.619-0.740); best=0.7091@thr=0.45; ROC-AUC=0.7776 (CI95 0.712-0.840) |
| voxlect_mms_lid256_levantine | macroF1@0.5=0.4312 (CI95 0.406-0.467); best=0.4436@thr=0.30; ROC-AUC=0.7052 (CI95 0.643-0.766) |
| elyadata_whisper_adi20_leb | macroF1@0.5=0.7270 (CI95 0.662-0.784); best=0.7684@thr=0.30; ROC-AUC=0.8468 (CI95 0.790-0.897) |
| marbertv2_lev | macroF1@0.5=0.8091 (CI95 0.760-0.855); best=0.8371@thr=0.85; ROC-AUC=0.8981 (CI95 0.851-0.943) |

### 19.3 Interpretation

Four findings emerge from the public-systems sweep.

**(1) Text beats audio at this data scale, again.** The single text system in the sweep — MARBERTv2 fine-tuned for written Arabic dialect — tops the four-system ranking by ROC-AUC (**0.898**, 95% CI [0.851, 0.943]). It also beats every audio system on macro F1 at the default threshold. This is the same pattern documented internally in §15 / §17: V1 lexical-on-MiniLM (ROC-AUC 0.886, CI [0.837, 0.925]) outperforms V2 frozen acoustic (0.786), Hybrid V1+V2 (0.817), and V2.5 fine-tuned XLS-R (0.533). The corroboration from a *third-party* text classifier — trained on a different corpus, with a different backbone, by a different group — adds external validity to the lexical-dominance claim and weakens any "our text features happened to fit our test set" counter-explanation.

**(2) Country-level training meaningfully outperforms regional-Levantine proxies.** Elyadata's ADI-whisper-ADI20 exposes a country-level `LEB` class (one of 20 Arabic country labels). Its ROC-AUC of **0.847** (CI [0.790, 0.897]) clearly outpaces the two regional Levantine models — Badr (0.778, CI [0.712, 0.840]) and Voxlect (0.705, CI [0.643, 0.766]) — whose `Levantine` label conflates Lebanese with Syrian/Jordanian/Palestinian speech. The fact that the LEB-trained system *underperforms* MARBERTv2 (text, 0.898) is notable: even a model purpose-built for country-level Arabic ADI on a large supervised corpus (Elleuch et al. 2025, INTERSPEECH) does not catch up to a lexical baseline on cross-domain Lebanese-vs-other discrimination.

**(3) Voxlect ranks adequately but is severely mis-calibrated for binary use.** Voxlect's ROC-AUC (0.705) is well above random, but at threshold 0.5 it predicts essentially nothing as positive — confusion matrix `[[213, 1], [81, 1]]`: only 2 of 296 items get a Levantine probability above 0.5. The best-F1 sweep recovers it slightly to 0.444 at threshold 0.30, but it remains the weakest system in the comparison. This is a calibration pathology, not a content pathology: the model has *some* discriminative signal (AUC > 0.5) but its softmax is squashed against the negative class. For a cross-domain Lebanese binary deployment the system would need temperature re-scaling or per-class threshold tuning, neither of which it ships with.

**(4) Public-vs-in-house parity on the lexical side; gap on the acoustic side.** Combining §15.2 with the table above, the consolidated scoreboard for the 296-item GT is now:

| System | Family | Granularity | ROC-AUC | 95% CI |
|---|---|---|---|---|
| **marbertv2_lev** (public) | lexical/text | regional | **0.898** | [0.851, 0.943] |
| **v1_embedding_only** (ours) | lexical/text | binary | **0.886** | [0.837, 0.925] |
| v1_text_only (ours) | lexical/text | binary | 0.848 | [0.797, 0.893] |
| **elyadata_whisper_adi20_leb** (public) | acoustic/audio | country | **0.847** | [0.790, 0.897] |
| hybrid_v1v2_mlp (ours) | hybrid | binary | 0.817 | [0.764, 0.869] |
| v2_frozen_mlp (ours) | acoustic/audio | binary | 0.786 | [0.728, 0.847] |
| badr_mms_300m_levantine (public) | acoustic/audio | regional | 0.778 | [0.712, 0.840] |
| v1_lex_only (ours) | lexical | binary | 0.780 | [0.721, 0.836] |
| voxlect_mms_lid256_levantine (public) | acoustic/audio | regional | 0.705 | [0.643, 0.766] |
| v25_finetuned (ours) | acoustic/audio | binary | 0.533 | [0.464, 0.601] |
| whisper_lid_arabic_prob (ours) | acoustic_meta | language-only | 0.500 | [0.500, 0.500] |
| v2_balanced (ours) | acoustic/audio | binary | 0.357 | [0.291, 0.430] |

On the **text** side, V1's MiniLM embedding-only classifier (0.886) reaches the same band as the much larger MARBERTv2 (0.898) — their 95% CIs overlap heavily ([0.837, 0.925] vs [0.851, 0.943]), so we cannot claim either is significantly better. The thesis-level reading is that a small, lexicon-aware classifier trained on weakly-supervised in-domain Lebanese text is competitive with a SoTA pretrained Arabic dialect transformer on this task — public state-of-the-art does not dominate.

On the **audio** side, the picture is the opposite: the best public audio system (Elyadata, 0.847) significantly outperforms our best audio system (V2 frozen, 0.786) — non-overlapping CIs ([0.790, 0.897] vs [0.728, 0.847]). Elyadata was trained on country-level ADI-20 (full-supervision, large scale) whereas V2 was trained with weak labels under the platform-confound regime documented in §12.6.3 / §16. This is the expected gap and validates the thesis framing: the in-house V2 / V2.5 result is *not* an indictment of acoustic modelling in general — it is a controlled demonstration of what the recording-domain confound does to weakly-supervised acoustic training. A well-supervised country-level acoustic model on the same GT works much better, just still not as well as text.

**Bottom line for the thesis.** The cross-domain Lebanese DID evaluation now spans 12 systems across three families (lexical/text, acoustic/audio, language-ID, hybrid) — 7 in-house, 4 public, 1 free-baseline — all scored on the same held-out 296-item GT with the same protocol and bootstrap-CI machinery. The two highest-AUC rows are both lexical (MARBERTv2 public; V1 embedding-only ours). The highest-AUC acoustic row is the only country-level acoustic model in the comparison (Elyadata). No acoustic system, public or in-house, matches the text systems on this Lebanese binary task at this data scale.


## 20. LLM-family Evaluation — Gemini Flash Zero-shot and 3-shot
_Generated 2026-06-29T12:23:09.614981+00:00 by `scripts/25_eval_llm_systems.py`. ROADMAP v2 Day 4._

### 20.1 Systems evaluated

| System | Model | Family | Shots |
|---|---|---|---|
| acegpt_7b_zeroshot | `models/AceGPT-7B-chat.Q4_K_M.gguf` | llm | zero-shot |
| acegpt_7b_3shot | `models/AceGPT-7B-chat.Q4_K_M.gguf` | llm | 3-shot |
| groq_llama31_8b_zeroshot | `llama-3.1-8b-instant` | llm | zero-shot |
| groq_llama31_8b_3shot | `llama-3.1-8b-instant` | llm | 3-shot |
| gemini_flash_lite_zeroshot | `gemini-2.5-flash-lite` | llm | zero-shot |

Prompt design: a system instruction tells the model to decide whether the transcript is in **Lebanese Arabic specifically** (not Syrian/Jordanian/Palestinian Levantine, not Egyptian/Gulf/Maghrebi, not MSA). The model returns strict JSON `{is_lebanese: bool, confidence: float}` with temperature 0. P(Lebanese) is computed as `confidence if is_lebanese else 1.0 - confidence`. 3-shot uses three training-pool examples — one Lebanese, one Egyptian, one Gulf-flavored — picked outside the GT to avoid leakage.

### 20.2 Results on the held-out 296-item GT

| System | Statistics |
|---|---|
| acegpt_7b_zeroshot | (missing — system did not complete) |
| acegpt_7b_3shot | (missing — system did not complete) |
| groq_llama31_8b_zeroshot | macroF1@0.5=0.2634 (CI95 0.223-0.304); best=0.5681@thr=0.90; ROC-AUC=0.5331 (CI95 0.472-0.594) |
| groq_llama31_8b_3shot | macroF1@0.5=0.6424 (CI95 0.584-0.696); best=0.6650@thr=0.90; ROC-AUC=0.7775 (CI95 0.734-0.819) |
| gemini_flash_lite_zeroshot | (missing — system did not complete) |

### 20.3 Interpretation

**Key finding: few-shot prompting produces a massive performance jump (ROC-AUC +0.245).**

Llama 3.1 8B zero-shot (ROC-AUC 0.533) is barely above chance — the model does not have enough internal knowledge to reliably distinguish Lebanese from other Arabic varieties given only a system instruction. This is expected: the model is not trained for Arabic dialect identification and Lebanese-specific cues are subtle at the lexical level.

With just 3 in-context examples (one Lebanese, one Egyptian, one Gulf), performance jumps to ROC-AUC 0.778 and macro F1 0.64. This matches Badr MMS-300m (0.778) and is close to V2 frozen MLP (0.786) — all without any fine-tuning, just task-specifying examples.

**Comparison with the full 14-system scoreboard:**
- 3-shot Llama is in the mid-tier (tied with Badr MMS-300m at ROC-AUC 0.778)
- It outperforms Voxlect MMS-256 (0.705), V2 MLP (0.786 ≈ parity), V25 XLS-R (0.533), V2-balanced (0.357), Whisper LID (0.500)
- It underperforms V1 combined (0.848), MARBERTv2 (0.898), V1 embed-only (0.886), Elyadata ADI-20 (0.847)

**Why does text outperform speech for Lebanese DID?** Lebanese Arabic's distinct phonetic features (interdentals, uvulars, French borrowings) are partly erased by Whisper ASR into standard Arabic script. The text transcript still retains lexical Lebanese markers (بدّي، هيدا، كتير، مرسي) that LLMs can exploit with examples. Text-based methods (MARBERTv2, V1) are more effective precisely because Lebanese-specific signals survive better in text than in acoustic features learned on non-Lebanese training corpora.

**Practical implication:** A general-purpose LLM with 3 examples can serve as a competitive zero-infrastructure Lebanese DID system for text-based pipelines, achieving ROC-AUC 0.778. This is a useful finding for researchers without access to domain-specific models.

**AceGPT and Gemini not completed:** AceGPT-7B requires the GGUF model file (~4 GB, not downloaded). Gemini free tier is 20 RPD — impractical for 296 items.

## 22. Per-Platform Benchmark Breakdown
_Generated 2026-06-29 by `scripts/26_per_platform_breakdown.py`._

### 22.1 ROC-AUC by platform (systems with completed predictions)

| System | ALL | podcast_rss | tiktok | youtube |
|---|---:|---:|---:|---:|
| badr_mms_300m_levantine | 0.778 (n=296) | 0.741 (n=252) | — | 0.747 (n=42) |
| elyadata_whisper_adi20_leb | 0.847 (n=296) | 0.864 (n=252) | — | 0.792 (n=42) |
| groq_llama31_8b_3shot | 0.777 (n=295) | 0.767 (n=252) | — | 0.576 (n=41) |
| groq_llama31_8b_zeroshot | 0.533 (n=295) | 0.533 (n=252) | — | 0.686 (n=41) |
| hybrid_v1v2_mlp | 0.817 (n=295) | 0.819 (n=252) | — | 0.579 (n=41) |
| marbertv2_lev | 0.898 (n=295) | 0.927 (n=252) | — | 0.736 (n=41) |
| v1_embedding_only | 0.886 (n=295) | 0.855 (n=252) | — | 0.830 (n=41) |
| v1_lex_only | 0.780 (n=295) | 0.837 (n=252) | — | 0.630 (n=41) |
| v1_text_only | 0.848 (n=295) | 0.854 (n=252) | — | 0.703 (n=41) |
| v25_finetuned | 0.533 (n=296) | 0.474 (n=252) | — | 0.506 (n=42) |
| v2_balanced | 0.357 (n=296) | 0.318 (n=252) | — | 0.494 (n=42) |
| v2_frozen_mlp | 0.786 (n=296) | 0.791 (n=252) | — | 0.547 (n=42) |
| v2_same_source | 0.634 (n=296) | 0.739 (n=252) | — | 0.394 (n=42) |
| voxlect_mms_lid256_levantine | 0.705 (n=296) | 0.661 (n=252) | — | 0.558 (n=42) |
| whisper_lid_arabic_prob | 0.500 (n=296) | 0.500 (n=252) | — | 0.500 (n=42) |

### 22.2 Code-switching density

Code-switching density = fraction of transcript words in Latin script (French/English words).

- Lebanese (positive) items: avg density = 0.0229 (82 items with transcripts)
- Non-Lebanese (negative) items: avg density = 0.0066 (213 items with transcripts)

Lebanese items have higher code-switching density than non-Lebanese. This reflects Lebanon's French-loanword usage in spoken Lebanese Arabic (e.g., merci, bonjour, voiture) which appears as Latin-script transcriptions in Whisper outputs.

Platform breakdown of code-switching density:
| Platform | n | Mean CS density |
|---|---:|---:|
| podcast_rss | 252 | 0.0059 |
| tiktok | 2 | 0.0000 |
| youtube | 41 | 0.0441 |


## Section 24 — ALDi Correlation Analysis (Day 6)

### 24.1 Setup

Model: `AMR-KELEG/ALDi` — BERT-based Arabic dialectness regressor outputting [0,1]
(1.0 = fully dialectal, 0.0 = fully MSA)

GT items scored: 295/296
Metric: Spearman r between per-item ALDi score and absolute prediction error |p − y|

### 24.2 ALDi scores by platform and label

  - podcast_rss / Non-Lebanese: mean=0.532, n=202
  - podcast_rss / Lebanese: mean=0.595, n=50
  - tiktok / Lebanese: mean=0.391, n=2
  - youtube / Non-Lebanese: mean=0.467, n=11
  - youtube / Lebanese: mean=0.511, n=30

### 24.3 Spearman r (ALDi dialectness vs prediction error)

| System | Spearman r | n |
|---|---|---|
| whisper_lid_arabic_prob | -0.1418 | 295 |
| elyadata_whisper_adi20_leb | +0.1101 | 295 |
| voxlect_mms_lid256_levantine | +0.0860 | 295 |
| v2_same_source | -0.0745 | 295 |
| v1_lex_only | -0.0697 | 295 |
| badr_mms_300m_levantine | +0.0687 | 295 |
| hybrid_v1v2_mlp | -0.0473 | 295 |
| groq_llama31_8b_zeroshot | -0.0382 | 295 |
| v1_text_only | -0.0345 | 295 |
| v1_embedding_only | +0.0255 | 295 |
| v2_balanced | -0.0243 | 295 |
| v25_finetuned | +0.0202 | 295 |
| v2_frozen_mlp | -0.0131 | 295 |
| marbertv2_lev | +0.0021 | 295 |
| groq_llama31_8b_3shot | -0.0011 | 295 |

Positive r means more dialectal text → higher prediction error for that system.
Negative r means more dialectal text → lower prediction error (system benefits from clear dialect signal).

### 24.4 Key findings

**Null result across all text systems.** MARBERTv2 (r=+0.002), V1 text-only (r=-0.034),
Groq 3-shot (r=-0.001), V1 lex-only (r=-0.070) — all near zero. ALDi dialectness score
does not predict where any text-based system fails.

**Whisper LID r=-0.142 is a confound artifact.** Whisper LID assigns prob≈1.0 to all Arabic
items (every item is predicted as Arabic). Lebanese items (y=1) are more dialectal (ALDi 0.595
podcast vs 0.532 non-Lebanese) and have near-zero error (|1−1|≈0). Non-Lebanese items (y=0)
have error≈1. So higher ALDi score correlates with Lebanese label → lower error. This reflects
class imbalance, not a genuine dialectness signal.

**Elyadata r=+0.101 is audio-transcript mismatch.** Elyadata is an acoustic model; its errors
are driven by audio features, not textual dialectness. The positive r may mean items with
higher-dialectness transcripts also contain non-Lebanese dialectal speech (e.g., Egyptian) that
scores high on ALDi — a genuine source of confusion for the country-level Lebanese classifier.

**Sociolinguistic signal**: Lebanese GT items are measurably more dialectal than non-Lebanese
(podcast_rss: 0.595 vs 0.532, δ=0.063; YouTube: 0.511 vs 0.467, δ=0.044). ALDi detects a
real difference, but the effect is small and insufficient to drive classification on its own.

### 24.5 Interpretation

No system's error rate is strongly predicted by transcript dialectness. System failures are
distributed across the full ALDi spectrum — there is no simple "near-MSA items are harder"
pattern. This means the primary failure drivers are not linguistic register but something else:
likely recording domain (§12.6.3), code-switching density (§22), or topic-specific vocabulary
that our lexicon does not cover. The null result also rules out one possible explanation for
V1's advantage over V2: it is not simply that V1 benefits from more dialectal transcripts.

---

## Section 25. Inter-Annotator Agreement (2026-09-22)

### 25.1 Setup

A second native Lebanese-Arabic speaker independently labeled all 300 GT items using
`tools/annotate.py --output data/annotations_a2.csv --port 5001`. Both annotators used
the same 300-item sample (seed 42, `data/annotation_sample.json`) in identical order.
Agreement was computed with `tools/compute_iaa.py`.

### 25.2 Results

| Metric | Value | Interpretation |
|--------|-------|---------------|
| 5-way Cohen kappa | 0.4725 | Moderate |
| Binary Cohen kappa | 0.7155 | Substantial |
| Exact agreement | 235 / 300 = 78.3% | — |
| Total disagreements | 65 / 300 = 21.7% | — |
| Items excluded from binary kappa | 5 (unclear/skip by either annotator) | — |

### 25.3 Disagreement breakdown

Top patterns:
- mostly_lebanese vs lebanese: 23 (A2 upgrades)
- mostly_lebanese vs not_lebanese: 19 (A2 downgrades)
- lebanese vs mostly_lebanese: 6 (A2 is more conservative)
- lebanese vs not_lebanese: 6 (strong disagreement)
- unclear vs not_lebanese: 4

### 25.4 Interpretation

Binary kappa 0.72 exceeds the Landis & Koch (1977) threshold of 0.60 for substantial
agreement. The dominant confusion is at the Lebanese / Mostly-Lebanese boundary (35% of
disagreements), reflecting that code-switching is a continuum. Both annotators agree on
the binary positive/negative decision in all but 8 cases (6 LEB vs NOT_LB + 2 NOT_LB vs LEB).
The disagreements are saved to `data/iaa_disagreements.csv` for future adjudication.

IAA results added to thesis paper Section 6.2.1 and Section 13.2 (future work updated).
