# Lebanese Audio Collector — Pipeline

## Active platforms

| Platform      | Discovery script              | Download support | Status   |
|---------------|-------------------------------|------------------|----------|
| YouTube       | 01a, 01b                      | ✓                | Active   |
| Podcast (RSS) | 01c                           | ✓                | Active   |
| TikTok        | 01d                           | ✓                | Active (limited) |
| ADI17         | 07 (HF dataset import)        | ✓                | Imported |
| FLEURS (MSA)  | colab_extract_msa_cv (Colab) → 10 (local import) | ✓ | Imported |
| Instagram     | 01e                           | —                | Disabled |
| Facebook      | 01f                           | —                | Disabled |

---

## Pipeline phases

The pipeline now spans three phases:

1. **Phase 1: Lebanese-only collection (steps 0–6)** — discover, download, transcribe, weak-label, score with v1 text-only classifier.
2. **Phase 2: Contrastive non-Lebanese collection (steps 7–10)** — ADI17 (EGY/Gulf/LEB) + FLEURS (MSA).
3. **Phase 3: Acoustic v2 classifier (steps 11–14)** — extract wav2vec2-xls-r-300m embeddings, train acoustic classifier, evaluate on held-out ground truth.

---

## Phase 1 — Lebanese collection

| Step | Script                          | Input                       | Output                                                |
|------|---------------------------------|-----------------------------|-------------------------------------------------------|
| 0    | `00_show_queue.py`              | —                           | Queue summary                                         |
| 1    | `01a/b/c/d_discover_*.py`       | —                           | DISCOVERED                                            |
| 2    | `02_download_audio.py`          | DISCOVERED                  | DOWNLOADED                                            |
| 3    | `03_transcribe_screening.py`    | DOWNLOADED                  | SCREENED                                              |
| 3b   | `03b_assign_weak_labels.py`     | SCREENED + trusted metadata | WEAK_POSITIVE                                         |
| 3c   | `03c_assign_lexical_negatives.py` | SCREENED w/o metadata     | WEAK_NEGATIVE (raw_score < 0)                         |
| 4    | `04_score_dialect.py`           | SCREENED                    | POTENTIAL_LB / BORDERLINE_LB / REJECTED               |
| 5    | `05_train_dialect_model.py`     | WEAK_POSITIVE + WEAK_NEGATIVE | `models/dialect_classifier.joblib` (v1 text-only) |
| 6    | `06_reset_podcast_downloads.py` | error states                | retry queue                                           |

---

## Phase 2 — Contrastive dataset

| Step | Script                                | Input                                  | Output                                       |
|------|---------------------------------------|----------------------------------------|----------------------------------------------|
| 7    | `07_download_adi17_contrastive.py`    | HF: `ArabicSpeech/ADI17` dev+test      | DOWNLOADED rows in queue (EGY, Gulf, LEB)    |
| 8    | `08_download_adi17_msa.py`            | HF: ADI17 train (40 files)             | **superseded** — 0 MSA in ADI17 (FINDINGS 8.2.3) |
| 9    | `09_find_msa_files.py`                | HF Parquet footers                     | Diagnostic only (no usable column stats)     |
| —    | `colab_extract_msa.py`                | HF ADI17 train (Colab GPU)             | Empirically confirmed 0 MSA → retired        |
| —    | `colab_extract_msa_fleurs.py`         | First-pass FLEURS attempt (Colab)      | Failed silently → see `colab_extract_msa_cv.py` |
| —    | `colab_extract_msa_cv.py`             | HF: `google/fleurs` (`ar_eg`) (Colab)  | `fleurs_msa.zip` ← **PRIMARY MSA SOURCE**    |
| 10   | `10_import_colab_msa.py`              | `data/fleurs_msa.zip`                  | DOWNLOADED rows in queue (MSA)               |

**Final contrastive dataset (2026-04-27):**
- 1,000 LEB (ADI17 — extra positives) · 1,000 EGY · 5,000 Gulf · 798 MSA (FLEURS) = 7,798 contrastive items.

---

## Phase 3 — Acoustic v2 classifier

| Step | Script                                      | Input                              | Output                                          |
|------|---------------------------------------------|------------------------------------|-------------------------------------------------|
| 11   | `11_prep_embeddings_audio.py`               | queue.db + `data/raw_audio/`       | `data/audio_clips_for_embed/*.mp3` (30s @ 96k)  |
| 11b  | `11b_compact_embeddings_zip.py`             | `data/audio_clips_for_embed/`      | `data/audio_clips_compact/*.mp3` (10s @ 64k) + zip |
| 12   | `12_extract_embeddings_local.py`            | clips + manifest                   | `data/embeddings.parquet` (1024-d, mean-pooled) |
| 13   | `13_load_embeddings.py`                     | `data/embeddings.parquet` + queue.db | `data/embeddings_with_labels.parquet`         |
| 13b  | `13b_eval_v1_on_gt.py`                      | v1 model + annotations + transcripts | v1 baseline numbers appended as FINDINGS Section 9.0 |
| 14   | `14_train_classifier_v2.py`                 | embeddings_with_labels parquet     | `models/dialect_classifier_v2_acoustic.joblib` + FINDINGS Section 12 |
| 15   | `15_remediation_v2.py`                      | embeddings_with_labels + transcripts + v2 model | `models/dialect_classifier_v2_balanced.joblib`, `models/dialect_classifier_hybrid.joblib`, `models/remediation_v2_results.json` + FINDINGS Section 12.6 (V2 remediation experiments: threshold tuning, per-source balancing, hybrid V1+V2). |
| —    | `colab_finetune_xlsr.py`                   | Colab T4 GPU + audio_for_embeddings_compact.zip | `xlsr_finetuned.zip` (fine-tuned model + eval log). End-to-end fine-tuning of XLS-R-300m, addressing the recording-domain confound diagnosed in 12.6. |
| 16   | `16_eval_finetuned_xlsr.py`                | `data/xlsr_finetuned.zip` + annotations.csv | `models/xlsr_finetuned/`, `models/v25_eval_local.json`, FINDINGS Section 13 (V2.5 fine-tuned XLS-R results + final scoreboard). |

**Model**: `facebook/wav2vec2-xls-r-300m` (Babu et al. 2022). Mean-pool last hidden state → 1024-d utterance embedding.

**Hardware decision (2026-04-29)**: switched from Colab GPU to local CPU due to Colab + external file-host friction (gofile, transfer.sh, pixeldrain all blocked or gated on the user's network). Local 22-core CPU runs ~6-10 hours overnight, fully resumable.

**Held-out evaluation**: 300 manually annotated items in `data/annotations.csv` are excluded from training and used as the test set for v2.

---

## Backfill / utility scripts

```bash
python scripts/backfill_youtube_metadata.py --limit 100
python scripts/backfill_podcast_metadata.py --limit 100
python scripts/05_reset_for_rescoring.py
python scripts/06_reset_podcast_downloads.py
python scripts/reset_weak_labels.py
python scripts/rerun_youtube_error_downloads.py
```

---

## Typical run order (full pipeline)

```bash
# Phase 1
python scripts/01b_discover_youtube_rss.py
python scripts/01c_discover_podcast_rss.py
python scripts/02_download_audio.py
python scripts/03_transcribe_screening.py
python scripts/03b_assign_weak_labels.py
python scripts/03c_assign_lexical_negatives.py
python scripts/05_train_dialect_model.py
python scripts/04_score_dialect.py

# Phase 2
python scripts/07_download_adi17_contrastive.py
# (Run colab_extract_msa_cv.py in Google Colab, download fleurs_msa.zip)
python scripts/10_import_colab_msa.py

# Phase 3
python scripts/11_prep_embeddings_audio.py
python scripts/11b_compact_embeddings_zip.py
python scripts/12_extract_embeddings_local.py        # ~6-10h on CPU
python scripts/13_load_embeddings.py
python scripts/14_train_classifier_v2.py             # appends to FINDINGS.md
```

---

## Current state (as of 2026-05-02)

- Audio storage: ~95 GB in `data/raw_audio/` (FLAC for YouTube, MP3 for podcast/ADI17/FLEURS)
- Trimmed audio for v2: `data/audio_clips_compact/` (~14K × 10s @ 64k = ~950 MB)
- Embedding extraction: **complete** — 14,177 embeddings in `data/embeddings.parquet`, 0 failures
- V2 training data: 5,866 positives + 7,758 negatives
- Held-out test: 300 ground-truth items (296 evaluable after excluding `unclear`/`skip`)
- **V1 (text-only) on held-out GT:** ROC-AUC 0.848, best macro F1 0.740 (thr 0.70)
- **V2 (acoustic) on held-out GT:** ROC-AUC 0.787, default macro F1 0.379 (collapse from val 0.86)
- **Hybrid (V1+V2) on held-out GT:** ROC-AUC 0.817, snooped-best macro F1 0.724 (still below V1)
- **V1 is the strongest model.** V2 + remediation experiments are documented as a methodological negative finding (recording-domain confound; FINDINGS Section 12.6).
