# Building an Open Lebanese Arabic Audio Corpus and an Acoustic Lebanese-vs-Other-Arabic Dialect Classifier

**Charlene El Khoury Eid** — Lebanese American University, Byblos
*Master's Thesis Draft, 2026*

---

## Abstract

Lebanese Arabic, like most regional dialects of Arabic, lacks the open, multi-speaker speech resources that have driven recent advances in Arabic NLP. This thesis (i) describes the construction of a queue-driven, multi-stage pipeline for collecting Lebanese Arabic audio from open platforms (YouTube, podcasts, TikTok), (ii) establishes a 300-item manually annotated ground-truth test set covering five register categories, (iii) trains a text-only baseline classifier (lexical features + multilingual sentence embeddings) using weak supervision from trusted-channel metadata and lexical scoring, and (iv) develops an acoustic classifier built on mean-pooled wav2vec2-XLS-R-300m representations and trained against a contrastive non-Lebanese set drawn from ADI17 (Egyptian, Gulf) and FLEURS (Modern Standard Arabic). On the held-out 300-item test set, the text-only baseline (V1) achieves macro F1 0.74 and ROC-AUC 0.85, while the acoustic classifier (V2) achieves macro F1 0.38 and ROC-AUC 0.79 at default decision threshold — a substantial collapse from its in-pool validation performance (macro F1 0.86). Three remediations — threshold tuning, per-(platform, label) class balancing, and hybrid V1+V2 features — fail to recover V2 above the V1 baseline. We diagnose V2's failure as a *recording-domain confound*: the training pool's positive and negative classes correlate with distinct recording domains (broadcast, read-prompt, podcast), and the frozen XLS-R encoder learns the easier acoustic-domain signal in preference to dialect signal. We report this negative finding as a methodological contribution: the standard frozen-encoder recipe for self-supervised speech embeddings does not transfer cleanly to Arabic dialect identification when training-class composition is correlated with recording-environment, and we propose end-to-end fine-tuning, broadcast-only contrastive sets (e.g., MGB-2 instead of FLEURS), and speaker-normalized features as candidate remedies for future work. Beyond performance numbers, the work contributes a documented end-to-end pipeline (open-source), a curated ground-truth set, and a reproducibility log of practical constraints — gated speech corpora (Mozilla Common Voice withdrawn from HuggingFace), dataset-loader-script deprecations (`datasets >= 4.0`), and content-filtered file hosts on residential networks — that are themselves contributions to the dialect-identification literature.

---

## 1. Introduction

Arabic is spoken by over 400 million people in 22+ countries, but its written-form unity (Modern Standard Arabic, MSA) masks substantial spoken-form variation. The standard taxonomy organizes spoken Arabic into broad regional groups — Maghrebi, Egyptian, Levantine, Gulf, Yemeni, and Iraqi [@bouamor2018madar] — each containing several country-level dialects. Within Levantine, Lebanese, Syrian, Palestinian, and Jordanian Arabics share the bulk of phonology and core lexicon while differing in prosody, function-word inventory, and high-frequency colloquialisms.

Automatic dialect identification (DID) for Arabic has been studied at the regional level (e.g., the MGB-3 challenge [@ali2017mgb3]) and at the country level (the ADI17 benchmark spans 17 countries; [@ali2019mgb5]). However, the resources that drive these benchmarks are dominated by broadcast and read speech, and Lebanese — though present in ADI17 — is rarely the focus of dedicated open-corpus or open-classifier work. The practical consequence is that researchers and product teams who need a "is this clip Lebanese Arabic?" classifier today face a choice between (a) repurposing a 17-way country classifier and thresholding a single class output, with all the calibration concerns that introduces, or (b) hand-curating bespoke training data, which is expensive and seldom released.

This thesis takes the second path but releases the artifacts: a documented collection pipeline, a labeled corpus, two trained classifiers (a text-only baseline and an acoustic-feature classifier), and a 300-item manually annotated test set. The pipeline is queue-driven and reproducible; the corpus is built from open platforms (YouTube channels, public podcasts, TikTok) and augmented with research-grade dialect labels imported from ADI17 and FLEURS. The two classifiers are explicitly comparable on the same held-out test, allowing us to measure the value of acoustic features over text-only features in a controlled way.

Our contributions are:

1. **A queue-driven Lebanese-Arabic audio collection pipeline**, comprising discovery (YouTube RSS, podcast RSS, TikTok), download with normalization, screening transcription, weak-label assignment via trusted-channel metadata and lexical scoring, and integration of external research datasets. The pipeline is reproducible from public sources and survives platform-specific quirks (RSS-vs-API discovery, MP3 vs FLAC storage tradeoffs, etc.).

2. **A curated ground-truth test set** of 300 items, manually annotated across five register categories (lebanese, mostly_lebanese, not_lebanese, unclear, skip), with stratified sampling over the pipeline's confidence tiers and a custom Flask annotation tool.

3. **A text-only baseline classifier** built on five lexical features (Lebanese, MSA, Egyptian, Gulf, Syrian word counts plus a strong-marker count) and a 384-dimensional multilingual sentence embedding [@reimers2019sbert], trained via weakly supervised binary labels and evaluated on the held-out test.

4. **An acoustic dialect classifier** trained on 1024-dimensional wav2vec2-XLS-R-300m utterance embeddings [@babu2022xlsr], with a contrastive set assembled from ADI17 (Egyptian, Gulf, additional Lebanese) and FLEURS (MSA from Egyptian speakers), and evaluated on the same held-out test as the baseline.

5. **A documented methodology** that catalogs negative results — including the empirical confirmation that ADI17's training partition contains zero MSA items, and the practical workarounds for recently-introduced gating on Mozilla Common Voice and `datasets`-library loading-script deprecations — useful for any future researcher working in this corner of speech.

The remainder of this paper is structured as follows. Section 2 surveys related work in Arabic dialect resources, dialect identification, and self-supervised speech representations. Section 3 describes the collection pipeline and resulting dataset. Section 4 details preprocessing and audio normalization. Section 5 describes the weak-supervision strategy and ground-truth construction. Section 6 covers the text-only baseline. Section 7 presents the acoustic classifier and its comparison against the baseline. Section 8 discusses limitations, with particular attention to single-annotator bias, label noise, and the recording-environment confound between the FLEURS MSA class and the ADI17 broadcast classes. Section 9 outlines future work.

---

## 2. Related Work

### 2.1 Arabic dialect resources

The MADAR Arabic Dialect Corpus [@bouamor2018madar] provides parallel sentences across 25 Arab city dialects, but is text-only and small in audio terms. The MGB-2 challenge [@ali2016mgb2] released approximately 1,200 hours of Al Jazeera broadcast Arabic, predominantly MSA, with utterance-level transcriptions; it is the de facto MSA broadcast resource but requires QCRI-managed access. The MGB-3 challenge [@ali2017mgb3] introduced four-way regional dialect data (Egyptian, Gulf, Levantine, North African). The ADI17 dataset [@ali2019mgb5] — used heavily in this thesis — provides 17 country-level Arabic dialect classes (including Lebanese) totaling roughly 3,000 hours of broadcast speech, distributed via HuggingFace as Parquet shards with embedded audio bytes. Salameh et al. [@salameh2018finegrained] explored fine-grained Arabic dialect identification at the city level using the MADAR corpus.

Notably, **ADI17 does not include MSA as a class**: the dataset is country-dialect-only, and MSA is treated as a separate register. We confirm this empirically in Section 3 by exhaustive scan of the train, validation, and test splits.

For MSA specifically, options are scarcer than commonly assumed. The Arabic Speech Corpus [@halabi2016msa] is high-quality but single-speaker, which presents a fatal acoustic confound for any dialect-identification work: a classifier trained against single-speaker MSA cannot disentangle MSA-as-register from this-particular-voice. Mozilla Common Voice [@ardila2020commonvoice] historically provided multi-speaker Arabic read-prompt audio, but Mozilla migrated all Common Voice resources off HuggingFace in October 2025 to its own Mozilla Data Collective portal. We use FLEURS [@conneau2022fleurs] as our MSA source instead — multi-speaker, open, citable, available via HuggingFace, and explicitly designed as a multilingual evaluation resource.

### 2.2 Arabic dialect identification

Early work used acoustic features (MFCCs, i-vectors) with GMM/SVM classifiers. End-to-end neural approaches with convolutional architectures and language embeddings followed [@shon2018adi]. The ADI17 baseline [@ali2019mgb5] reports country-level accuracy in the 70-80% range across the 17 classes, with substantial confusion within regional groups (e.g., among Levantine country variants). More recent work fine-tunes transformer-based speech encoders on dialect data; XLS-R [@babu2022xlsr] and HuBERT [@hsu2021hubert] are the dominant backbones, with mean-pooling or attention-pooling over time being the standard utterance-level aggregation.

For Lebanese specifically, dedicated dialect classifiers are rare in the open literature. The most closely related work is country-level binary thresholding over multi-class country classifiers, which suffers from calibration and class-imbalance issues. To our knowledge this thesis is among the first to release a dedicated open Lebanese-vs-other Arabic classifier alongside its training data.

### 2.3 Self-supervised speech representations and dialect ID

wav2vec 2.0 [@baevski2020wav2vec2] established self-supervised pretraining for speech, learning representations that transfer well to downstream tasks (ASR, language identification, speaker verification). XLS-R [@babu2022xlsr] extends this to 128 languages and 436K hours of pretraining audio, including Arabic. For dialect identification specifically, the literature has converged on a recipe of (i) freeze the pretrained encoder, (ii) take the last hidden states, (iii) mean-pool over time using attention masks, (iv) train a small classifier head (logistic regression or shallow MLP) on the resulting fixed-length utterance embeddings. We follow this recipe.

### 2.4 Weak supervision for speech corpora

Manually labeling speech is expensive. Weak supervision strategies — pulling labels from metadata (channel identity, source venue), from text-domain heuristics, or from coarse classifier outputs — have been used for ASR [@radford2023whisper], dialect ID, and topic classification. The trade-off is well-known: weak labels are abundant but noisy; explicit measurement of that noise via a held-out manually annotated test set is the standard correction. We measure noise rates in Section 5 and use the ground-truth test only for evaluation, never for training.

---

## 3. Data Collection Pipeline and Dataset

### 3.1 Architecture

The pipeline is queue-driven: items flow through a SQLite database (`data/queue.db`) with explicit status transitions. Each pipeline stage is a standalone script that reads items at a given status and writes them to the next. The status flow is:

```
DISCOVERED → DOWNLOADED → SCREENED ──► WEAK_POSITIVE  (metadata trusted)
                                   ──► WEAK_NEGATIVE  (lexical scoring)
                                   ──► POTENTIAL_LB / BORDERLINE_LB / REJECTED  (v1 model)
```

This design admits incremental, restartable runs; failed stages can be retried; new stages can be added without disrupting earlier ones.

### 3.2 Sources

**YouTube** — 521 trusted Lebanese channel IDs were curated by hand, then expanded via RSS-feed discovery (no API quota required). The full set of channels covers news, talk shows, comedy, music, and educational content. RSS feeds yield video metadata (title, channel, duration); audio is downloaded via `yt-dlp` and normalized to mono 16 kHz with FFmpeg loudnorm, capped at 1200 seconds.

**Podcasts** — Approximately 151 Arabic-language RSS feeds, mixed dialects, are followed via the PodcastIndex API and direct enclosure download. Many feeds are pan-Arab in scope, providing natural negatives for the Lebanese classifier; lexical scoring (Section 5) labels these.

**TikTok** — Limited; 13 items collected. The platform's authentication and rate limits make systematic discovery brittle.

**External research datasets:**
- **ADI17** [@ali2019mgb5]: 1,000 LEB items as additional research-grade Lebanese positives; 1,000 EGY and 5,000 Gulf items (KSA, KUW, UAE, QAT, OMA combined) as contrastive negatives. All items extracted from the ADI17 dev+test splits, which contain country-dialect labels assigned by QCRI.
- **FLEURS** [@conneau2022fleurs], `ar_eg` configuration: 798 unique MSA items. The FLoRes-101 sentences read by FLEURS speakers are written in literary Arabic (al-Fuṣḥā), so although the speakers are based in Egypt, the read content is MSA. This provides the MSA contrastive class.

### 3.3 Dataset size

As of 2026-04-29 the corpus comprises:

| Status / Class | Count | Source |
|----------------|------:|--------|
| WEAK_POSITIVE  | 3,232 | YouTube (Lebanese channels) |
| POTENTIAL_LB   | 1,769 | podcast_rss, model-scored |
| BORDERLINE_LB  |   313 | podcast_rss, model-scored |
| WEAK_NEGATIVE  | 1,005 | podcast_rss, lexical scoring |
| REJECTED       | 2,583 | YouTube + podcast_rss, model-scored |
| ADI17 LEB (extra positives) | 1,000 | ADI17 |
| ADI17 EGY (negatives) | 1,000 | ADI17 |
| ADI17 Gulf (negatives) | 5,000 | ADI17 |
| FLEURS MSA (negatives) |   798 | FLEURS |
| Held-out ground truth | 300 | Manual annotation |

Total items with audio + screening transcripts (Phase 1): ~8,902.
Total contrastive (Phase 2): 7,798 (1,000 LEB + 6,798 non-LB).
Items selected for v2 acoustic embedding extraction: 14,177 (all labeled Phase-1 items + Phase-2 contrastive + 300 GT).

### 3.4 Storage

Audio is stored as FLAC for YouTube content (lossless; ~20% of the WAV size after compression) and as MP3 (96 kbps, mono, 16 kHz, loudnorm-normalized) for podcast, ADI17, and FLEURS audio. The total corpus footprint is approximately 95 GB.

A migration from WAV to FLAC was performed mid-project after the working drive reached 100% capacity; this is documented as a methodological decision in the FINDINGS log.

### 3.5 Empirical observation: ADI17 contains no MSA

We initially planned to extract MSA items from the ADI17 train split, on the assumption that the 17-way country labels would include an MSA class concentrated in 1-3 dialect-sorted Parquet shards. A footer-only Parquet scan was inconclusive (column statistics were not populated for the dialect column), and a partial column-projection scan over flaky home connection produced no MSA. Running an exhaustive scan of all 40 train shards on a Colab GPU runtime confirmed that **ADI17 contains zero MSA items** across 990,821 total rows; the 17 dialect classes are all country-level (ALG, EGY, IRA, JOR, KSA, KUW, LEB, LIB, MAU, MOR, OMA, PAL, QAT, SDN, SYR, UAE, YEM). This is consistent with the ADI17 paper's framing of MSA as a separate register, but is sometimes overlooked by users browsing the dataset's HuggingFace page. This thesis switches to FLEURS for the MSA contrastive class.

---

## 4. Preprocessing and Audio Normalization

All audio entering the pipeline — regardless of source — is normalized to a uniform format to avoid recording-environment confounds:

- **Channels:** mono (1 channel)
- **Sampling rate:** 16 kHz
- **Loudness:** normalized via FFmpeg's `loudnorm` filter
- **Storage codec:** FLAC for YouTube WAV inputs (lossless); MP3 96 kbps for everything else

For screening (Section 5), three 20-second random chunks are extracted per item and stored as WAV in `data/samples/<id>_chunk<N>.wav`. Faster-Whisper [@radford2023whisper] transcribes each chunk with the `base` model. We benchmarked Whisper `medium` vs. `small` vs. `base` on five real audio chunks (two Lebanese YouTube, three non-Lebanese TikTok) and found that `base` achieves identical language-detection confidence (lang_prob = 1.00) at 15× the throughput of `medium` (1.3 sec/chunk vs. 20 sec/chunk on CPU); transcription quality at `base` is sufficient for downstream lexical scoring even though it is below the bar one would set for production ASR.

For the v2 acoustic classifier (Section 7), we further trim each clip to 10 seconds at 64 kbps mono 16 kHz MP3 — the standard window for utterance-level wav2vec2/XLS-R embeddings in dialect-ID literature. This trimming is uniform across all classes (Lebanese candidates, ADI17 dialects, FLEURS MSA, ground-truth items), so any acoustic-domain differences across classes survive only at the source-recording level, not from differing clip lengths or codecs introduced by the pipeline.

---

## 5. Weak Supervision and Ground-Truth Construction

### 5.1 Weak labeling

**Metadata-based positives (WEAK_POSITIVE).** Items whose YouTube channel ID matches the curated trusted-Lebanese-channel list are labeled positive. This produces 3,232 candidates. Spot-checking against the ground-truth set (Section 5.3) reveals that approximately 33% of WEAK_POSITIVE items are not actually Lebanese-dialect content — they are MSA news segments, formal interviews, or music videos posted to Lebanese channels. We address this label noise in two ways: (i) a lexical-verification filter `strong_lb_hits ≥ 1` for v1 training, raising precision from 67% to 81% at the cost of dropping ~38% of items; (ii) the v2 acoustic classifier learns acoustic correlates that are likely more robust to this content drift.

**Lexical-based negatives (WEAK_NEGATIVE).** Items with screening-transcript dialect score `raw_score < 0` (formula in Section 5.2) and Whisper language probability ≥ 0.70 are labeled negative. This produces 1,005 negatives, predominantly from podcast feeds with non-Lebanese content (Egyptian and Gulf interviews, MSA news segments). We initially tried the criterion `lb == 0` (zero Lebanese-marker matches) but it produced only 2 negatives because the Lebanese lexicon contains pan-Arabic words that occur in nearly all Arabic transcripts; this is itself a useful methodological observation.

### 5.2 Lexical scoring

Curated word lists for Lebanese, MSA, Egyptian, Gulf, and Syrian Arabic [^src:`src/dialect/lexicons.py`] are used to compute, for each transcript:

```
raw_score = lb × 1.8 − msa × 0.6 − egy × 1.0 − gulf × 1.0 − sy × 0.5
final_score = max(0, min(1, raw_score / 5.0))
```

The five lexical counts plus `final_score` form the lexical features for the v1 model. A separate `strong_lb_hits` feature counts occurrences of high-precision Levantine markers (شو، ليش، هيك، هلق، عنجد، بدي، كتير، وين، هون، هيدا، هيدي، هدول).

A known limitation of this lexical scheme is that the strong markers are **Levantine**, not uniquely Lebanese — they also appear in Syrian and Palestinian Arabic. Lexical features can therefore distinguish Lebanese from Egyptian, Gulf, or MSA, but not from other Levantine variants. Acoustic features (Section 7) are intended in part to address this.

### 5.3 Ground-truth construction

A 300-item sample was drawn from the corpus with stratified random sampling over the pipeline's confidence tiers: 90 POTENTIAL_LB, 60 BORDERLINE_LB, 60 REJECTED, 45 WEAK_POSITIVE, 45 WEAK_NEGATIVE. The sample is over-weighted toward the discriminative bands (BORDERLINE_LB, REJECTED) where classifier disagreement is most informative; the confident bands (WEAK_POSITIVE, WEAK_NEGATIVE) are sampled to verify the noise rate of weak labels.

Annotation was performed by the thesis author using a custom Flask web application with HTML5 audio playback, keyboard shortcuts, and CSV persistence. Five labels were used: `lebanese`, `mostly_lebanese` (Lebanese with code-switching to MSA or other), `not_lebanese`, `unclear`, `skip`. The sample order was deterministically shuffled (random seed 42) to prevent the annotator from labeling an entire confidence tier in a single sitting.

For binary evaluation, we map `lebanese` and `mostly_lebanese` to the positive class (1) and `not_lebanese` to the negative class (0); `unclear` and `skip` are excluded. This yields 295 evaluable items: 82 positive, 213 negative.

A known limitation is **single-annotator bias**: there is no inter-annotator agreement metric and no formal kappa. We discuss this in Section 8.

---

## 6. v1: Text-Only Baseline Classifier

### 6.1 Architecture

The v1 model is a logistic regression trained on a 389-dimensional feature vector:

- 5 lexical features: `[lb, msa, strong_lb_hits, msa_ratio_core, final_score]`, derived from the screening transcript and the `lexicon_score()` function (Section 5.2)
- 384-dimensional sentence embedding from `paraphrase-multilingual-MiniLM-L12-v2` [@reimers2019sbert] applied to the screening transcript

We use `class_weight="balanced"` to compensate for the slight imbalance between positives and negatives in the training pool. Training uses scikit-learn `LogisticRegression` with default L2 regularization.

Training data after the lexical-verification filter (`strong_lb_hits ≥ 1` on WEAK_POSITIVE): 2,003 positives + 1,651 negatives (1,005 WEAK_NEGATIVE + 646 REJECTED items with valid transcripts).

### 6.2 Validation results

On a stratified 80/20 train/validation split of the training pool, v1 achieves:
- Accuracy: 89%
- ROC-AUC: 0.9643

These are *in-pool* validation numbers; they are subject to label noise (the pool itself is weakly labeled) and therefore optimistic relative to real-world precision.

### 6.3 Results on the held-out 300-item ground truth

This is the apples-to-apples comparison number for v2. Of the 300 GT items, 295 have valid binary labels and parsable screening transcripts. v1's threshold-independent metrics:

- ROC-AUC: **0.8477** (drop from 0.9643 on validation, consistent with label noise in the training pool)
- PR-AUC: **0.7123**

At the threshold maximizing macro F1 (0.70):

- Accuracy: 0.766
- Macro F1: **0.7397**
- Negative class (precision/recall/F1): 0.909 / 0.751 / 0.823 (n=213)
- Positive class (precision/recall/F1): 0.555 / 0.805 / 0.657 (n=82)
- Confusion matrix (rows=true, cols=pred): `[[160, 53], [16, 66]]`

The ROC-AUC drop from validation to held-out is the noise-correction signal: weak training labels overestimate v1's discrimination by roughly 0.12 AUC. This finding alone is methodologically significant — any future work that reports validation numbers without a held-out set will systematically overstate generalization.

### 6.4 Error patterns

(To be expanded with the v1 error-analysis script's output.)

The dominant error mode at threshold 0.70 is **false positives**: 53 negative items predicted as positive, vs. 16 false negatives. This is consistent with the lexicon-overlap problem (Section 5.2): Egyptian, Gulf, and MSA transcripts that happen to contain pan-Arabic words trigger the Lebanese lexical features, and the sentence embedding alone cannot recover the dialect distinction.

---

## 7. v2: Acoustic Classifier

### 7.1 Backbone selection

We use `facebook/wav2vec2-xls-r-300m` [@babu2022xlsr], the standard 300M-parameter multilingual self-supervised speech encoder pretrained on 436K hours of speech across 128 languages including Arabic. Alternatives considered and rejected:

- **wav2vec2-base** (95M, English-only pretrain): smaller and faster, but Arabic transfer is untested and would require fine-tuning.
- **HuBERT-base** [@hsu2021hubert]: comparable size and quality to wav2vec2-base, same constraints.
- **Whisper-large-v2 encoder** [@radford2023whisper]: already in the pipeline for ASR, but the encoder-only forward pass is heavier per-utterance.
- **MFCC + classical ML**: a useful sanity-check baseline; we leave it for future work.
- **elgeish/wav2vec2-large-xlsr-53-arabic**: an Arabic-finetuned 300M variant. Same size class as XLS-R, no clear advantage, and using a generic multilingual backbone leaves more room for the classifier head to specialize without contaminating the pretrained features with task-specific Arabic ASR signal.

### 7.2 Feature pipeline

For each clip we:
1. Load 10 seconds of audio at 16 kHz mono.
2. Forward through XLS-R-300m with the model in `eval()` mode (no fine-tuning of the backbone).
3. Mean-pool the last hidden state over the time axis, masked by the attention mask via `_get_feat_extract_output_lengths` so that padded frames do not contribute to the average.
4. The output is a 1024-dimensional fixed-length utterance embedding.

This is the standard recipe for dialect ID with frozen self-supervised encoders. We do not fine-tune XLS-R end-to-end for two reasons: (i) the training data, while large for a Lebanese corpus, is small relative to what XLS-R was pretrained on and end-to-end fine-tuning risks catastrophic forgetting of the multilingual representations; (ii) frozen features keep the comparison to v1 controlled — only the feature extractor changes, the classifier head is the same logistic regression family.

### 7.3 Classifier head

We train two classifier heads on the 1024-d embeddings and report both:

- **Logistic regression** (`class_weight="balanced"`, `max_iter=2000`, default L2): same family as v1 for direct comparison.
- **Multi-layer perceptron** (one hidden layer of 256 units, early stopping on a 10% internal validation slice): a slightly more expressive head that can pick up nonlinearities in the embedding space.

For each head we report (i) validation metrics on a stratified 80/20 split of the training pool, and (ii) held-out metrics on the 300-item GT test. The "winner" is selected by held-out macro F1.

### 7.4 Results

V2 was trained on 13,624 items (5,866 positive, 7,758 negative) drawn from WEAK_POSITIVE, POTENTIAL_LB, ADI17 LEB (positive class) and WEAK_NEGATIVE, ADI17 EGY/Gulf, FLEURS MSA (negative class). The 300 ground-truth items were excluded from training. Two heads were trained on the same 80/20 stratified train/val split.

**On the validation split (in-pool):**

| Head | Acc | Macro F1 | ROC-AUC |
|------|----:|---------:|--------:|
| LogReg | 0.822 | 0.819 | 0.890 |
| MLP-256 | 0.858 | 0.856 | 0.933 |

These numbers look strong — and they are, *within* the training-pool distribution.

**On the held-out 300-item ground-truth test:**

| Head | Acc | Macro F1 | ROC-AUC |
|------|----:|---------:|--------:|
| LogReg | 0.331 | 0.304 | 0.642 |
| MLP-256 | 0.392 | 0.379 | 0.787 |

V2 collapses on the held-out test. The val→GT drop in accuracy (0.86 → 0.39 for the MLP head) is the largest in the pipeline, far exceeding the val→GT drop observed for v1 (0.89 → 0.77).

The confusion matrix for V2 MLP on the GT is `[[37, 177], [3, 79]]` — V2 predicts "Lebanese" on 256 of 296 items (87%), catching almost all true positives (96% recall) at the cost of a false-positive rate of 83% on the negatives. The decision boundary that worked on validation does not generalize to the GT distribution.

### 7.5 v1 vs v2 comparison

| Model | Held-out Acc | Held-out Macro F1 | Held-out ROC-AUC |
|-------|-------------:|------------------:|-----------------:|
| **V1 text-only** (Section 6) | **0.766** | **0.740** | **0.848** |
| V2 acoustic LogReg | 0.331 | 0.304 | 0.642 |
| V2 acoustic MLP-256 | 0.392 | 0.379 | 0.787 |

V1 substantially outperforms V2 on every metric. This is a **negative result** that is itself a methodological contribution: the standard recipe for dialect-ID with frozen self-supervised speech embeddings (mean-pool last hidden state, train a small head) does not transfer cleanly when the training-class composition correlates with recording-domain.

### 7.6 Diagnosis: training-class / recording-domain confound

The V2 training pool's positive class is dominated by ADI17 LEB broadcast clips and YouTube channels with Lebanese-specific recording profiles; the negative class is dominated by ADI17 broadcast (EGY/Gulf), FLEURS read-prompt audio, and lexically-scored podcast clips. Each *class label* is correlated with a specific *recording domain* in the training pool. The frozen XLS-R encoder, never updated for dialect, produces utterance embeddings that capture acoustic-domain features (broadcast vs read-prompt vs podcast) at least as strongly as dialect features. The classifier head learns the easier signal first.

The ground-truth test items, however, are drawn entirely from the in-pipeline podcast and YouTube collection. They lack the ADI17 broadcast acoustic signature and the FLEURS read-prompt acoustic signature. When the recording-domain shortcut is removed, V2's discrimination collapses. ROC-AUC of 0.79 (MLP) on the GT confirms there is *some* dialect signal in the embeddings — V2 is not random — but the calibration learned during training is wrong for the test distribution.

This is consistent with a known caveat in the dialect-ID and speaker-ID literature: frozen self-supervised speech encoders inherit whatever invariances were salient in pretraining, and Arabic dialect is not one of them. Without targeted fine-tuning or careful cross-source balancing, the encoder's embeddings are dominated by acoustic-environment features that happen to be far more salient than dialect for the pretraining objective.

### 7.7 Remediation experiments

To probe whether V2's collapse is fixable without re-architecting, three remediations were tried. Full numbers are in FINDINGS Section 12.6; held-out GT macro F1 summary:

| Approach | Held-out ROC-AUC | Macro F1 (default thr) | Macro F1 (best snooped) |
|----------|----------------:|----------------------:|------------------------:|
| V1 text-only (baseline) | 0.848 | — | **0.740** (thr 0.70) |
| V2 acoustic MLP | 0.787 | 0.379 | 0.701 (thr 0.85) |
| V2 retrained, per-source balanced | 0.357 | 0.370 | 0.420 (thr 0.55) |
| Hybrid LR (V1 + V2) | 0.729 | 0.548 | 0.620 (thr 0.85) |
| Hybrid MLP (V1 + V2) | 0.817 | 0.512 | 0.724 (thr 0.90) |

**Threshold tuning (Experiment A).** Sweeping V2's decision threshold and choosing the value that maximizes held-out macro F1 — this is *snooping* on the test set and is reported as a sensitivity analysis, not as a fair number. Even at the snooped optimum (threshold 0.85), V2 alone reaches macro F1 0.701, **still below V1's 0.740**.

**Per-source class balancing (Experiment B).** V2 was retrained with sample weights such that each (platform, label) pair contributed equal total weight, removing the recording-domain shortcut at training time. The result is striking: held-out ROC-AUC drops to 0.357 — below random (0.5). This tells us that **the residual dialect signal in V2 was riding on the recording-domain shortcut**. When we explicitly remove the shortcut at training, the model retains no useful dialect signal. This is consistent with V2 having learned recording-domain features primarily and dialect features only as a noisy byproduct of the training-class composition.

**Hybrid V1 + V2 (Experiment C).** Concatenating V1's 389-d text features with V2's 1024-d acoustic embeddings (1413-d total) and training LogReg and MLP heads. The hybrid model only saw the 5,427 items that have screening transcripts (excluding ADI17 + FLEURS, which were imported as DOWNLOADED and never transcribed). On the GT, the best hybrid (MLP at snooped threshold 0.90) reaches macro F1 0.724 — within 2 F1 points of V1 alone. **Hybrid does not improve over V1.** For this binary Lebanese-vs-other task on this dataset, frozen-encoder acoustic embeddings are subsumed by the lexical and sentence-embedding signal.

### 7.8 Verdict

V1 is the strongest classifier produced by this work. V2 with the standard frozen-encoder + small head recipe does not transfer to held-out evaluation, and its failure mode is empirically diagnosed as recording-domain confound rather than weak signal. Hybrid features fail to improve on V1.

The negative result is itself a contribution. Practitioners attempting the same recipe — collecting research-grade dialect labels from one corpus and combining with weakly-labeled in-distribution data — should expect this confound and design accordingly: end-to-end fine-tuning of the encoder, careful cross-source balancing, speaker-normalized features (i-vectors, x-vectors), or multi-class formulations are all candidate paths and are deferred to future work.

---

## 8. Limitations and Discussion

### 8.1 Single annotator

The 300-item ground-truth test was annotated by the thesis author alone. There is no inter-annotator agreement statistic. For Lebanese-vs-other classification, label difficulty is highest in the `mostly_lebanese` band (50 of 300 items, 16.7%); a second annotator on this subset would meaningfully improve confidence in the test-set labels. We mark this as future work.

### 8.2 Lexical overlap

The Lebanese lexicon contains pan-Arabic words; the Lebanese strong markers are Levantine, not uniquely Lebanese. Lexical features alone cannot distinguish Lebanese from Syrian, Palestinian, or Jordanian Arabic. This is part of the motivation for v2 acoustic features, and a known constraint we accept.

### 8.3 Recording-domain confound — confirmed empirically

The FLEURS MSA class is read-prompt audio recorded by Egyptian-based speakers; the ADI17 dialect classes are broadcast audio; the WEAK_POSITIVE/POTENTIAL_LB items are YouTube and podcast audio. This introduces recording-environment differences between training classes that turn out to be the dominant signal V2 learns. We document this *a priori* concern in Section 7.6 and confirm it empirically via Experiment B (Section 7.7): when we explicitly remove the per-source class imbalance via sample weights, V2's held-out ROC-AUC drops to 0.36 — below random — indicating that the residual dialect signal in V2 was riding on the recording-domain shortcut, and removing the shortcut eliminates the model's discrimination.

Mitigation candidates we did not pursue:
- **MGB-2 broadcast MSA** instead of FLEURS, which would acoustically match ADI17 broadcast at the cost of QCRI-managed access. We attempted to integrate this and deferred (Section 8.7 of FINDINGS).
- **Speaker-normalized features** (i-vectors, x-vectors), which explicitly normalize for recording environment.
- **End-to-end fine-tuning** of the encoder, which we declined for the frozen-features comparison but is the obvious next remedy.

Additionally, the FLEURS `ar_eg` configuration uses Egyptian speakers; while the read content is MSA, residual Egyptian phonetic features may bleed into the embeddings. A multi-config FLEURS alternative (averaging across `ar_eg`, `ar_jo`, `ar_lb` if available) is left for future work.

### 8.4 Ground-truth size

300 items is sufficient for stable per-class precision and recall estimates with reasonable confidence intervals (95% CI half-width on the order of 5-7 percentage points for class-wise rates), but is small relative to evaluations on standard speech benchmarks. Scaling the test set would tighten our reported numbers.

### 8.5 Deployment readiness

The classifiers are research artifacts, not production systems. V1 (text-only) is the strongest model from this work, but its 0.66 positive-class F1 and 25% false-positive rate at threshold 0.70 mean it is not ready for an automated "filter Lebanese clips" tool without human review. V2 in any of its forms is not deployable: even the snooped best-threshold v2 underperforms v1, and the diagnosed recording-domain confound makes the model's behavior on out-of-distribution audio unpredictable. Threshold calibration here was done with respect to the held-out 300-item test; deployment in any new distribution (e.g., live YouTube comment audio, broadcast news) would require fresh calibration on a sample of that distribution.

---

## 9. Future Work

Ranked roughly by expected payoff for the dialect-ID task on this corpus:

- **End-to-end fine-tuning of XLS-R** — the most direct fix for V2's failure. Allowing the encoder weights to update during training would let dialect features compete with recording-domain features for representational capacity. This is the highest-priority remediation given the empirical diagnosis in Section 7.6/7.7.

- **MGB-2 broadcast MSA** — substitute MGB-2 for FLEURS to put MSA on the same broadcast acoustic substrate as ADI17 EGY/Gulf. Eliminates one of the two main domain confounds. Requires QCRI-managed access.

- **Speaker-normalized features.** Compute i-vectors or x-vectors per clip and concatenate or replace the XLS-R embedding. Speaker representations explicitly factor out recording channel, which is what V2 lacks.

- **Multi-class extension.** Replace the binary Lebanese-vs-other formulation with a four-way classifier (Lebanese / MSA / Egyptian / Gulf). The held-out test set already supports this via its raw labels. Multi-class often outperforms binary for dialect ID because the negative class becomes more homogeneous.

- **Larger Lebanese-specific corpus.** The pipeline's discovery and download stages are reusable; with more YouTube channel curation and podcast feed coverage, the Lebanese-positive pool could grow by an order of magnitude. This may also reduce the per-source confound.

- **Inter-annotator agreement.** Recruit a second Lebanese-speaking annotator for a 50-item overlap on the GT test, compute Cohen's kappa, and revise borderline labels accordingly.

- **Live evaluation.** Calibrate the v1 thresholds on a fresh sample of live YouTube/podcast audio and report deployed precision/recall.

---

## References

Full BibTeX entries are in [`paper/references.bib`](references.bib). In-text citations use Pandoc-style `[@bibkey]` syntax, which compiles to numbered or author-year format depending on the chosen output (e.g., LaTeX with `natbib` for ACL/IEEE styles).

**Cited in this draft:**
- `@bouamor2018madar` — MADAR Arabic Dialect Corpus
- `@ali2016mgb2` — MGB-2 broadcast Arabic
- `@ali2017mgb3` — MGB-3 regional dialect challenge
- `@ali2019mgb5` — ADI17 / MGB-5 Arabic dialect identification benchmark
- `@halabi2016msa` — Modern Standard Arabic Phonetics for Speech Synthesis (rejected source for MSA contrastive due to single-speaker confound)
- `@ardila2020commonvoice` — Common Voice (rejected as MSA source due to October 2025 withdrawal from HuggingFace)
- `@conneau2022fleurs` — FLEURS (selected MSA source)
- `@babu2022xlsr` — XLS-R (v2 acoustic backbone)
- `@baevski2020wav2vec2` — wav2vec 2.0 (background for XLS-R)
- `@hsu2021hubert` — HuBERT (alternative SSL speech backbone)
- `@radford2023whisper` — Whisper (screening transcription)
- `@reimers2019sbert` — Sentence-BERT (v1 sentence embedding origin)
- `@salameh2018finegrained` — fine-grained Arabic dialect ID at city level
- `@shon2018adi` — CNN + language embeddings dialect recognition baseline

**In `references.bib` but not yet cited in-text** (will be added in implementation/methodology subsections): `@wolf2020transformers`, `@paszke2019pytorch`, `@pedregosa2011sklearn`, `@lhoest2021datasets`, `@inoue2021camelbert`, `@wang2020minilm`, `@habash2010introduction`.

**Flagged `% VERIFY`** in `references.bib` and need exact venue/page confirmation before final submission: `@harrat2019arabic`, `@abdul-mageed2018you`, `@el-haj2018habibi`.

---

## Appendix A: Pipeline Reproduction

See `PIPELINE.md` in the repository root for the full step-by-step run order, scripts, and current pipeline state. The pipeline is reproducible from public sources only; the only credentials required are a YouTube Data API key (for optional API-based discovery; RSS-based discovery does not require one) and PodcastIndex API credentials.

## Appendix B: Negative Results and Reproducibility Friction

Several practical constraints are worth recording for any researcher who attempts to reproduce or extend this work:

- **Mozilla Common Voice was withdrawn from HuggingFace in October 2025** and migrated to the Mozilla Data Collective. Any code using `mozilla-foundation/common_voice_*` paths will fail with `DatasetNotFoundError` regardless of authentication state.
- **`datasets >= 4.0` removed support for script-based dataset loaders.** FLEURS still ships with a `fleurs.py` loading script, so loading FLEURS via modern `datasets` versions raises `RuntimeError: Dataset scripts are no longer supported`. The workaround is to pin `datasets == 2.21.0` together with `fsspec <= 2024.12.0`.
- **gofile.io's free-tier API was paywalled** at some point before our usage; the `/contents/{code}` endpoint returns `error-notPremium` without paid credentials, even though uploads still work. The website-token (`wt`) needed for unauthenticated access is now extracted from heavily obfuscated JavaScript, making programmatic access fragile.
- **HuggingFace Windows symlink limitations** mean that without Developer Mode or admin Python, the cache uses file copies instead of symlinks. The first model load reads through Windows Defender file scanning, which can take ~30 minutes for a 1.2 GB model. Subsequent loads with a warm OS disk cache complete in seconds.

These are not flaws in the methodology; they are the kinds of moving-target practicalities that any field-current paper should record so that reviewers and downstream users do not blame the methodology when the underlying tooling has changed.
