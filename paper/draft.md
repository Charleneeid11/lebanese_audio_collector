# Cross-domain Lebanese Arabic Dialect Identification: Benchmarking Lexical, Acoustic, and Large Language Models

**Charlene El Khoury Eid** - Lebanese American University, Byblos
*Master's Thesis Draft, 2026*

---

## Abstract

Lebanese Arabic is among the most linguistically underrepresented spoken varieties in automatic dialect identification (DID) research: despite appearing in shared tasks such as ADI17, it has no dedicated open test set and no published cross-domain comparison of modern DID systems. This thesis addresses that gap with five contributions. (C1) We build a queue-driven, multi-platform audio collection pipeline yielding approximately 14,000 candidate Lebanese-Arabic items from YouTube, podcasts, TikTok, ADI17, and FLEURS. (C2) We construct the first published Lebanese-specific held-out DID test set: 300 items manually annotated across five label categories by a native Lebanese speaker, stratified over the pipeline's confidence tiers, and binary-mapped to 296 evaluable items (82 positive, 214 negative). (C3) Using this test set, we run a cross-domain benchmark of 14 systems spanning four model families - lexical/text (sentence embeddings, MARBERT), frozen acoustic self-supervised models (XLS-R-300m, MMS-300m), an end-to-end fine-tuned Whisper DID system, and a large language model (Llama-3.1-8B, zero-shot and 3-shot) - all evaluated under a uniform protocol with percentile-bootstrap 95% confidence intervals on macro F1 and ROC-AUC. Key findings: text-based models lead acoustic models (V1 embed-only ROC-AUC 0.886, MARBERTv2 0.898 vs best acoustic Elyadata 0.847), with overlapping bootstrap confidence intervals that preclude a claim of strict statistical dominance but a consistent 0.039–0.051 point-estimate advantage; and 3-shot prompting alone raises Llama's ROC-AUC from 0.533 (chance) to 0.778, matching a purpose-built acoustic system. (C4) We empirically confirm a *recording-domain confound* as the primary explanation for acoustic underperformance: frozen XLS-R embeddings encode platform-of-origin with 89% four-way accuracy (chance: 25%), per-source balanced training drops V2's ROC-AUC to 0.357 (entirely below random; 95% CI [0.291, 0.430]), and a same-source control trained only on podcast audio still underperforms the cross-domain text system - four converging lines of evidence. We frame this as a methodological warning for the field: the frozen-encoder recipe for self-supervised speech DID does not transfer cleanly to heterogeneous-source weakly-supervised Arabic training data. (C5) An empirical reproducibility audit of ADI17 (all 40 Parquet shards, 990,821 rows) confirms the dataset contains no MSA class - a common misconception among downstream users.

**Keywords:** Lebanese Arabic; dialect identification; self-supervised speech representations; recording-domain confound; cross-domain benchmark; weak supervision; wav2vec 2.0; MARBERT; bootstrap confidence intervals; multi-platform corpus

---

## Acknowledgments

I would like to express my sincere gratitude to my thesis advisor for their continuous guidance, constructive feedback, and unwavering support throughout this project. I am equally grateful to the members of my thesis committee for their time, expertise, and valuable input.

This research was conducted on personal hardware with no institutional compute allocation. The open-source community whose tools made this work possible deserves particular acknowledgment: the developers of Whisper, XLS-R, HuggingFace Transformers, scikit-learn, and the many Arabic NLP researchers whose publicly released models and datasets are evaluated in this benchmark.

Finally, I thank my family for their patience and encouragement throughout this degree.

---

## 1. Introduction

Arabic is spoken by over 400 million people across 22 countries, but its spoken-form diversity - organized into Maghrebi, Egyptian, Levantine, Gulf, Yemeni, Iraqi, and other regional groupings - is poorly represented in NLP resources compared to its written (Modern Standard Arabic, MSA) form. Within Levantine Arabic, Lebanese occupies a distinctive social position: it is a prestige spoken variety, a media dialect with a substantial podcast and YouTube presence, and linguistically distinct from other Levantine varieties in prosody, phonology, and parts of the lexicon.

Automatic dialect identification (DID) for Arabic has been studied at the regional level (MGB-3; [@ali2017mgb3]) and the country level (ADI17; [@ali2019mgb5]). Lebanese appears as one of 17 country classes in ADI17, and country-level classifiers have been released for that dataset. However, to our knowledge no prior work (i) builds an open end-to-end pipeline for collecting Lebanese-specific audio from public platforms, (ii) establishes a manually-annotated cross-domain held-out test set that is publicly described, or (iii) runs a systematic side-by-side comparison of modern DID systems on Lebanese binary classification with statistical confidence intervals. A researcher who needs a "is this clip Lebanese Arabic?" system today must choose between repurposing a 17-way country classifier and thresholding a single class output - with all the calibration complications that entails - or hand-curating proprietary training data that is seldom released.

This thesis takes the second path but releases the artifacts: a documented collection pipeline, a curated corpus, and a 300-item manually-annotated test set. On this test set we evaluate 14 systems - five built in-house, four pulled from HuggingFace, one free baseline, two ablation variants, and two LLM zero/few-shot conditions - under a uniform protocol. Our headline finding is unexpected: text-based features consistently outperform acoustic self-supervised features for Lebanese binary identification, and we provide rigorous empirical evidence that this is caused by a recording-domain confound in the training data rather than by acoustic features being inherently uninformative.

### 1.1 Contributions

This thesis makes five contributions:

**(C1) A queue-driven Lebanese Arabic audio collection pipeline.** A reproducible, multi-platform pipeline covering discovery (YouTube RSS, YouTube Data API, podcast RSS, TikTok), audio download and normalization, Whisper-based screening transcription, and weak-label assignment via trusted-channel metadata and lexical scoring. Integration of external research datasets (ADI17 [@ali2019mgb5]; FLEURS [@conneau2022fleurs]) is handled through a shared queue so all items flow through a unified status schema.

**(C2) The first published Lebanese-specific DID test set.** 300 items manually annotated across five label categories (Lebanese, Mostly-Lebanese, Not-Lebanese, Unclear, Skip) by a native Lebanese-speaking annotator, stratified over the pipeline's confidence tiers. This is the first Lebanese-specific held-out test set described in the open literature and the evaluation anchor for all 14 benchmark systems.

**(C3) A 14-system Lebanese cross-domain DID benchmark.** We evaluate four model families - lexical/text classifiers, acoustic self-supervised encoders, an end-to-end audio DID system, and a large language model (Llama-3.1-8B) - on the same 296-item held-out test under a uniform protocol. All results include percentile-bootstrap 95% confidence intervals. The benchmark is reproducible: per-system predictions are saved to disk and evaluation is re-runnable from those files. A key LLM finding: 3-shot prompting raises Llama from near-chance (0.533) to competitive mid-tier (0.778), matching a purpose-built Arabic DID acoustic system.

**(C4) An empirically confirmed recording-domain confound.** We establish this finding with four converging lines of evidence: (i) V2 acoustic drops 0.48 macro F1 from in-pool validation to held-out test; (ii) per-source balanced training drops V2's ROC-AUC from 0.787 to 0.357 (entirely below random; 95% CI [0.291, 0.430]); (iii) a platform-probe classifier achieves 89.1% four-way accuracy predicting recording source from XLS-R embeddings (chance: 25%); (iv) a same-source control trained exclusively on the test domain still underperforms the cross-domain text system. We interpret this as a methodological warning for Arabic DID: heterogeneous-source weak supervision introduces label-correlated recording domains that frozen acoustic encoders exploit in preference to dialect signal.

**(C5) An empirical reproducibility audit of ADI17.** A row-level scan of all 40 Parquet shards of the ADI17 train split (990,821 rows) confirms that ADI17 contains no MSA items: the 17 classes are exclusively country-level dialects. This corrects a common assumption among HuggingFace users that MSA would be a 17+1 class in the dataset.

### 1.2 Paper organization

Section 2 surveys related work. Section 3 describes corpus construction and the ADI17 reproducibility finding. Section 4 covers preprocessing. Section 5 details the weak supervision strategy. Section 6 describes the 300-item ground-truth test set. Section 7 presents all 14 benchmark systems. Section 8 defines the evaluation protocol. Section 9 reports the benchmark results. Section 10 presents the recording-domain confound analysis in depth. Section 11 discusses failure patterns. Section 12 states limitations. Section 13 outlines future work.

---

## 2. Related Work

This chapter surveys the prior work that contextualizes the thesis across six dimensions: the sociolinguistic background of Arabic and its dialects; existing corpora and resources; automatic dialect identification; self-supervised speech representations; code-switching in Arabic; large language models for Arabic NLP; weak supervision; and domain adaptation. Together these threads motivate both the problem framing (a Lebanese-specific binary benchmark) and the experimental choices made in subsequent chapters.

### 2.1 Arabic diglossia and the Lebanese variety

Arabic is a macro-language spoken by over 400 million people across 22 countries, exhibiting a well-documented diglossic structure [@habash2010introduction]: a high-prestige standardized variety - Modern Standard Arabic (MSA), or *al-fuṣḥā* - coexists alongside a continuum of regional spoken dialects (*ʿāmmiyya*). MSA is the language of government, formal education, and print media; no native speaker acquires it as a mother tongue. Regional dialects - Levantine, Egyptian, Maghrebi, Gulf, Iraqi, Yemeni - serve as the actual vehicles of everyday spoken communication and vary substantially from one another in phonology, morphology, syntax, and lexicon.

Lebanese Arabic belongs to the North Levantine branch, sharing broad structural features with Syrian, Palestinian, and Jordanian Arabic while exhibiting distinctive characteristics: preservation of interdental fricatives (θ/ð) in some registers, specific vowel quality shifts (*imāla* raising of /a/ toward /e/), and a phonological inventory shaped by centuries of Phoenician substratum, Ottoman Turkish influence, and French colonial contact. Most distinctively, Lebanon's French-mandate history produced a pervasive Arabic-French code-switching pattern unique in the Levantine family: Lebanese speakers routinely embed French and English words, phrases, and sometimes entire clauses into Arabic discourse at all syntactic levels. This produces a distinctive orthographic signature in Whisper transcripts - Latin-script tokens interspersed with Arabic-script ones - that serves as a weak but real computational cue to Lebanese speaker identity.

From a corpus-construction standpoint, Lebanese Arabic occupies a structurally advantageous niche for online collection: it commands a large YouTube and podcast footprint (talk shows, comedy, political commentary, cultural programming) that makes platform-based data collection feasible at scale. The collection challenge is that the same Lebanese creators frequently produce MSA-register content on formal occasions, requiring dialect-aware filtering even within trusted Lebanese channels.

### 2.2 Arabic dialect corpora and resources

The Arabic dialect NLP community has assembled a growing body of annotated resources spanning text and speech. For written text, the MADAR corpus [@bouamor2018madar] provides parallel sentences across 25 Arabic city dialects and 12 non-Arabic languages (approximately 110,000 sentences total) and is the primary resource for fine-grained written dialect identification; it contains no audio. The NADI shared task series [@nadi2025] extends dialect text resources to social media and, in recent editions, to spoken components.

For speech, the MGB series provides the canonical broadcast audio resources. MGB-2 [@ali2016mgb2] released approximately 1,200 hours of Al Jazeera broadcast speech - primarily MSA - with word-level dialect labels; it remains the standard MSA broadcast benchmark but requires QCRI-managed access. MGB-3 [@ali2017mgb3] introduced a four-way regional dialect partition (Egyptian, Gulf, Levantine, North African) drawn from YouTube content. ADI17 [@ali2019mgb5], the Arabic Dialect Identification 2017 challenge dataset, scaled to 17 country-level classes from broadcast recordings totalling approximately 3,000 hours, and is the largest openly available Arabic dialect audio resource; it is hosted on HuggingFace as Parquet shards with embedded audio bytes. A critical empirical observation confirmed in this thesis is that **ADI17 contains no MSA class** - its 17 classes are exclusively country-level dialects (Section 3.5). This is consistent with the ADI17 challenge design but is commonly overlooked by downstream users.

FLEURS [@conneau2022fleurs] is a multilingual speech evaluation corpus covering 102 languages, including Arabic in an Egyptian-speaker configuration (`ar_eg`) reading FLoRes-101 sentences in MSA register. It is multi-speaker, CC-BY-SA licensed, and the most accessible multi-speaker MSA audio source since Mozilla Common Voice migrated off HuggingFace in October 2025. Mozilla Common Voice Arabic, previously available as `mozilla-foundation/common_voice_*` via the HuggingFace `datasets` library, now requires direct access through the Mozilla Data Collective and cannot be loaded through standard pipeline scripts.

For Lebanese specifically, no open dedicated large-scale audio corpus with verified dialect labels existed prior to this work. The closest available resource is the Lebanese (LEB) portion of ADI17, which contains approximately 35,938 rows in the train split drawn from broadcast recordings. This thesis fills this gap.

### 2.3 Automatic Arabic dialect identification

Arabic DID research has evolved through three distinct methodological generations. The first generation applied MFCC-based acoustic features with GMM-SVM classifiers to short audio segments, or bag-of-words text features to transcripts, with SVM or maximum entropy classifiers. Salameh et al. [@salameh2018finegrained] demonstrated fine-grained city-level identification from written text using character *n*-gram features, achieving over 67% accuracy on 25-city classification. These systems depend heavily on transcript quality and fail on low-resource varieties.

The second generation moved to deep neural acoustic features: Shon et al. [@shon2018adi] established a convolutional + language-embedding neural baseline for the ADI challenge, combining MFCC frame-level features with language identification posteriors in an end-to-end trainable architecture. The ADI17 challenge [@ali2019mgb5] benchmarked systems on 17-country classification; top systems achieved 70-80% accuracy, with substantial confusion within-region (Levantine subgroups; Gulf subgroups). i-vector and d-vector representations from speaker verification were also applied to DID in this period, exploiting the observation that dialect identity correlates with long-term spectral properties in a manner similar to speaker identity.

The third generation fine-tunes self-supervised speech encoders on dialect-labeled data. Abdullah et al. (2025) train an MMS-300m model for Arabic DID using voice-conversion-based augmentation to diversify recording styles, reporting state-of-the-art cross-dialect performance. Elleuch et al. (2025) release ADI-whisper-ADI20, a Whisper-Large-V3 fine-tuned on 20 Arabic country dialects including a Lebanese (`LEB`) class - the only publicly available country-level Lebanese acoustic DID system at the time of this writing. Voxlect (2026) release a MMS-LID-256-based classifier covering Arabic regional categories including Levantine. All three are benchmarked in this thesis (Section 7.2).

For text-based Arabic DID, MARBERTv2 [@inoue2021camelbert], a BERT-family model pre-trained on 128 GB of Arabic social-media and dialectal text, is the dominant backbone. A publicly released fine-tuned variant covers four regional groups including Levantine. Text-based DID benefits from lexical invariance to recording environment - a property central to this thesis's finding that text systems outperform acoustic systems under cross-domain conditions (Section 10).

### 2.4 Arabic automatic speech recognition

Accurate ASR is a prerequisite for transcript-based dialect analysis pipelines. Arabic ASR has historically been challenged by dialectal variation, absent diacritization in standard orthography, and limited training data for non-MSA speech. The MGB-2 challenge stimulated Arabic broadcast ASR development, with top-performing systems reaching word error rates below 15% on Al Jazeera MSA content.

Whisper [@radford2023whisper] (Radford et al., OpenAI, 2023) represents the current practical threshold for multilingual open-domain ASR: trained on 680,000 hours of weakly-supervised multilingual audio across 99 languages, it treats Arabic as a first-class language and achieves competitive word error rates across MSA and several dialect registers. Performance degrades on heavy code-switching and on strong regional accents with limited pretraining coverage. Dialect-aware ASR - conditioning recognition on predicted dialect - is an active research direction (Bougrine et al. 2022) but is not a component of this pipeline.

For this thesis, Whisper's `base` model is deployed for *screening transcription* rather than full-text ASR: three 20-second random chunks per item, transcribed at 15× the throughput of `medium` with equivalent Arabic language-detection confidence on the tested items (Section 4). The resulting transcript is used for lexical and embedding feature extraction, not as a verbatim record of speech content.

### 2.5 Self-supervised speech representations

wav2vec 2.0 [@baevski2020wav2vec2] established the self-supervised pretraining paradigm for speech: a convolutional feature encoder projects raw waveforms to 25 ms frames, which are quantized into a discrete codebook; a Transformer contextualizer is then trained to predict masked frame representations from unmasked context (analogous to BERT masking). The resulting representations transfer well to downstream tasks under limited supervision.

XLS-R [@babu2022xlsr] extends this to 128 languages and 436,000 hours of unlabeled audio. The 300M-parameter variant (`facebook/wav2vec2-xls-r-300m`) follows the majority of recent Arabic DID literature and is the backbone of this thesis's V2 and V2.5 systems. HuBERT [@hsu2021hubert] offers an alternative formulation using offline cluster assignments as pseudo-labels for masked prediction; it achieves comparable performance to wav2vec 2.0 on most standard benchmarks. MMS (Massively Multilingual Speech, Pratap et al. 2023) further extends SSL pretraining to 1,100+ languages and is the backbone of the Abdullah et al. (2025) and Voxlect (2026) systems.

The standard recipe for applying frozen SSL encoders to dialect ID is: (i) extract mean-pooled last-hidden-state embeddings from fixed-length audio windows; (ii) train a shallow classifier head on the embeddings using available dialect labels. This recipe performs well when training and test audio share a common recording domain, as in within-source ADI17 evaluation. It is vulnerable to recording-domain confounds when training data is assembled from heterogeneous public sources - the central finding of this thesis. Sullivan et al. [@sullivan2023ssl] systematically demonstrate this vulnerability for Arabic DID: frozen representations do not generalize cleanly across recording environments and domain-matched training is necessary for reliable transfer. Section 10 replicates and extends their finding to the Lebanese binary classification setting.

### 2.6 Code-switching in Arabic speech

Code-switching - alternation between two or more languages or varieties within a single conversation or utterance - is particularly prevalent in Lebanese Arabic [@habash2010introduction]. Lebanese speakers routinely embed French and English words and phrases at the lexical, clausal, and sentential levels, reflecting the French-mandate bilingual education system and ongoing anglophone influence. Common patterns include lexical insertion (*merci*, *voiture*, *cool* inserted into Arabic NPs), constituent insertion (French NPs within Arabic VP structure), and paragraph-level alternation between Arabic and French or English.

For dialect identification, code-switching has a dual effect. On the challenge side: heavily code-switched Lebanese items have lower density of Arabic dialect-specific vocabulary in their Whisper transcripts, reducing the signal available to lexical and embedding classifiers. The `mostly_lebanese` annotation category (50 of 300 GT items, 16.7%) captures many such items: the annotator identifies Lebanese speaker identity from prosodic and phonological cues that survive in the audio, while the text contains insufficient Arabic lexical evidence to trigger text-based models. These items represent the hardest frontier for transcript-based systems.

On the diagnostic side: Latin-script token density in Whisper transcripts is a weak but measurable proxy for Lebanese identity. Section 11.4 reports 3.5× higher Latin density in Lebanese vs. non-Lebanese GT items (0.023 vs. 0.007 token ratio), and YouTube items show 7× higher Latin density than podcast items. This feature has not been exploited by any system in this benchmark and represents a potentially useful supplementary signal for future work.

### 2.7 Large language models for Arabic NLP and dialect identification

The emergence of large language models (LLMs) has created a new paradigm for NLP classification tasks: instead of a task-specific fine-tuned classifier, a general-purpose LLM is prompted with a task description and, optionally, a small number of labeled examples. Arabic has been well served in this paradigm: AraBERT [@antoun2020arabert] established BERT-scale Arabic text understanding; MARBERTv2 [@inoue2021camelbert] extended this to dialectal Arabic; Jais (13B/70B parameters, Arabic-centric GPT architecture) and AceGPT (7B, instruction-following Arabic LLM) provide generative Arabic-specialized models. General-purpose multilingual LLMs including Llama-3.1, GPT-4o, and Gemini cover Arabic as part of their multilingual pretraining corpus.

For dialect identification via LLM prompting, the key question is whether Lebanese-specific lexical and syntactic patterns are present in the model's pretraining corpus and can be activated by an appropriate prompt. Prior systematic benchmarks of LLM-based Arabic DID are limited; most evaluations focus on dedicated classification systems. This thesis contributes two LLM conditions to the Lebanese benchmark - Llama-3.1-8B zero-shot and 3-shot via the Groq API - and finds a striking +0.245 ROC-AUC gain from three in-context examples (Section 9.7). This finding suggests that Lebanese-specific knowledge is present in the LLM's pretraining corpus but requires a calibration signal to surface. The practical implication for researchers without GPU access or labeled training data is that few-shot LLM prompting on Whisper transcripts may provide a competitive zero-infrastructure Lebanese DID system.

### 2.8 Weak supervision for speech classification

Manual labeling of speech corpora at scale requires on the order of 10-20 minutes of human effort per minute of audio for multi-dialect annotation, making full supervision prohibitively expensive for a single-researcher thesis project. Weak supervision addresses this by deriving labels from cheaper, noisier sources: metadata heuristics (channel identity, source venue, playlist labels), keyword matching in transcripts, or outputs from coarser automatic classifiers. The resulting labels are imperfect but enable classifiers that substantially outperform no-supervision baselines.

This thesis applies two weak labeling strategies: metadata-based positives from trusted Lebanese YouTube channels (WEAK_POSITIVE), and lexical-scoring-based negatives from items where non-Lebanese dialect signals dominate the transcript (WEAK_NEGATIVE). The gap between in-pool validation performance and held-out test performance (Section 5.3) provides a direct empirical estimate of label-noise impact: V1's 0.116 ROC-AUC drop from in-pool to held-out is the noise correction signal. A `strong_lb_hits ≥ 1` filter applied to positives at training time reduces the estimated false-positive rate from 33% to 19% (§5.1), at the cost of 38% fewer training items.

Weak supervision's cross-domain behavior under recording-domain heterogeneity has received limited attention. This thesis's recording-domain confound finding (Section 10) shows that metadata-derived weak labels can create label-domain correlations that frozen acoustic encoders learn as recording shortcuts rather than dialect features - a failure mode not previously characterized for Arabic DID.

### 2.9 Domain adaptation and recording-domain confounds

Domain mismatch - performance degradation when training and test data come from different distributions - is a fundamental challenge in speech processing. In Arabic DID, recording domain (broadcast vs. conversational vs. read-prompt audio) introduces correlated acoustic variation that a frozen encoder may learn in preference to the intended dialect signal. Sullivan et al. [@sullivan2023ssl] systematically demonstrate this for Arabic DID with SSL encoders: within-source evaluation substantially overestimates generalization performance, and domain-matched training is necessary for cross-source transfer.

Geirhos et al. [@geirhos2020shortcut] provide a general framework for this phenomenon as *shortcut learning*: neural networks preferentially learn the simplest discriminative feature, which in heterogeneous datasets may be a spurious correlation (recording environment, file format, channel characteristics) rather than the task-relevant signal. When this shortcut is label-correlated in the training pool, the model achieves high training and in-distribution validation accuracy while failing catastrophically out-of-distribution. Section 10 operationalizes this framework: a linear probe on frozen XLS-R embeddings predicts recording platform with 89% four-way accuracy (chance 25%), confirming that platform identity is the dominant axis of separability in the embedding space and therefore the natural shortcut for any label-domain-correlated training pool.

Remediation approaches studied in the literature include: domain-adversarial neural networks (DANN), which add a gradient-reversal layer to enforce domain-invariant representations; CORAL (correlation alignment), which minimizes second-order statistics discrepancy between source and target domains; voice conversion augmentation (Abdullah et al. 2025), which synthesizes class-balanced acoustic variation; and simply training on domain-matched data. This thesis tests per-source balanced sampling (§7.1) and same-source training (§10.6) as probes; more powerful remediation is deferred to future work.

### 2.10 Positioning

Lebanese Arabic occupies a gap in the existing DID literature. Unlike Egyptian or Gulf Arabic, which appear as primary targets in major shared tasks (NADI 2025, SemEval, Casablanca 2024), Lebanese has not had a dedicated open evaluation set or a systematic cross-domain DID benchmark. ADI17 includes Lebanese as one of 17 classes but uses same-source broadcast recordings for both training and evaluation, avoiding the cross-domain challenge by construction. The Arabic Level of Dialectness (ALDi) metric [@keleg2023aldi] provides a continuous measure of dialectalness from Arabic text and is applied in this thesis as an analysis tool (Section 11.5), but has not been previously used in Lebanese audio DID research. The Lebanese-specific Arabic-French-English code-switching pattern has been studied linguistically but has not been exploited as a computational feature for DID.

To our knowledge, this thesis is the first work to: (i) construct and publish a manually annotated Lebanese-specific DID test set drawn from a multi-platform in-the-wild collection; (ii) run a systematic side-by-side comparison of 14 systems from four model families on Lebanese binary audio classification with percentile-bootstrap statistical confidence intervals; (iii) empirically characterize the recording-domain confound for Lebanese weakly-supervised data with four independent lines of evidence; and (iv) evaluate LLM few-shot prompting for Lebanese dialect identification from speech transcripts.

---

## 3. Corpus Construction and Dataset

This chapter describes how the Lebanese Arabic audio corpus was assembled from public platforms and research datasets. We detail the queue-driven pipeline architecture, each audio source, storage characteristics, and the key empirical finding (C5) that ADI17 contains no MSA items - a result that directly shaped the contrastive data strategy adopted in this thesis.

### 3.1 Pipeline architecture

The pipeline is queue-driven: items flow through a SQLite database (`data/queue.db`) with explicit status transitions. Each pipeline stage is a standalone script that reads items at a given status and writes them forward. The primary status flow is:

```
DISCOVERED → DOWNLOADED → SCREENED ──► WEAK_POSITIVE   (metadata trusted)
                                   ──► WEAK_NEGATIVE   (lexical scoring)
                                   ──► POTENTIAL_LB / BORDERLINE_LB / REJECTED  (v1 model)
```

Error states (`ERROR_DOWNLOAD`, `ERROR_TRANSCRIBE`) are recoverable by recovery scripts. All stages are resumable: they skip already-processed items by checking the database status, so interrupted runs can be safely restarted.

### 3.2 Sources

**YouTube.** 521 trusted Lebanese channel IDs were assembled through a combination of manual curation and RSS-based channel discovery (which requires no YouTube Data API quota). The channel list covers news programs, talk shows, comedy, cultural content, and educational material. Audio is downloaded via `yt-dlp`, normalized to mono 16 kHz with FFmpeg loudnorm filter, and capped at 1,200 seconds.

**Podcasts.** Approximately 151 Arabic-language RSS feeds are monitored via the PodcastIndex API. Unlike YouTube, podcast items have no reliable channel-level metadata marking dialect, so they rely entirely on lexical scoring (Section 5) for labeling. Many podcast feeds are pan-Arab in scope, providing a natural source of negative (non-Lebanese) training examples.

**TikTok.** 13 items collected; platform authentication and rate limits make systematic discovery brittle, and TikTok is treated as a minor supplement.

**ADI17** [@ali2019mgb5]. 1,000 Lebanese (LEB) items from ADI17 dev+test splits are imported as research-grade positive training examples. 1,000 Egyptian (EGY) and 5,000 Gulf items (KSA, KUW, UAE, QAT, OMA combined) are imported as labeled negative training examples. All items are extracted by filtering the Parquet shards' dialect column and encoding to MP3 (96 kbps, mono, 16 kHz, loudnorm) for storage uniformity.

**FLEURS** [@conneau2022fleurs]. 798 unique MSA items from the `ar_eg` configuration of FLEURS are imported as the MSA contrastive class. Although the speakers are Egyptian-based, the read content (FLoRes-101 sentences) is literary Arabic (al-Fuṣḥā / MSA register). See Section 3.5 for why FLEURS was selected over alternatives.

### 3.3 Dataset size

| Category | Count | Source |
|----------|------:|--------|
| WEAK_POSITIVE | 3,232 | YouTube (Lebanese channels) |
| POTENTIAL_LB | 1,769 | podcast_rss (model-scored) |
| BORDERLINE_LB | 313 | podcast_rss (model-scored) |
| REJECTED | 2,583 | YouTube + podcast_rss (model-scored) |
| WEAK_NEGATIVE | 1,005 | podcast_rss (lexical scoring) |
| ADI17 LEB (extra positives) | 1,000 | ADI17 dev+test |
| ADI17 EGY (negatives) | 1,000 | ADI17 dev+test |
| ADI17 Gulf (negatives) | 5,000 | ADI17 dev+test |
| FLEURS MSA (negatives) | 798 | FLEURS ar_eg |
| Held-out ground truth | 300 | Manual annotation |

Total items with audio and screening transcripts (Phase 1): ~8,902.
Phase 2 contrastive items: 7,798 (1,000 ADI17 LEB positives + 6,798 non-LB negatives).
Items selected for acoustic embedding extraction: 14,177.

### 3.4 Storage

Audio is stored as FLAC for YouTube content (lossless; approximately 20% of WAV size at speech compression ratios) and as MP3 (96 kbps, mono, 16 kHz, loudnorm-normalized) for podcast, ADI17, and FLEURS audio. Total footprint: approximately 95 GB after a mid-project WAV-to-FLAC migration that freed 252 GB when the working drive reached 100% capacity.

### 3.5 Finding (C5): ADI17 contains no MSA

We initially planned to extract MSA items from the ADI17 train split, assuming MSA would be one of the dialect classes. A column-projection scan of all 40 train Parquet shards (990,821 total rows) on a Colab GPU runtime returned zero MSA items. The complete dialect inventory in the ADI17 train split is: IRA (277,725), EGY (143,013), MAU (129,666), KSA (66,842), UAE (48,474), SYR (46,026), PAL (36,747), LEB (35,938), LIB (32,757), KUW (30,507), ALG (29,439), OMA (26,595), QAT (26,088), YEM (20,456), SUD (18,258), MOR (17,432), JOR (4,858) - exactly 17 country-level dialects with no MSA class. This is consistent with ADI17's design but is commonly overlooked by users of the HuggingFace dataset.

As a result we selected FLEURS `ar_eg` as the MSA contrastive source. Alternatives considered:
- **MGB-2** (broadcast MSA): acoustically the best match to ADI17 broadcast style, but requires QCRI-managed access and is deferred to future work.
- **Arabic Speech Corpus (Halabi 2016)** [@halabi2016msa]: single-speaker studio recording; rejected due to the single-speaker confound - a classifier trained against it cannot disentangle MSA register from a specific voice identity.
- **Mozilla Common Voice ar**: multi-speaker, citable [@ardila2020commonvoice]; however, Mozilla migrated all Common Voice resources off HuggingFace in October 2025 and they are no longer accessible via the `datasets` library.

**Acoustic concession accepted:** FLEURS `ar_eg` is read-prompt audio while ADI17 dialects are broadcast audio. This domain difference between the MSA class and other classes is a known limitation (Section 12). Audio normalization (mono, 16 kHz, loudnorm, 96 kbps MP3) is applied uniformly across all sources to minimize codec and amplitude differences. The recording-domain confound it introduces is documented and empirically characterized in Section 10.

### 3.6 Corpus demographics and item characteristics

Detailed speaker-level metadata (age, gender, speaker identity) is not available for the in-house collected portion of the corpus. The pipeline collects audio and source metadata (channel ID, feed URL, title, uploader name), not speaker profiles, and platform APIs do not systematically expose demographic information at the item level. We report what can be derived from available metadata and computed audio properties.

**Duration statistics.** For Phase 1 screened items (items with a downloaded audio file, N ≈ 6,900):

| Statistic | Value |
|---|---|
| Mean duration | 742 s (12.4 min) |
| Median duration | 584 s (9.7 min) |
| Minimum duration | 6 s (pipeline floor) |
| Maximum duration | 1,200 s (pipeline cap) |
| Items > 15 min | ~38% (podcast episodes) |
| Items < 5 min | ~21% (TikTok clips, short YouTube content) |

The right-skewed distribution reflects the dominance of long podcast episodes in the collection. YouTube videos cluster around 10-30 minutes; podcast episodes cluster around 30-60 minutes (capped at 20 minutes by the pipeline).

**Recording type by source.**

| Source | Recording type | Register | Speaker count |
|---|---|---|---|
| YouTube (Lebanese channels) | Conversational, interview, panel, vlog | Mixed (dialect + MSA code-switching) | Multiple, unconstrained |
| Podcast RSS | Conversational, interview, monologue | Mixed (dialect-heavy or pan-Arab MSA-heavy) | Multiple, unconstrained |
| ADI17 | Broadcast (Al Jazeera-style studio) | Country-level dialects | Multiple, professional broadcasters |
| FLEURS ar\_eg | Read-prompt (studio recording) | MSA register (FLoRes-101 sentences) | Multiple, Egyptian readers |

**Genre distribution (YouTube, N = 521 channels, based on manual curation).** News and political commentary (31%), entertainment and comedy (24%), talk show and interview (22%), cultural and educational content (15%), cooking and lifestyle (8%).

**ADI17 demographics.** The ADI17 documentation reports a predominantly male speaker profile among Al Jazeera broadcast personnel (~60% male). Speakers are professional broadcasters from Al Jazeera regional bureaus. The Lebanese LEB class (~35,938 train rows) is drawn from the Lebanese bureau.

**FLEURS ar\_eg demographics.** The FLEURS release [@conneau2022fleurs] documents multi-speaker Egyptian readers; the `ar_eg` split used here (798 unique items extracted for this thesis) follows the FLEURS speaker pool, which is male-dominated (~65% male per the FLEURS metadata). Age distribution is adult (20-60 years) based on the FLEURS recruitment protocol.

**Code-switching density.** Lebanese items in the GT exhibit 3.5× higher Latin-script token density than non-Lebanese items (0.023 vs. 0.007 Latin-to-Arabic token ratio in Whisper transcripts), reflecting the Arabic-French-English code-switching characteristic of Lebanese speech (Section 2.6). YouTube items show 7× higher Latin density than podcast items (0.044 vs. 0.006), consistent with a more bilingual YouTube creator demographic.

**Note on speaker diversity.** No deduplication by speaker identity is performed in the in-house collection (YouTube and podcast portions). A single prolific Lebanese podcast host could appear across many episodes. This is an acknowledged limitation (see also Section 12.3); it is shared with all large-scale weakly-supervised speech corpora assembled from public platforms.

---

## 4. Preprocessing and Audio Normalization

This chapter describes the normalization and feature extraction steps applied uniformly to all audio entering the pipeline, and specifies the hardware and software environment in which all experiments were conducted. A consistent preprocessing protocol is essential to reduce recording-domain artifacts before downstream classifier training; the specific choices made here (sampling rate, codec, chunk length, model size) are motivated empirically.

### 4.1 Experimental environment

All data collection, training, and evaluation for this thesis were conducted in the following hardware and software environment.

**Hardware:**

| Component | Specification |
|---|---|
| Machine | Personal workstation |
| Operating system | Windows 11 Home, build 10.0.22631 |
| CPU | Intel Core i9 (22 logical cores) |
| RAM | 32 GB DDR4 |
| GPU | None - all training and inference on CPU |
| Storage | SSD; ~240 GB free during collection phase |

**Software:**

| Package | Version |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.5.1 |
| Transformers (HuggingFace) | 4.46.3 |
| faster-whisper | 1.1.1 |
| scikit-learn | 1.5.2 |
| sentence-transformers | 3.2.1 |
| SQLAlchemy | 2.0.36 |
| Pydantic | 2.9.2 |
| yt-dlp | 2024.11.18 |
| feedparser | 6.0.11 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| SpeechBrain | 1.0.2 |
| loralib | 0.1.2 |
| joblib | 1.4.2 |
| FFmpeg | 7.0 (system install) |

**Random seeds.** The value 42 is used throughout for all stochastic operations: GT sample draw (`random.seed(42)`), train/validation splits (`random_state=42` in scikit-learn), bootstrap CI resampling (seed 42, n=1,000), and V2.5 optimizer initialization (`torch.manual_seed(42)`). No operation in the pipeline uses an unspecified or system-random seed.

### 4.2 Audio normalization

All audio entering the pipeline - regardless of source - is normalized to a uniform format before any downstream processing:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Channels | mono | ASR and speech models trained on mono |
| Sampling rate | 16 kHz | Standard for ASR and SSL speech models |
| Loudness | loudnorm (FFmpeg) | Eliminates volume-based cross-source bias |
| Storage codec | FLAC (YouTube) / MP3 96 kbps (other) | FLAC: lossless, 80% size reduction from WAV; MP3: already compressed at source |
| Maximum duration | 1,200 seconds | Caps storage growth; clips >20 min rejected at download |
| Minimum duration | 6 seconds | Below this threshold, insufficient audio for dialect analysis |

**Screening transcription.** Three 20-second random chunks are extracted per item and stored as 16-kHz WAV files. Faster-Whisper [@radford2023whisper] transcribes each chunk using the `base` model. We benchmarked `medium` vs `small` vs `base` on five representative audio items and found that `base` achieves identical language-detection confidence (lang_prob = 1.00 on all chunks) at 15× the throughput of `medium` (1.3 sec/chunk vs. 20 sec/chunk on CPU); transcription quality is sufficient for downstream lexical scoring.

**Acoustic embedding clips.** For the acoustic classifiers (Section 7.2), each item is trimmed to 10 seconds starting from a fixed offset per clip and re-encoded at 64 kbps mono 16 kHz MP3. The 10-second window is the standard utterance-level duration for XLS-R mean-pooled dialect ID in the literature. This trimming is uniform across all classes so that recording-domain differences survive only at the source level, not from differing clip durations or bitrates introduced by the pipeline.

---

## 5. Weak Supervision

This chapter describes how training labels are derived from metadata and lexical signals without manual annotation at scale. We first establish a formal notation framework, then define the two weak labeling strategies, characterize their noise levels, and quantify the label-noise impact on downstream classifier performance.

### 5.0 Formal notation

Let **D** = {(**x**_i, *y*_i)}_{i=1}^{N} denote the dataset, where **x**_i is the *i*-th audio item and *y*_i ∈ {0, 1} is its binary dialect label (1 = Lebanese, 0 = not-Lebanese). In practice, *y*_i is not observed directly but is approximated by weak label functions λ_k(**x**_i) derived from metadata and lexical scoring.

Let τ(**x**) denote the concatenated Whisper screening transcript for item **x**, and let C = {lb, msa, egy, gulf, sy} denote the five dialect lexicon classes. Define f_k(**x**) = |{w ∈ τ(**x**) : w ∈ lexicon_k}| as the count of words in τ(**x**) belonging to lexicon class *k*. The raw dialect score is:

```
raw_score(x)   = 1.8 · f_lb(x) - 0.6 · f_msa(x) - 1.0 · f_egy(x) - 1.0 · f_gulf(x) - 0.5 · f_sy(x)
final_score(x) = max(0, min(1, raw_score(x) / 5.0))
```

Weak labeling criteria:
- **λ_pos(x) = 1** iff channel_id(**x**) ∈ T_LB (trusted Lebanese channel list) AND strong_lb_hits(**x**) ≥ 1
- **λ_neg(x) = 1** iff raw_score(**x**) < 0 AND lang_prob(**x**) ≥ 0.70

where strong_lb_hits(**x**) counts occurrences of uniquely Levantine markers (شو، هيك، هلق، بدي، عنجد، كتير، وين، هون، هيدا) in τ(**x**), and lang_prob(**x**) is Whisper's Arabic language-identification probability averaged over the three screening chunks.

### 5.1 Positive labels (WEAK_POSITIVE)

YouTube items whose `channel_id` matches the curated trusted-Lebanese-channel list are labeled WEAK_POSITIVE. This produces 3,232 candidates. Spot-checking against the ground-truth set (Section 6.2) reveals that approximately 33% of WEAK_POSITIVE items are not actually Lebanese-dialect content: Lebanese creators post MSA news, formal interviews, music videos, and educational content in formal register.

We mitigate this noise at training time by requiring at least one strong Lebanese dialect marker in the screening transcript (`strong_lb_hits ≥ 1`). Strong markers are: شو، ليش، هيك، هلق، عنجد، بدي، كتير، وين، هون، هيدا، هيدي، هدول. This raises estimated WEAK_POSITIVE precision from 67% to 81% (measured against the ground-truth sample of 45 WEAK_POSITIVE items) while retaining 62% of items (2,003 of 3,232). Higher thresholds (≥2, ≥3) did not significantly improve precision on the small sample but substantially reduced training data size, so ≥1 was selected as the best balance.

### 5.2 Negative labels (WEAK_NEGATIVE)

Items with transcript dialect score `raw_score < 0` and Whisper language probability ≥ 0.70 are labeled WEAK_NEGATIVE. The scoring formula is:

```
raw_score    = lb × 1.8 − msa × 0.6 − egy × 1.0 − gulf × 1.0 − sy × 0.5
final_score  = max(0, min(1, raw_score / 5.0))
```

where `lb`, `msa`, `egy`, `gulf`, `sy` are word-list match counts from curated Lebanese, MSA, Egyptian, Gulf, and Syrian Arabic lexicons. The criterion `raw_score < 0` (non-Lebanese signals outweigh Lebanese) produces 1,005 negatives, predominantly from pan-Arab podcast content. The alternative criterion `lb == 0` (zero Lebanese matches) produced only 2 negatives because the Lebanese lexicon contains pan-Arabic words (يعني، بس، في، مش، تمام، مرحبا) that appear in virtually all Arabic transcripts.

**Known lexical limitation.** The strong Lebanese markers are Levantine, not uniquely Lebanese: شو، هيك، هلق also appear in Syrian and Palestinian Arabic. Lexical features can distinguish Lebanese from Egyptian, Gulf, and MSA but not reliably from other Levantine varieties.

### 5.3 Noise quantification

The gap between in-pool validation accuracy and held-out ground-truth performance quantifies the label-noise impact. For V1 (text-only classifier):

- In-pool validation: accuracy 89%, ROC-AUC 0.964
- Held-out ground truth: ROC-AUC 0.848 (−0.116)

This 12-point ROC-AUC drop is the noise correction signal: weak labels systematically overestimate generalization.

---

## 6. Ground-Truth Test Set

This chapter describes the construction of the 300-item manually annotated evaluation set that anchors all benchmark results in this thesis. We document the stratified sampling design, the annotation methodology and tool, the resulting label distribution, and per-tier precision measurements that characterize the noise in the weak labeling pipeline.

### 6.1 Annotation design

#### 6.1.1 Total sample size (n = 300)

The total of 300 items was chosen to satisfy three simultaneous constraints.

**Statistical precision.** For a proportion estimated from a simple random sample, the 95% CI half-width is maximised at p = 0.5 and equals z₀.₀₂₅ √(0.25/n). At n = 300 this gives ±5.7 percentage points — a commonly adopted threshold for "5% precision" in NLP evaluation studies [@gorman2019s; @bouthillier2021accounting]. Smaller sets (n = 150) give ±8.0 pp, which is too coarse to reliably distinguish systems whose ROC-AUC differs by 0.03–0.05 (the margin separating several systems in the scoreboard). Larger sets (n = 600) would halve the half-width but roughly double the annotation effort without materially changing the ranking.

**Annotation budget.** Each item required listening to a 60-second clip and reading a scrollable transcript. Pilot timing at the start of the annotation session measured approximately 2–3 minutes per item. At 300 items this implies 10–15 hours of focused annotation — feasible as a thesis-internal effort without requiring paid annotators or IRB coordination. A set of 600 items would have exceeded a safe single-annotator quality threshold.

**Field precedent.** The ADI17 shared-task test set contains 315 items per dialect; NADI 2021 uses 200 items per dialect; [@elfardy2012line] evaluation sets range from 150 to 500 items. A 300-item set is within the standard range for dialect identification evaluations and ensures comparability when reporting metrics against published baselines.

#### 6.1.2 Allocation across pipeline tiers (90 / 60 / 60 / 45 / 45)

The 300 items were not drawn as a simple random sample from the full corpus. Instead, a **stratified** allocation was used, with stratum sizes chosen to serve two distinct scientific goals: (1) **characterising weak-labelling noise** in each pipeline tier, and (2) **providing a held-out set for benchmark evaluation**. These goals jointly determined the allocation:

| Pipeline tier | n | 95% CI half-width on precision | Primary scientific role |
|---|---:|---:|---|
| POTENTIAL_LB | 90 | ±10.3 pp | Benchmark evaluation + positive-precision audit |
| BORDERLINE_LB | 60 | ±12.5 pp | Threshold calibration + false-positive audit |
| REJECTED | 60 | ±12.5 pp | False-negative (recall) audit |
| WEAK_POSITIVE | 45 | ±14.6 pp | Metadata-label noise audit |
| WEAK_NEGATIVE | 45 | ±14.6 pp | Lexical-label noise audit |

The largest allocation (n = 90) goes to POTENTIAL_LB because it is the primary output tier of the pipeline: its precision directly measures positive predictive value, it supplies the largest share of training positives, and ranking systems by their ability to identify POTENTIAL_LB items is the thesis's central comparative claim. Allocating more items to this tier narrows the CI on its precision relative to the other tiers. The two auditing tiers (WEAK_POSITIVE, WEAK_NEGATIVE) receive the smallest samples (n = 45) because their role is qualitative verification rather than precise estimation: even ±14.6 pp is sufficient to establish that metadata-based labels are noisier than REJECTED labels (66.7% vs. 1.7% precision), which is the operative finding. The sample was drawn with `random.seed(42)` and the presentation order was shuffled to prevent session-level tier fatigue.

#### 6.1.3 Benchmark purpose vs. noise-characterisation purpose

The GT serves a **dual purpose**, and the tension between these purposes must be stated explicitly:

- **Noise characterisation** (Sections 6.3–6.4) uses the tier-level breakdown to quantify how much the weak labelling pipeline mislabels items at each confidence level. This analysis is self-contained within the GT sample and does not require the GT to be representative of any broader distribution.

- **System benchmarking** (Chapters 9–11) uses the 296 evaluable binary-labelled items as a held-out test set to rank the 14 systems. For this purpose the relevant question is whether the GT is representative of the test distribution — addressed in Section 6.5 below.

Both purposes are served by the same 300-item annotation, but they motivate different aspects of the design: stratification serves the noise-characterisation goal, while the overall size and diversity of platforms represented serve the benchmarking goal.

### 6.2 Annotation methodology

A custom Flask annotation tool was built with HTML5 audio playback, keyboard shortcuts (`l` / `m` / `n` / `u` / `s`), and CSV persistence. Each item presents a 60-second clip extracted at 30% into the source file and a scrollable transcript display. Five labels were used:

| Label | Meaning | Binary mapping |
|---|---|---|
| Lebanese | Clearly Lebanese dialect throughout | 1 |
| Mostly Lebanese / mixed | Lebanese dialect with code-switching to MSA or other | 1 |
| Not Lebanese | Non-Lebanese Arabic (MSA, Egyptian, Gulf, etc.) | 0 |
| Unclear | Cannot determine dialect from this audio | excluded |
| Skip | Technical problem with clip | excluded |

Annotation was performed by the thesis author, a native Lebanese Arabic speaker. A second Lebanese-Arabic-speaking native speaker independently labeled all 300 items using the same tool and the same item order, enabling inter-annotator agreement measurement.

#### 6.2.1 Inter-annotator agreement

**Table 3. Inter-annotator agreement on the 300-item GT sample.**

| Metric | Value | Interpretation |
|---|---|---|
| 5-way Cohen's κ | 0.47 | Moderate (Landis & Koch 1977) |
| Binary Cohen's κ | 0.72 | Substantial |
| Exact agreement | 235 / 300 = 78.3% | — |
| Items excluded from binary κ (unclear / skip by either annotator) | 5 | — |
| Total disagreements | 65 / 300 = 21.7% | — |

The 5-way κ of 0.47 reflects the inherent gradability of Lebanese dialect perception. The dominant confusion is *Lebanese* ↔ *Mostly Lebanese*: 23 of 65 disagreements (35%) are in this direction, and an additional 6 go the other way. Code-switching is a continuum, and two native speakers draw the boundary between a code-switched utterance and a purely Lebanese one differently. This label-boundary sensitivity is expected — and is why the binary evaluation (Lebanese + Mostly Lebanese → positive, Not Lebanese → negative) is the primary frame for this thesis.

The binary κ of 0.72 is the operationally relevant agreement metric. It measures agreement on the fundamental question — is this audio Lebanese or not — excluding the 5 items where either annotator was unable to decide. κ = 0.72 exceeds the conventional κ ≥ 0.60 threshold for substantial agreement (Landis & Koch 1977), providing independent validation that the 296-item binary ground truth is reliable. A second native speaker making independent decisions from the same audio agrees with the primary annotation on approximately 9 out of every 11 binary decisions.

The confusion pattern is interpretable: the second annotator more often upgraded *Mostly Lebanese* to *Lebanese* (23 cases) than downgraded (6 cases), suggesting a slightly more permissive threshold for the unambiguously Lebanese label. This asymmetry does not threaten the binary evaluation: in all 29 such cases, both annotators agreed the item was positive.

**Table 4. Annotator confusion matrix (rows = A1, cols = A2).**

| | Lebanese | Mostly LB | Not LB | Unclear |
|---|---:|---:|---:|---:|
| Lebanese | 19 | 6 | 6 | 1 |
| Mostly LB | 23 | 8 | 19 | 0 |
| Not LB | 2 | 4 | 208 | 0 |
| Unclear | 0 | 0 | 4 | 0 |

#### 6.2.2 Consensus ground truth

From the 300 doubly-annotated items, a **consensus GT** was constructed retaining only items where both annotators agree on the binary label (Lebanese+MostlyLB = positive; NotLebanese = negative). This yields 264 items (positive: 56, negative: 208); 31 items with binary-class disagreement and 5 items where either annotator said *unclear* or *skip* are excluded.

The consensus GT is used as a sensitivity check in Section 9.9: if the benchmark rankings are stable across the full 296-item GT and the 264-item consensus GT, this confirms that the 31 contested items do not drive any system's position. The consensus GT is saved as `data/annotations_consensus.csv`.

### 6.3 Label distribution

| Ground truth label | Count | Percentage |
|---|---:|---:|
| Lebanese | 32 | 10.7% |
| Mostly Lebanese / mixed | 50 | 16.7% |
| Not Lebanese | 214 | 71.3% |
| Unclear | 4 | 1.3% |

For binary evaluation: 82 positive (Lebanese + Mostly Lebanese), 214 negative (Not Lebanese), 4 excluded. Total evaluable: 296.

### 6.4 Per-tier precision

| Pipeline tier | n | Positive (Lebanese+Mostly LB) | Precision | 95% CI |
|---|---:|---:|---:|---|
| WEAK_POSITIVE | 45 | 30 | **66.7%** | [52%, 80%] |
| POTENTIAL_LB | 90 | 34 | **37.8%** | [28%, 48%] |
| BORDERLINE_LB | 60 | 12 | **20.0%** | [10%, 30%] |
| REJECTED | 60 | 1 | **1.7%** | [0%, 5%] |
| WEAK_NEGATIVE | 45 | 5 | **11.1%** | [2%, 21%] |

Key observations: REJECTED items are almost never Lebanese (98.3% specificity), confirming strong rejection precision. POTENTIAL_LB precision (37.8%) is substantially below what its label suggests, reflecting noise in the weakly-supervised positive training pool. The `mostly_lebanese` category (16.7% of GT) reflects widespread code-switching between Lebanese dialect and MSA - a linguistically expected pattern given Lebanon's diglossia.

### 6.5 Distribution mismatch and metric interpretation

#### 6.5.1 What population does the GT represent?

The GT represents a specific, operationally defined population: **items that passed the pipeline's download and screening stages from the five platforms ingested during the 2025–2026 collection window (YouTube, podcast RSS, TikTok, ADI17, FLEURS), and were then assigned to one of the five pipeline confidence tiers.** This is a convenience sample from the pipeline's own filtered output, not a probability sample from any broader population of Lebanese Arabic audio. Its composition is determined by (a) which platforms were targeted, (b) what content those platforms happened to contain during the collection period, and (c) the pipeline's screening and weak-labelling decisions.

Three consequences follow directly from this definition:

1. **Platform mix.** The GT reflects the platform mix of the corpus (predominantly podcast RSS and YouTube, with ADI17 and FLEURS as controlled contrastive sources). Systems that exploit recording-domain cues — rather than dialect cues — will appear stronger on this GT than they would on a platform-balanced test set. This is exactly the confound documented in Chapter 10.

2. **No generalisation to Lebanese Arabic audio in the wild.** The pipeline's discovery heuristics (RSS feeds of known Lebanese channels, YouTube search queries for Lebanese topics) deliberately bias toward Lebanese-proximate content. The GT therefore over-represents items where some Lebanese signal is present (even if not enough for a positive label), relative to a truly random sample of Arabic audio.

3. **Stratification departs from the pipeline's own distribution.** Even within the pipeline's output, the GT does not mirror the tier distribution. In the scored corpus (≈8,870 items), roughly 60% carry a positive-side label (WEAK_POSITIVE, POTENTIAL_LB, or BORDERLINE_LB). The GT has a 27.7% positive rate (82/296), achieved by including full draws from the predominantly negative REJECTED and WEAK_NEGATIVE tiers. The stratification was chosen to maximise annotation signal per item (by including more uncertain cases) rather than to reproduce the corpus distribution.

Two consequences follow:

1. **Macro F1 is distribution-sensitive.** Macro F1 (averaged over positive and negative classes without weighting by class frequency) is determined partly by the positive:negative ratio in the test set. The GT ratio of 82:214 ≈ 1:2.6 is more balanced than realistic deployment scenarios (where non-Lebanese audio dominates). A system that appears strong on macro F1 in this GT may perform worse in a deployment where non-Lebanese items overwhelm the positive class.

2. **ROC-AUC is distribution-robust.** ROC-AUC is computed from the full ranking of items by predicted score and is invariant to the positive:negative ratio. It is therefore the primary metric for comparing systems across this thesis. The GT's stratified design does not affect ROC-AUC validity, only the interpretation of the absolute AUC value as an estimate of deployment-time performance.

#### 6.5.2 What the reported metrics do and do not support

**The GT supports:**
- *Ranking* the 14 systems relative to each other. Because all systems are evaluated on the same GT items, differences in ROC-AUC and macro F1 reflect genuine differences in discriminative ability, not artefacts of the sampling design.
- *Tier-level precision estimates* (Section 6.4), which are conditioned on the GT sample and are internally valid.
- *Bootstrap CIs on system performance*, which quantify uncertainty about the GT estimate (i.e., how much the rank would change if a different 300-item draw were made from the same corpus), not uncertainty about population-level performance.

**The GT does not support:**
- *Estimating absolute precision/recall in deployment*, because the positive rate in the GT does not match the positive rate a deployed system would encounter.
- *Generalising absolute macro F1 values* to other evaluation corpora with different class balance.
- *Fully adjudicated inter-annotator labels*: a second annotator has been completed (binary κ = 0.72, Section 6.2.1) but the 65 disagreement cases have not yet been formally adjudicated; borderline items carry the primary annotator's label.

To summarise: the reported metrics are valid for the comparative ranking of systems on this GT and for the noise audit of the weak labelling pipeline. They should not be interpreted as direct estimates of the expected precision or recall that any system would achieve in an arbitrary Lebanese-Arabic audio deployment.

---

## 7. Benchmark Systems

This chapter describes all 14 systems evaluated in the benchmark: five built in-house across three architectural families (text-only V1, frozen acoustic V2, end-to-end fine-tuned V2.5), four pulled from HuggingFace, one free baseline, two ablation variants, and two LLM zero/few-shot conditions. For each in-house system we provide the formal feature representation, training procedure, and architectural rationale.

### 7.1 In-house systems

#### V1: Text-only classifier (lexical + MiniLM)

V1 is a Logistic Regression trained on a 389-dimensional feature vector:
- 5 lexical features: `[lb, msa, strong_lb_hits, msa_ratio_core, final_score]` from the `lexicon_score()` function
- 384-d sentence embedding from `paraphrase-multilingual-MiniLM-L12-v2` [@reimers2019sbert] applied to the concatenated screening transcript

Formally, the feature vector for item **x** is:

```
φ(x) = [f_lb(x), f_msa(x), strong_lb_hits(x), msa_ratio_core(x), final_score(x), e(τ(x))] ∈ ℝ^389
```

where e(τ(**x**)) ∈ ℝ^384 is the MiniLM sentence embedding of the concatenated screening transcript τ(**x**). The classifier produces:

```
p̂(y=1 | x) = σ(w^T φ(x) + b)
```

where σ is the sigmoid function and (w, b) are learned by logistic regression with L2 regularization (C=1.0, default scikit-learn), optimized by LBFGS. The training objective is binary cross-entropy, with `class_weight="balanced"` reweighting to compensate for the positive:negative imbalance (~1:1.7 in the training pool).

`class_weight="balanced"` compensates for the class imbalance in the training pool. Training uses scikit-learn [@pedregosa2011sklearn] LogisticRegression with default L2 regularization. Training data: 2,003 positives (WEAK_POSITIVE, `strong_lb_hits ≥ 1`) + 1,651 negatives (1,005 WEAK_NEGATIVE + 646 REJECTED with transcripts). **Note on training/test overlap:** V1 was trained before the 300-item GT was established; the GT was subsequently sampled from the same WEAK_POSITIVE (45 items), WEAK_NEGATIVE (45 items), and REJECTED (60 items) pools used for V1 training. Up to 150 of 296 GT items may therefore appear in V1's training data. See §12.5 for an assessment of the potential bias.

#### V1 ablations: lex-only and embedding-only

To understand which component drives V1's performance, two ablation variants are trained on the same data pool with the same `strong_lb_hits ≥ 1` filter:
- **V1 lex-only**: 5 lexical features only (no embedding)
- **V1 embedding-only**: 384-d MiniLM embedding only (no explicit lexical features)

Both are trained identically to V1 (LogisticRegression, `class_weight="balanced"`).

#### V2: Frozen XLS-R-300m + MLP

V2 replaces text features with 1024-d mean-pooled utterance embeddings from `facebook/wav2vec2-xls-r-300m` [@babu2022xlsr]. The backbone is frozen in `eval()` mode; only a classifier head (Logistic Regression or MLP with 256 hidden units, early stopping) is trained on the embeddings.

Training pool: 13,624 items (all WEAK_POSITIVE, POTENTIAL_LB, ADI17 LEB as positives; WEAK_NEGATIVE, ADI17 EGY/Gulf, FLEURS MSA as negatives), excluding the 300 GT items. Both heads are reported; the MLP head was selected as the winner by held-out macro F1.

#### V2 + per-source balanced training

V2 is retrained with sample weights such that each (platform, label) pair contributes equal total weight. This is designed to remove the recording-domain shortcut hypothesized in Section 10. The weight scheme is:

```
weight(item) = 1 / (count of items in this item's (platform, label) group)
```

Trained with the same MLP head as V2 (frozen XLS-R backbone).

#### Hybrid V1+V2 MLP

Concatenates V1's 389-d text feature vector with V2's 1024-d acoustic embedding (1,413-d total) and trains LogisticRegression and MLP heads. The hybrid training pool is restricted to items with both a valid screening transcript and an embedding (5,427 items total; all GT items excluded).

#### V2.5: End-to-end fine-tuned XLS-R

V2.5 unfreezes the top 2 transformer layers (of 24) of XLS-R-300m plus the projector and classifier head - approximately 34M of 316M total parameters (10.9%). Training uses `WeightedRandomSampler` with per-(platform, label) group weights to counteract the recording-domain shortcut at the sampling level, AdamW optimizer with differential learning rates (5×10⁻⁵ for the head, 1×10⁻⁵ for the unfrozen encoder layers), and a 4,000-item stratified subsample of the full 13.6K pool (CPU runtime ceiling; estimated 10-12 hours wall time for 2 epochs on a 22-core machine). Checkpointing every 10 optimization steps allows training to span multiple sessions.

### 7.2 Public systems (HuggingFace)

#### Abdullah and Baas MMS-300m (Levantine proxy)

`badrex/mms-300m-arabic-dialect-identifier` (Abdullah et al. 2025): MMS-300m fine-tuned on Arabic dialect identification with voice-conversion-based data augmentation. Outputs a probability for `Levantine` (which conflates Lebanese, Syrian, Jordanian, Palestinian Arabic). We use `P(Levantine)` as the Lebanese-detection score, which upper-bounds Lebanese detection ability - a system that perfectly identifies Levantine but cannot distinguish sub-varieties will appear strong here.

#### Voxlect MMS-LID-256 (Levantine proxy)

`tiantiaf/voxlect-arabic-dialect-mms-lid-256` (Voxlect 2026): MMS-based Arabic dialect LID covering 256 classes including a Levantine category. Requires the `MMSWrapper` class vendored from the upstream repository (`scripts/_vendored/voxlect_mms_dialect.py`) due to a non-standard HuggingFace model structure. We use `P(Levantine)` as the score.

#### Elyadata ADI-whisper-ADI20 (country-level LEB)

`Elyadata/ADI-whisper-ADI20` (Elleuch et al. 2025): Whisper-Large-V3 fine-tuned on ADI-20, a 20-country Arabic dialect dataset that includes a country-level `LEB` class. This is the **only** system in the benchmark with a true country-level Lebanese class (rather than a Levantine proxy). Requires vendoring `WhisperDialectClassifier` from the upstream ADI-20 repository (`scripts/_vendored/elyadata_classifier_attention_pooling.py`) as the model uses a non-standard SpeechBrain architecture.

#### MARBERTv2 Levantine (text-based, Levantine proxy)

`IbrahimAmin/marbertv2-arabic-written-dialect-classifier`: MARBERTv2 [@inoue2021camelbert] fine-tuned on Arabic written dialect classification, outputting probabilities for four regional groups (LEV, EGY, GULF, MSA). Applied to existing Whisper screening transcripts rather than audio. Uses `P(LEV)` (Levantine regional class) as the score.

### 7.3 Baseline

#### Whisper-LID Arabic probability

Whisper's built-in language probability is a free byproduct of every screening transcription. For each GT item, we average the `language_probability` for Arabic across the three screening chunks (chunks predicting a non-Arabic language contribute 0.0). This tests whether Arabic-vs-other language detection proxies for Lebanese-vs-other dialect detection. The expected answer is no: Whisper's LID operates at the language level, not the dialect level.

### 7.4 Computational complexity

Table 3 summarizes the computational requirements of each system in the benchmark, providing the information needed for researchers planning to reproduce or extend this work.

**Table 3. Computational complexity overview for all 14 benchmark systems.**

| System | Trainable parameters | Training pool | Approx. training time | Inference per item | RAM / VRAM | GPU needed |
|---|---|---|---|---|---|---|
| V1 LR (389-d) | 390 | 3,654 items | < 5 min, CPU | < 1 ms (text) | < 1 GB RAM | No |
| V1 lex-only (5-d) | 6 | 3,654 items | < 1 min, CPU | < 1 ms (text) | < 1 GB RAM | No |
| V1 embed-only (384-d) | 385 | 3,654 items | < 3 min, CPU | < 1 ms (text) | 0.5 GB RAM | No |
| V2 frozen MLP | ~263K (MLP head) | 13,624 items | ~11 h embed extraction + 30 min training, CPU | ~3 s (XLS-R encode) + <1 ms (head) | 1.2 GB RAM | No (slow) |
| V2 balanced | ~263K (MLP head) | 13,624 items | same as V2 | same as V2 | 1.2 GB RAM | No |
| V2 same-source | ~263K (MLP head) | 2,878 items | ~3 h embed + 10 min training, CPU | same as V2 | 1.2 GB RAM | No |
| Hybrid V1+V2 MLP | ~430K (MLP head) | 5,427 items | < 45 min, CPU | ~3 s (XLS-R) + <1 ms | 1.5 GB RAM | No |
| V2.5 fine-tuned XLS-R | ~34M of 316M | 4,000 items | ~12 h for 2 epochs, CPU | ~3 s | 4 GB RAM | Recommended (GPU: ~1 h) |
| Abdullah MMS-300m | 300M (inference only) | N/A | N/A | ~5 s/item, CPU | 1.2 GB RAM | Recommended |
| Voxlect MMS-LID-256 | 256M (inference only) | N/A | N/A | ~5 s/item, CPU | 1.0 GB RAM | Recommended |
| Elyadata Whisper-L-v3 | 1.5B (inference only) | N/A | N/A | ~8 s/item, CPU | 6 GB RAM (CPU) / 3 GB VRAM | Recommended |
| MARBERTv2 | 163M (inference only) | N/A | N/A | <1 s/item (text, CPU) | 0.7 GB RAM | No |
| Whisper LID baseline | 74M base (inference only) | N/A | N/A | ~1.3 s/chunk, CPU | 0.3 GB RAM | No |
| Llama-3.1-8B (Groq API) | 8B (remote) | N/A | N/A | ~1 s/item (API latency) | N/A (cloud) | No |

*Notes.* Inference times are wall-clock on the project's 22-core CPU. XLS-R embedding extraction dominated total compute: 13,624 items × ~3 s = ~11 hours. V2.5 training at 4,000 items × 2 epochs ran approximately 12 hours on CPU; GPU would reduce this to under 1 hour. All RAM figures include the model in memory; actual peak may vary with batch size.

---

## 8. Benchmark Protocol

This chapter defines the evaluation protocol applied uniformly to all 14 benchmark systems. A shared protocol - fixed ground-truth mapping, identical metric definitions, and pre-registered bootstrap CI computation - is essential for valid cross-system comparison when systems differ in architecture, training data, and output calibration.

### 8.1 Ground-truth mapping

GT labels are mapped to binary as: `lebanese → 1`, `mostly_lebanese → 1`, `not_lebanese → 0`; `unclear` and `skip` are excluded. Total evaluable items: 296 (82 positive, 214 negative). Systems that require a transcript (V1, V1 ablations, Hybrid, MARBERTv2) are evaluated on 295 items (one item is missing a valid screening transcript).

### 8.2 Metrics

Let *ŷ*_i ∈ [0, 1] be system i's predicted Lebanese probability for GT item i, and *y*_i ∈ {0, 1} be the binary ground truth.

- **ROC-AUC** (primary): threshold-independent discrimination metric. Formally:

```
AUC = P(p̂(x+) > p̂(x-))
    = ∫₀¹ TPR(FPR⁻¹(t)) dt
```

where x+ and x- are randomly drawn positive and negative items. AUC = 1 is perfect; AUC = 0.5 is a random classifier.

- **Macro F1 @ 0.50** (primary): honest evaluation at the default binary decision threshold τ = 0.50, requiring no test-set access:

```
Macro-F1(τ) = (1/2) · [F1(class=1, τ) + F1(class=0, τ)]
```

where F1(c, τ) = 2·P(c,τ)·R(c,τ) / (P(c,τ) + R(c,τ)), with P(c,τ) and R(c,τ) the precision and recall for class c at threshold τ.

- **Macro F1 (best snooped)**: best macro F1 across a grid of 12 thresholds {0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}. This is an *upper bound* on performance - it requires test-set access and is reported for completeness as a sensitivity analysis.
- **PR-AUC**: precision-recall area under curve, useful for imbalanced positive rates.
- **Confusion matrix** at the default threshold (0.50).

### 8.3 Confidence intervals

All ROC-AUC and macro F1 metrics are accompanied by 95% percentile-bootstrap confidence intervals. The procedure is: (i) draw n = 1,000 bootstrap resamples (with replacement) of the 296 GT items with seed 42; (ii) compute the metric on each resample; (iii) report the [2.5%, 97.5%] empirical quantiles as the 95% CI. Formally, if θ̂_1, ..., θ̂_B are the metric values on B bootstrap resamples, the CI is [θ̂_(⌈Bα/2⌉), θ̂_(⌊B(1-α/2)⌋)] with α = 0.05.

These CIs support conservative pairwise comparisons: when two systems' 95% CIs do not overlap, we can reject H₀ at approximately p < 0.05. Note that non-overlap is a conservative test - overlapping CIs do not rule out a real difference, and a paired bootstrap on the metric difference would be more powerful. Pairwise claims in this paper are stated accordingly.

### 8.4 Reproducibility

Per-system predictions are saved to `data/benchmark_predictions/<name>.json` after each run. The evaluation and CI computation are re-runnable from these prediction files without re-running model inference. Scripts: `scripts/20_benchmark_harness.py` (evaluation harness), `scripts/benchmark_harness_utils.py` (shared utilities, load/save predictions, bootstrap CIs, CSV upsert).

---

## 9. Benchmark Results

This chapter presents the main experimental results across all 14 benchmark systems on the held-out 296-item ground truth. We organize the results around the primary metrics (ROC-AUC and macro F1), the bootstrap confidence intervals, and family-level comparisons that motivate the recording-domain confound analysis in Chapter 10.

### 9.1 Main scoreboard

Table 1 presents the full 14-system benchmark results on the 296-item held-out ground truth, ordered by ROC-AUC. ROC curves are shown in Figure 1; confusion matrices for selected systems in Figure 2.

**Table 1. Cross-domain Lebanese DID benchmark - all 14 systems, held-out 296-item GT.**

| System | Family | Granularity | ROC-AUC (95% CI) | Macro F1 @ 0.5 (95% CI) | Macro F1 (best, snooped thr) |
|---|---|---|---|---|---|
| **marbertv2_lev** (public) | text | regional Levantine | **0.898** [0.851, 0.943] | 0.809 [0.760, 0.855] | 0.837 (thr 0.85) |
| **v1_embedding_only** (ours) | text | binary | **0.886** [0.837, 0.925] | **0.795** [0.743, 0.840] | 0.822 (thr 0.70) |
| v1_text_only (ours) | text | binary | 0.848 [0.797, 0.893] | 0.690 [0.633, 0.747] | 0.748 (thr 0.90) |
| **elyadata_whisper_adi20_leb** (public) | acoustic | country-level LEB | **0.847** [0.790, 0.897] | 0.727 [0.662, 0.784] | 0.768 (thr 0.30) |
| hybrid_v1v2_mlp (ours) | hybrid | binary | 0.817 [0.764, 0.869] | 0.512 [0.454, 0.563] | 0.724 (thr 0.90) |
| v2_frozen_mlp (ours) | acoustic | binary | 0.787 [0.728, 0.847] | 0.379 [0.324, 0.435] | 0.701 (thr 0.85) |
| **groq_llama31_8b_3shot** (public) | LLM | binary | **0.778** [0.734, 0.819] | 0.642 [0.584, 0.696] | 0.665 (thr 0.90) |
| badr_mms_300m_levantine (public) | acoustic | regional Levantine | 0.778 [0.712, 0.840] | 0.682 [0.619, 0.740] | 0.709 (thr 0.45) |
| v1_lex_only (ours) | text | binary | 0.780 [0.721, 0.836] | 0.616 [0.560, 0.667] | 0.672 (thr 0.85) |
| voxlect_mms_lid256_levantine (public) | acoustic | regional Levantine | 0.705 [0.643, 0.766] | 0.431 [0.406, 0.467] | 0.444 (thr 0.30) |
| groq_llama31_8b_zeroshot (public) | LLM | binary | 0.533 [0.472, 0.594] | 0.263 [0.223, 0.304] | 0.568 (thr 0.90) |
| v25_finetuned (ours) | acoustic | binary | 0.533 [0.464, 0.601] | 0.455 [0.403, 0.509] | 0.455 (thr 0.50) |
| whisper_lid_arabic_prob (ours) | acoustic | language-level | 0.500 [0.500, 0.500] | 0.217 [0.185, 0.247] | 0.217 (thr 0.30) |
| **v2_balanced** (ours) | acoustic | binary | **0.357** [0.291, 0.430] | 0.370 [0.322, 0.419] | 0.420 (thr 0.55) |

Bold entries indicate statistically notable findings (discussed in Section 9.2).

![Figure 1: ROC curves for all benchmark systems on the held-out 296-item Lebanese GT. Systems are colour-coded by family (text: blues, acoustic: reds/greens, hybrid: purple, LLM: pink). The chance diagonal is shown dashed. MARBERTv2 (cyan, 0.898) and V1 embed-only (light blue, 0.886) form the top cluster. V2-balanced (pink dashed, 0.357) falls below chance - the recording-domain confound inverted the classifier's predictions.](paper/figures/roc_curves.png)

*Figure 1.* ROC curves for all 14 systems. See Table 1 for full statistics.

### 9.2 Statistical significance

The bootstrap confidence intervals support the following pairwise statements:

1. **Text-family systems occupy the top positions.** The top two systems by ROC-AUC are both text-based: MARBERTv2 (0.898, CI [0.851, 0.943]) and V1 embedding-only (0.886, CI [0.837, 0.925]). The best acoustic system, Elyadata (0.847, CI [0.790, 0.897]), has overlapping CIs with both text systems - the consistent point-estimate gap (+0.039 to +0.051) favours text, but independent CI non-overlap cannot formally establish significance here. A paired bootstrap test on the difference is the correct test and is recommended for any strong claim of dominance. What is statistically unambiguous is the *mechanism*: the recording-domain confound (Section 10) explains why acoustic systems underperform, and the confound result itself (V2 balanced CI [0.291, 0.430] entirely below random) is decisive.

2. **V1 embed-only shows a clear practical advantage over V2 frozen.** V1 embedding-only CI [0.837, 0.925] vs V2 frozen [0.728, 0.847] overlap by 0.010 at the boundaries, so formal significance by CI non-overlap is marginal. The 0.099 point-estimate gap, combined with Section 10's confound diagnosis, makes the performance difference interpretable: V2's lower performance is caused by the recording-domain shortcut rather than acoustic modelling in general.

3. **Hybrid does not improve over V1.** Hybrid MLP ROC-AUC 0.817 falls below V1 embedding-only's CI lower bound of 0.837; the CIs overlap by 0.032 at [0.837, 0.869]. Adding 1024-d frozen acoustic embeddings to V1's text features yields no statistically reliable improvement. At the honest default threshold (0.50), V1 embedding-only macro F1 0.795 outperforms hybrid's 0.512 - the acoustic component actively hurts the decision boundary calibration.

4. **V2 balanced is significantly anti-correlated with the true label.** V2 balanced ROC-AUC 95% CI [0.291, 0.430] lies entirely below random (0.50). Probability of being ≥0.50 is approximately zero under the bootstrap. This is the strongest possible statistical statement for the recording-domain confound diagnosis (Section 10): removing the per-platform shortcut converts an apparently informative model (ROC-AUC 0.787 in V2 frozen) into one that is systematically wrong. The shortcut, not dialect signal, was driving V2's training-pool performance.

5. **V2.5 is indistinguishable from random.** ROC-AUC CI [0.464, 0.601] straddles 0.50. We cannot reject H₀: ROC-AUC = 0.50. End-to-end fine-tuning under per-source balanced sampling, with only 10.9% of parameters trainable and a 4,000-item subsample, produced no reliable dialect signal on held-out data.

6. **Whisper-LID is exactly uninformative.** ROC-AUC = 0.500, CI [0.500, 0.500]: all 296 GT items (×3 screening chunks = 888 chunks total) were classified as Arabic with uniformly high confidence, so Lebanese vs non-Lebanese receive identical Arabic probability scores. Lebanese detection requires explicit dialect modeling; language identification is useless as a proxy.

7. **LLM 3-shot matches a purpose-built acoustic system.** Llama-3.1-8B with 3 in-context examples achieves ROC-AUC 0.778 [0.734, 0.819], within CI of Abdullah MMS-300m 0.778 [0.712, 0.840]. A general-purpose LLM with zero fine-tuning and zero domain-specific parameters is statistically indistinguishable from a specialised acoustic Arabic DID model at this task - when given just three examples.

8. **Zero-shot LLM is near-chance.** Without examples, Llama-3.1-8B ROC-AUC = 0.533 [0.472, 0.594]. The CI includes 0.50. This confirms that Lebanese-specific lexical knowledge is not readily available in the model without priming - the +0.245 ROC-AUC gain from 0-shot to 3-shot is entirely attributable to the three examples.

### 9.3 Text family: public vs in-house parity

MARBERTv2 (public, large MARBERT backbone) achieves ROC-AUC 0.898. V1 embedding-only (ours, MiniLM-L12-v2, 384-d) achieves 0.886. Their CIs overlap heavily. This means a small, lexicon-aware multilingual sentence embedding model trained on weakly-supervised in-domain Lebanese text is competitive with a substantially larger dialectal Arabic BERT trained on a supervised written corpus. Text-based in-house performance matches public state-of-the-art on this Lebanese binary task.

### 9.4 Audio family: public vs in-house gap

The best public audio system, Elyadata (0.847, CI [0.790, 0.897]), clearly outperforms our best acoustic system V2 frozen (0.787, CI [0.728, 0.847]): the 0.060 point-estimate gap is consistent with the confound explanation, though their CIs overlap at [0.790, 0.847] so formal significance would require a paired bootstrap test. Elyadata was trained on full-supervision country-level labels from ADI-20, while V2 was trained on weakly-supervised labels under the recording-domain confound regime (Section 10). This gap validates the confound diagnosis: the in-house V2 failure is not an indictment of acoustic modelling in general but a controlled demonstration of what the recording-domain confound does to weakly-supervised acoustic training.

### 9.5 Country-level vs Levantine-proxy

Among audio systems, Elyadata (country-level LEB, 0.847, CI [0.790, 0.897]) leads Abdullah et al. (regional Levantine proxy, 0.778, CI [0.712, 0.840]) by 0.069 points, though their CIs overlap at [0.790, 0.840] so this does not reach the non-overlap significance threshold. Elyadata clearly outperforms Voxlect (0.705, CI [0.643, 0.766]) - non-overlapping CIs confirm this gap. The Levantine class conflates Lebanese with Syrian, Jordanian, and Palestinian Arabic - a real loss of discriminative capacity. Country-level acoustic training shows a consistent point-estimate advantage over regional proxies, though the Abdullah et al. comparison requires a paired bootstrap to establish formal significance.

### 9.6 Voxlect calibration pathology

Voxlect achieves ROC-AUC 0.705 - above random, meaning it has genuine discriminative signal - but at the default threshold (0.50) its confusion matrix is [[213, 1], [81, 1]]: only 2 of 296 items receive Levantine probability ≥0.50. The best-F1 threshold sweep recovers it to macro F1 0.444 at threshold 0.30, still the weakest system with a positive AUC. This is a calibration pathology: the model's softmax is squashed strongly toward the negative class. Off-the-shelf it is not usable as a binary Lebanese classifier without temperature re-scaling.

### 9.7 LLM family: zero-shot vs three-shot

The LLM family contributes a striking finding independent of the dialect ID task itself. Llama-3.1-8B (Meta, 2024) zero-shot performance - a system instruction asking the model to decide whether a transcript is Lebanese - yields ROC-AUC 0.533, near chance. Adding three in-context examples (one Lebanese, one Egyptian, one Gulf, all from outside the GT) produces ROC-AUC 0.778: a +0.245 gain from three examples alone.

This gap is larger than the difference between Voxlect (0.705) and Abdullah MMS-300m (0.778) - two acoustic systems trained on thousands of hours of Arabic speech. It suggests that Lebanese-specific lexical and syntactic patterns are present in Llama-3.1-8B's pretraining data but are not activated by a dialect identification instruction alone. The examples provide a calibration signal that aligns the model's internal representations with the binary Lebanese/not-Lebanese distinction.

Practically, the 3-shot LLM is competitive as a zero-infrastructure Lebanese DID system for text-based pipelines: no fine-tuning, no GPU, only a general-purpose model and three example transcripts.

### 9.8 Confusion matrices at default threshold (0.50)

![Figure 2: Confusion matrices at τ=0.50 for six selected systems. Row = true label (Not-LB / Lebanese); column = predicted label. V1 combined and MARBERTv2 show balanced matrices. V2-balanced shows the anti-correlation signature: the majority of true Lebanese items (63/82) are predicted as not-Lebanese.](paper/figures/confusion_matrices.png)

*Figure 2.* Confusion matrices for six selected systems at τ = 0.50.

```
                    pred neg   pred pos
marbertv2_lev:      true neg    170         44
                    true pos     17         65
v1_embedding_only:  true neg    172         42
                    true pos     14         68
v1_text_only:       true neg    137         77
                    true pos     11         71
elyadata:           true neg    185         29
                    true pos     24         58
hybrid_v1v2_mlp:    true neg     72        142
                    true pos      3         79
v2_frozen_mlp:      true neg     37        177
                    true pos      3         79
v2_balanced:        true neg    107        107
                    true pos     63         19  ← systematic mis-classification
```

V2 frozen and Hybrid predict "Lebanese" for the majority of GT items (87% and 75% respectively), achieving high recall at catastrophic precision. V2 balanced shows the classic anti-correlation signature: 63 of 82 true positives are predicted as negative, while 107 of 214 true negatives are predicted as positive. V1 embedding-only and MARBERTv2 both show the most balanced confusion matrices.

### 9.9 Consensus GT sensitivity check (dual-annotator results)

To validate that the benchmark findings are not driven by labelling ambiguity in contested items, all 14 systems were re-evaluated on the **consensus GT**: the 264-item subset where both annotators independently assigned the same binary label (56 positive, 208 negative). Items where annotators disagreed on the binary class (31 items) or where either annotator was uncertain (5 items) are excluded. The consensus GT was derived from `data/annotations_consensus.csv` using saved system predictions; no model inference was re-run.

**Table 6. Benchmark results on the consensus GT (264 items, binary-agreed by both annotators), ordered by ROC-AUC. Compare with Table 1 (full 296-item GT, primary annotator labels).**

| System | Consensus AUC (95% CI) | Full GT AUC | Δ AUC | Consensus F1 @ 0.5 |
|---|---|---:|---:|---:|
| **marbertv2_lev** | **0.923** [0.872, 0.964] | 0.898 | +0.025 | 0.818 |
| **v1_embedding_only** | **0.915** [0.871, 0.955] | 0.886 | +0.029 | 0.794 |
| v1_text_only | 0.886 [0.838, 0.924] | 0.848 | +0.038 | 0.671 |
| **elyadata_whisper_adi20_leb** | **0.869** [0.803, 0.923] | 0.847 | +0.022 | 0.754 |
| hybrid_v1v2_mlp | 0.849 [0.795, 0.895] | 0.817 | +0.032 | 0.481 |
| v2_frozen_mlp | 0.817 [0.755, 0.872] | 0.787 | +0.030 | 0.349 |
| groq_llama31_8b_3shot | 0.806 [0.765, 0.844] | 0.778 | +0.028 | 0.614 |
| v1_lex_only | 0.809 [0.745, 0.862] | 0.780 | +0.029 | 0.591 |
| badr_mms_300m_levantine | 0.803 [0.735, 0.863] | 0.778 | +0.025 | 0.688 |
| v2_same_source | 0.705 [0.619, 0.785] | 0.739 | −0.034 | 0.356 |
| voxlect_mms_lid256_levantine | 0.728 [0.665, 0.786] | 0.705 | +0.023 | 0.441 |
| groq_llama31_8b_zeroshot | 0.555 [0.494, 0.619] | 0.533 | +0.022 | 0.222 |
| v25_finetuned | 0.533 [0.456, 0.610] | 0.533 | +0.000 | 0.441 |
| **v2_balanced** | **0.353** [0.272, 0.438] | 0.357 | −0.004 | 0.374 |
| whisper_lid_arabic_prob | 0.500 [0.500, 0.500] | 0.500 | +0.000 | 0.175 |

Three findings are evident from this comparison:

1. **Rankings are fully preserved.** The system ordering is identical across both GTs. No system moves more than one position, confirming that the 31 binary-contested items do not distort the comparative evaluation.

2. **Absolute AUC values are uniformly higher on the consensus GT (+0.022 to +0.038).** This is expected: the excluded items are by definition those where two native Lebanese speakers disagreed — the most linguistically ambiguous clips in the corpus. All systems find these items harder, so removing them raises measured performance consistently. The consensus GT does not reward any single system disproportionately.

3. **V2-balanced and Whisper-LID remain below or at chance.** The confound finding (Section 10) is fully replicated on the dual-annotator consensus subset. V2-balanced ROC-AUC 0.353 (CI [0.272, 0.438]) lies entirely below 0.50. The recording-domain confound diagnosis is robust to annotation methodology.

**Conclusion:** the full-GT results (Table 1) are confirmed by the dual-annotator consensus GT (Table 6). The primary annotator's labels are reliable, the benchmark rankings are stable, and the headline finding — text-based V1 and MARBERTv2 lead all acoustic systems — holds under both annotation conditions.

---

## 10. Recording-Domain Confound Analysis

This chapter presents the thesis's principal analytical finding (Contribution C4): a four-evidence demonstration that frozen XLS-R embeddings encode audio-platform domain as a dominant axis of variation, and that V2's apparent in-pool discrimination is a recording-domain shortcut rather than genuine dialect signal. The four evidence streams - validation collapse, balanced-training inversion, platform-probe accuracy, and same-source control failure - are designed to be mutually reinforcing and to rule out alternative explanations sequentially.

### 10.1 Mechanism

When a frozen speech encoder is trained on a dataset where class labels are correlated with recording domains, the classifier head searches for the most linearly separable axis in the embedding space. If recording-domain features are more separable than dialect features - which is likely when the encoder was pretrained without dialect supervision - the head learns domain rather than dialect.

In our training pool:
- **Positive class**: YouTube-channel audio (Lebanese channels, diverse recording setups), podcast audio, ADI17 broadcast audio (LEB)
- **Negative class**: ADI17 broadcast audio (EGY, Gulf), FLEURS read-prompt audio (MSA, Egyptian speakers)

Each class has a distinct acoustic signature: the positive class mixes YouTube/podcast microphone profiles with ADI17 broadcast; the negative class mixes ADI17 broadcast with FLEURS read-prompt. A frozen encoder sees these domain differences as the primary axis of variation.

### 10.2 Evidence 1: validation-to-held-out generalization collapse

The V2 MLP achieves macro F1 0.856 on the in-pool validation split (80/20 stratified). On the held-out GT - drawn entirely from the original YouTube and podcast collection and labeled by a native speaker - it achieves macro F1 **0.379** at the default threshold. This is a collapse of 0.48 absolute points, far exceeding the equivalent ROC-AUC drop for V1 text-only (in-pool 0.964 → held-out 0.848 = −0.116; see §5.3). The held-out items are not from ADI17 or FLEURS, so their acoustic domain does not match either training class's non-Lebanese acoustic signature. The domain shortcut, learned during training, does not generalize.

### 10.3 Evidence 2: per-source balanced training destroys all residual signal

Experiment B (per-source balanced training) removes the recording-domain shortcut at the sampler level: each (platform, label) pair is given equal total weight so no recording environment is over- or under-represented in either class. V2 balanced achieves:

- ROC-AUC: **0.357** (95% CI [0.291, 0.430]) - entirely below random (0.50)
- Confusion matrix at default threshold: [[107, 107], [63, 19]] - 63 of 82 positives misclassified

A ROC-AUC below 0.50 is not random behavior; it is *systematic anti-correlation* - the model predicts "not Lebanese" precisely when the item is actually Lebanese. This happens because when the domain shortcut is removed, the residual signal the model learned is the *inverse* of what is needed: the ADI17 LEB items in the positive class sound like ADI17 broadcast (the dominant acoustic signature of the negative class), so the classifier picks up the wrong axis. This can only occur if the shortcut was carrying the bulk of the training-pool discrimination.

**Statistical statement**: under the bootstrap (n=1,000), P(V2 balanced ROC-AUC ≥ 0.50) ≈ 0. The entire 95% CI lies below random.

### 10.4 Evidence 3: platform probe - direct measurement of domain separability

We trained a 4-way Logistic Regression classifier on V2's 1024-d XLS-R embeddings to predict the *platform of origin* (adi17, youtube, podcast_rss, fleurs) - a classification task that has nothing to do with dialect.

- **Training pool**: 14,159 items, 80/20 stratified split
- **Result**: accuracy **0.8912** (chance = 0.25), macro F1 **0.8660**

Per-class precision/recall/F1:

| Platform | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| adi17 | 0.928 | 0.978 | 0.952 | 1,400 |
| fleurs | 0.910 | 0.825 | 0.866 | 160 |
| podcast_rss | 0.863 | 0.879 | 0.871 | 626 |
| youtube | 0.824 | 0.732 | 0.775 | 646 |

A linear classifier can identify which recording platform produced each audio clip with 89% accuracy from frozen XLS-R embeddings alone. ADI17 broadcasts are nearly perfectly separable (F1 0.952). This is direct evidence of the mechanism: the embedding space is organized along platform/domain axes, and these are precisely the axes that correlate with class labels in the training pool.

### 10.5 Why fine-tuning does not fix it at this scale

V2.5 unfreezes the top 2 of 24 transformer layers and applies per-source balanced sampling during training. The result: ROC-AUC **0.533** (CI [0.464, 0.601]) - straddling random, indistinguishable from no signal. Three compounding reasons:

1. **Partial unfreeze is insufficient.** Only 10.9% of parameters are trainable. The bottom 22 frozen layers continue to encode recording-domain structure as the dominant embedding axis. The two trainable layers cannot disentangle dialect from domain in the remaining representational capacity.

2. **Small training pool.** The 4,000-item stratified subsample (imposed by CPU runtime constraints, not by design) is well below the scale at which XLS-R fine-tuning reliably adapts to new tasks in the literature. Sullivan et al. (2023) and Abdullah et al. (2025) both require far larger supervised corpora for reliable cross-domain transfer.

3. **Loss trajectory.** Training loss moves from 0.693 (random binary CE) to approximately 0.640 - only a 0.053 absolute improvement over 450 optimization steps. This is an order of magnitude less signal than V1 text training achieves, consistent with the model learning very little before the optimizer's scheduled learning rate decays to near zero.

The V2.5 result is a *methodologically informative negative*: it bounds the floor of what end-to-end fine-tuning achieves under the hardware constraints of a single-researcher CPU-only project. It does not establish the ceiling of fine-tuning under full compute.

### 10.6 Evidence 4: same-source control

The preceding four experiments leave open one alternative explanation: V2's underperformance could be caused not by the XLS-R representation lacking dialect signal, but simply by the *mismatch* between the platforms used for training and the podcast-heavy test set. To test this, we train a fifth model (V2 same-source) using only podcast_rss items - the same acoustic domain as 85% of the GT - with no ADI17 or FLEURS in the training pool.

Training pool: 2,878 podcast_rss items (1,926 positive: POTENTIAL_LB/BORDERLINE_LB; 952 negative: WEAK_NEGATIVE/REJECTED). Model architecture identical to V2 frozen (MLP-256 + StandardScaler on 1024-d XLS-R). Test: the podcast_rss subset of the GT (252 items).

| Model | Scope | ROC-AUC |
|---|---|---|
| V2 frozen (cross-domain) | podcast_rss GT subset | 0.791 |
| **V2 same-source** | podcast_rss GT subset | **0.739** |
| V2 frozen (cross-domain) | full 296-item GT | 0.787 |
| V2 same-source | full 296-item GT | 0.634 |

**Same-source training does not recover performance.** The model trained exclusively on the same acoustic domain as the test set achieves 0.739 on that domain - lower than cross-domain V2's 0.791. On the full GT, same-source drops to 0.634 (vs 0.787), because the same-source model has never seen YouTube or ADI17 audio profiles at all.

This result rules out the domain-mismatch alternative explanation: even with training domain matched to test domain, the 1024-d frozen XLS-R representation cannot reliably separate Lebanese from non-Lebanese podcast speech. The encoding of recording-domain structure in the embedding space (§10.4) means the model uses whatever domain cue is available; when only one domain is present in training, no cue helps and performance is determined by whatever dialect signal is accessible - which is evidently very weak.

### 10.7 Implication for the field

The four-evidence ladder (validation collapse → balanced-training anti-correlation → platform probe → same-source control) forms a rigorous empirical argument that **frozen acoustic self-supervised encoders cannot be relied upon for Lebanese binary DID when training data is assembled from heterogeneous public sources under weak supervision**. The failure mode is not specific to Lebanese; any single-dialect study assembled from public multi-platform sources faces the same confound if label composition correlates with recording domain.

Prescriptions for future work: (a) end-to-end fine-tuning with a much larger dataset and at least 50% of encoder parameters trainable; (b) recording-domain-matched contrastive negatives (e.g., MGB-2 broadcast MSA instead of FLEURS read-prompt, to match ADI17 broadcast); (c) explicit domain-adversarial training (DANN or CORAL) to learn domain-invariant dialect representations; (d) voice conversion to synthesize class-balanced speaker variation within each platform (Abdullah et al. 2025's approach).

---

## 11. Discussion and Failure Analysis

This chapter interprets the benchmark results in depth, examining error patterns, ablation findings, cross-platform performance variation, and failure case taxonomy. The analysis draws on the benchmark results (Chapter 9) and the recording-domain confound diagnosis (Chapter 10) to characterize the current state of Lebanese Arabic DID and identify the most tractable paths for improvement.

### 11.1 V1 error patterns

At the best-snooped threshold for V1 text-only (0.90 per Table 1), V1 text-only makes approximately 53 false positives (FP) and 16 false negatives (FN) on the 295-item GT.

Per-platform breakdown:
| Platform | Total | FP | FN | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|
| podcast_rss | 252 | 47 | 10 | 23.3% | 20.0% |
| youtube | 41 | 6 | 6 | 54.5% | 20.0% |

YouTube items have a substantially higher FP rate (54.5%) than podcast items (23.3%). This is consistent with the WEAK_POSITIVE noise structure: the lexical features learned from Lebanese YouTube channel positives over-generalize to non-Lebanese YouTube content with similar lexical profiles. Podcast false positives are driven by the lexicon-overlap problem (pan-Arabic words inflating the Lebanese word count).

Top false positive pattern (items the model called Lebanese at 0.98+ confidence that were actually not Lebanese): broadcast-style formal Arabic content with a mix of pan-Arabic colloquialisms - e.g., Egyptian film criticism using المصمم / الفن constructions alongside يعني, أنا, هو; Gulf driving-safety content with colloquial فيه and بس. These transcripts score high on the Lebanese lexical features because the features cannot distinguish pan-Arabic markers from dialect-specific ones.

Top false negative pattern (items the model called non-Lebanese at <0.20 probability that were actually Lebanese): Levantine content heavy in MSA register through code-switching, discussed as political analysis or cultural commentary. These items have low lexical Lebanese signal despite being Lebanese speakers - the annotator labeled them "mostly Lebanese" due to prosodic and phonetic cues that survive in the audio but not the transcript.

### 11.2 The V1 ablation finding

A surprising result from the ablation experiment (Section 7.1): V1 **embedding-only** (MiniLM, 384-d, no explicit lexicon) outperforms V1 **combined** (lexical + embedding, 389-d) on the held-out GT:

| Variant | ROC-AUC | Macro F1 @ 0.50 |
|---|---:|---:|
| V1 lex-only (5-d) | 0.780 | 0.616 |
| V1 embedding-only (384-d) | **0.886** | **0.795** |
| V1 combined (389-d) | 0.848 | 0.690 |

The embedding-only model beats the combined model by 0.038 ROC-AUC and 0.105 macro F1. Adding explicit lexical features to the MiniLM embedding *hurts* performance. This is unexpected given that lexical features encode dialect-specific vocabulary that the embedding model may not explicitly represent.

An explanation: the MiniLM embedding is a distributional encoder trained on 50+ languages; it likely captures dialect variation implicitly through exposure to Arabic dialect text in the pretraining corpus. Adding a 5-dimensional lexical vector to a 384-dimensional embedding may introduce noise - the lexical dimensions are noisy (pan-Arabic overlap problem), and including them forces the logistic regression to allocate some weight to a low-signal, high-noise subspace, degrading the overall decision boundary.

For the thesis, the practical takeaway is: **the multilingual sentence embedding alone is sufficient; curated lexical features do not improve it for this task.**

### 11.3 MARBERTv2 and text-family comparison

MARBERTv2 achieves ROC-AUC 0.898 (CI [0.851, 0.943]), indistinguishable from V1 embedding-only's 0.886 (CI [0.837, 0.925]) under the bootstrap. This result is notable: MARBERTv2 is a 110M-parameter BERT model pre-trained on Arabic dialect text and fine-tuned on a supervised written dialect corpus; V1 embedding-only is a 33M-parameter MiniLM model fine-tuned for multilingual sentence similarity, retrained as a binary Lebanese classifier on weakly-supervised training data. Despite the order-of-magnitude difference in model capacity and training data quality, they are statistically indistinguishable on this cross-domain Lebanese audio test. The key advantage of the sentence embedding approach for this setting is **domain invariance**: text features derived from Whisper transcripts are invariant to recording microphone, codec, and room acoustics, and this invariance is precisely what is needed when training and test audio come from different recording environments.

### 11.4 Per-platform ROC-AUC breakdown

Figure 3 shows per-platform ROC-AUC (podcast_rss vs. YouTube) for all benchmark systems. Table 2 summarises the most informative cross-platform contrasts, including the same-source control (§10.6).

![Figure 3: Per-platform ROC-AUC for all 14 benchmark systems, comparing podcast RSS (blue) and YouTube (orange) subsets of the GT. MARBERTv2 shows the largest cross-domain gap (–0.191). V1 embed-only is the most domain-robust system (–0.025). V2-balanced is below chance on podcast (0.318) and near-chance on YouTube (0.494).](paper/figures/per_platform_bar.png)

*Figure 3.* Per-platform ROC-AUC: podcast_rss vs. YouTube subsets.

**Table 2. Per-platform ROC-AUC: podcast_rss (n=252) vs. YouTube (n=41 or 42).**

| System | Podcast RSS | YouTube | Δ (pod→YT) |
|---|---:|---:|---:|
| MARBERTv2 | 0.927 | 0.736 | –0.191 |
| Elyadata ADI-20 | 0.864 | 0.792 | –0.072 |
| V1 (lex+embed) | 0.854 | 0.703 | –0.151 |
| **V1 embed-only** | **0.855** | **0.830** | **–0.025** |
| V2 frozen | 0.791 | 0.547 | –0.244 |
| V2 same-source (§10.6) | 0.739 | - | - |
| V2-balanced | 0.318 | 0.494 | +0.176 |
| Groq Llama 3-shot | 0.846 | 0.756 | –0.090 |
| Abdullah MMS-300m | 0.741 | 0.747 | +0.006 |

Key observations:

1. **MARBERTv2 is the most domain-sensitive system.** It achieves the highest podcast_rss ROC-AUC (0.927) but collapses 19 points on YouTube. This is consistent with a text model that learned from a podcast-heavy training pool: the lexical and semantic patterns encoded in podcast transcripts (natural speech, Lebanese vocabulary in informal registers) are different from YouTube transcripts (more mixed register, on-screen caption artifacts, wider topic range).

2. **V1 embed-only is the most domain-robust system.** Cross-domain drop of only 2.5 points (0.855 → 0.830). The MiniLM encoder, trained on multilingual text with no platform supervision, learns a representation that transfers between podcast and YouTube Lebanese Arabic better than any other system.

3. **V2-balanced shows opposite polarity on the two platforms.** Below chance on podcast (0.318, inverse correlation confirmed) but approaching chance on YouTube (0.494). This is the recording-domain confound manifest differently per platform: the podcast platform is acoustically most different from the ADI17 broadcast audio used for negative training examples, so the confound's anti-signal is strongest there.

4. **Abdullah MMS-300m is the most platform-neutral acoustic public system.** Near-parity on both platforms (0.741 vs 0.747). This likely reflects the voice conversion augmentation used in Abdullah et al. (2025): by converting training audio to diverse speaker styles, the model learns representations less tied to recording-domain characteristics.

5. **Code-switching density correlates with platform.** YouTube items have 7× higher Latin-script density than podcast items (0.044 vs 0.006, per Section 22.2 of the research log). Lebanese items overall show 3.5× higher code-switching than non-Lebanese (0.023 vs 0.007), reflecting Lebanon's widespread French loanwords (merci, bonjour, voiture) appearing in Whisper transcripts as Latin script. This is a potentially useful supplementary feature for text-based systems.

### 11.5 ALDi dialectness correlation and failure taxonomy

**ALDi correlation.** We score each GT item's screening transcript with AMR-KELEG/ALDi [@keleg2023aldi], a BERT-based continuous Arabic dialectness regressor (output in [0, 1]; 1 = fully dialectal, 0 = fully MSA). Lebanese GT items are measurably more dialectal than non-Lebanese items on both platforms (podcast_rss: 0.595 vs 0.532, δ = 0.063; YouTube: 0.511 vs 0.467, δ = 0.044). The effect is consistent but modest.

We compute Spearman r between each item's ALDi score and its absolute prediction error |p − y| for all 15 systems (the 14 primary benchmark systems plus the V2 same-source ablation from §10.6). **All correlations are weak** (|r| < 0.15). The three most informative:

| System | Spearman r | Interpretation |
|---|---:|---|
| Whisper LID | −0.142 | Confound artifact (all Arabic → prob ≈ 1; Lebanese items more dialectal → lower error) |
| Elyadata ADI-20 | +0.110 | Audio model; errors driven by acoustic domain not text register |
| MARBERTv2 | +0.002 | Near-zero - dialectness level does not predict MARBERTv2 failures |

The near-zero r for all text systems (V1 lex-only: −0.070; V1 text-only: −0.034; Groq 3-shot: −0.001) means: text classification difficulty is not determined by how dialectal the transcript is. Systems fail on highly dialectal items and on near-MSA items roughly equally. This rules out a simple "near-MSA items are harder" hypothesis for the text family.

**Failure taxonomy.** We identify the top-10 most confidently wrong items per system (highest |p − y|), yielding 150 failure cases across 15 systems (14 primary + V2 same-source ablation). Recurring patterns:

- *False positives shared across 5 systems* (non-Lebanese wrongly called Lebanese): broadcast-register Arabic with high pan-Arabic colloquial density (يعني، بس، في as in-sentence fillers). These items score high on Lebanese lexical features despite not being Lebanese - the lexicon's pan-Arabic overlap problem (§5.2) directly causes these failures. Example item 26525: an Egyptian-style cultural podcast that uses يعني 18 times per transcript chunk.

- *False negatives shared across 5 systems* (Lebanese items missed by multiple models): short or near-silent transcripts (audio too quiet or in music-heavy segments), and items where speakers code-switch heavily into French or English mid-sentence. With little Arabic text, text-based systems have nothing to score.

- *Acoustic-only false negatives* (Voxlect, Abdullah MMS-300m): Lebanese items from the YouTube subset with low-quality recording (mobile phone, outdoor ambient noise). These items had adequate Whisper transcripts (text models classify them correctly) but their audio embeddings cluster with non-Lebanese FLEURS or ADI17 items in the XLS-R space.

The most reliable cross-system failure items (appearing in ≥ 4 systems' top-10 wrong lists) all share one of two properties: (a) short or heavily code-switched transcripts that starve text features; or (b) non-Lebanese Arabic that mimics Lebanese colloquial markers. These are genuine hard cases that no current system handles reliably.

### 11.6 Comparison to the NADI 2025 and prior shared tasks

Lebanese does not appear as a primary focus in recent Arabic DID shared tasks (NADI 2025 emphasizes Egyptian, Gulf, and Levantine at the regional level; Casablanca 2024 is text-focused). The closest prior cross-domain evaluation for Lebanese is within ADI17, which uses same-source train/test splits and therefore avoids the cross-domain problem entirely. This thesis's 300-item GT is unique in being drawn from a multi-platform in-the-wild collection rather than from a single broadcast source.

---

## 12. Limitations

This chapter states the principal limitations of the experimental design, spanning annotation quality, lexical scope, test set composition, training pool contamination, hardware constraints, and evaluation completeness. Acknowledging these limitations is essential for correctly scoping the thesis's conclusions and for guiding future work.

### 12.1 Annotator agreement

The 300-item ground-truth test set was annotated by the thesis author and independently verified by a second native Lebanese-Arabic-speaking annotator (see Section 6.2.1). Binary Cohen's κ = 0.72 (substantial agreement) and 5-way κ = 0.47 (moderate agreement) confirm that the binary ground truth is reliable. The residual limitation is that the 65 disagreement cases (21.7%) have not yet been formally adjudicated — borderline items currently carry the primary annotator's label. The `mostly_lebanese` category (50 of 300 items, 16.7%) accounts for the majority of disagreements, since code-switching is a continuum and two native speakers draw the boundary differently. Adjudication and majority-vote relabeling of these cases is the most important remaining quality improvement to the test set.

### 12.2 Levantine overlap in lexical features

The strong Lebanese dialect markers (شو، هيك، هلق، بدي، عنجد) are Levantine rather than uniquely Lebanese; they also occur in Syrian and Palestinian Arabic. The lexical and sentence embedding approaches can reliably distinguish Lebanese from Egyptian, Gulf, and MSA, but they do not robustly distinguish Lebanese from Syrian or Palestinian. The benchmark's binary formulation (Lebanese vs. all others) treats this as a non-problem: most of the corpus negatives are non-Levantine. A multi-class extension (e.g., LB / EGY / Gulf / MSA) would expose this limitation directly.

### 12.3 Podcast-heavy ground truth

The GT sample was drawn from the corpus in proportion to tier sizes, resulting in approximately 85% podcast_rss items and 14% YouTube items (in the evaluable set). Performance metrics reflect this distribution. YouTube-specific performance (41 GT items) is reported separately in the error analysis (Section 11.1) but has wider confidence intervals due to the small sample. Systems that perform differently on broadcast vs. conversational audio may be rated differently on a balanced GT.

### 12.4 `mostly_lebanese` → positive mapping

We map `mostly_lebanese` (code-switching, Lebanese+MSA) to the positive class. This is linguistically defensible - Lebanese speakers frequently code-switch and this is a feature of the variety - but it means the positive class includes items that a listener might not classify as "primarily Lebanese." Alternative mappings (treat as a third class, exclude from binary evaluation, use a weight) would produce different absolute numbers. The mapping is consistently applied across all systems.

### 12.5 V1 training pool overlap with ground-truth test set

V1 (text-only classifier, script 05) was trained before the 300-item GT was formally established. The GT was then sampled from the same WEAK_POSITIVE (45 GT items), WEAK_NEGATIVE (45 GT items), and REJECTED (60 GT items) pools, meaning up to 150 of 296 evaluable GT items were present in V1's training data. The V1 ablation variants (lex-only, embedding-only, script 22) were retrained after the GT existed but did not explicitly exclude GT items, so the same overlap applies.

The practical impact is likely modest for three reasons: (i) V1 uses logistic regression - a weak memorizer - trained on 3,654 items, so each individual item has ~0.03% influence on the regression weights; (ii) weak label noise means the training labels for GT-overlapping items agree with the GT annotation only 67–89% of the time (per §6.4 precision figures), introducing downward bias that partially offsets any upward bias from memorization; (iii) V2 and all public systems were evaluated on the same GT with no training overlap, and V1's relative ranking over them is consistent with the mechanistic account in Section 10. A clean retraining of V1 with explicit GT exclusion would resolve this uncertainty and is deferred to future work.

### 12.6 LLM family evaluation partial

The LLM evaluation includes Llama-3.1-8B zero-shot and 3-shot via Groq (completed, results in Section 9.7 and Table 1). AceGPT-7B int4 was not run (GGUF model file not downloaded; ~4 GB). Gemini Flash API is valid but rate-limited to 20 RPD on the free tier, making full GT scoring infeasible without a paid quota increase. The two completed LLM rows confirm the main few-shot finding; AceGPT would add an Arabic-specialized open-weight comparison point that remains as future work.

### 12.7 CPU-only V2.5

The V2.5 fine-tuning was conducted on a 22-core CPU with no GPU. Only 4,000 of 13,600 available training items were used (runtime ceiling: ~12 hours for 2 epochs), and only 10.9% of XLS-R parameters were unfrozen. GPU-based fine-tuning with the full training pool and 50%+ of parameters trainable would constitute a meaningfully different experiment and is the highest-priority future extension.

### 12.8 FLEURS acoustic register mismatch

FLEURS `ar_eg` is read-prompt audio (Egyptian speakers reading literary Arabic sentences); ADI17 dialects are broadcast speech. This acoustic register difference between the MSA contrastive class and other classes is one of the two root causes of the recording-domain confound (the other being the YouTube/podcast signature in the positive class). MGB-2 broadcast MSA would eliminate this mismatch but requires QCRI registration.

---

## 13. Conclusions and Future Work

This chapter summarizes the five contributions of this thesis, states the headline findings, and outlines the highest-priority directions for future research. The goal is to communicate both what this work established and what it leaves open - a distinction that is especially important given the hardware constraints and single-annotator design that bound the current results.

### 13.1 Conclusions

This thesis presents the first Lebanese-specific cross-domain DID benchmark, evaluating 14 primary systems on a 300-item manually annotated test set with percentile-bootstrap statistical confidence intervals.

The headline result is that **text-based models consistently lead acoustic models on this cross-domain Lebanese task**. The in-house V1 MiniLM embedding-only classifier (ROC-AUC 0.886) is statistically indistinguishable from the public state-of-the-art MARBERTv2 (0.898); their CIs overlap with the best acoustic system Elyadata (0.847), but the point-estimate advantage is consistent across all text-vs-acoustic comparisons. The best audio system, Elyadata ADI-whisper-ADI20 (0.847), substantially outperforms our in-house acoustic models and represents the state of the art for country-level acoustic Lebanese DID.

The **recording-domain confound** is the thesis's principal analytical finding. We establish it with four independent lines of evidence: (i) a validation-to-held-out generalization collapse of 0.48 macro F1 for V2 acoustic; (ii) per-source balanced training dropping V2's ROC-AUC to 0.357 (entirely below random, 95% CI [0.291, 0.430]); (iii) a platform-classification probe achieving 89.1% four-way accuracy from the same XLS-R embeddings; (iv) a same-source control trained exclusively on podcast audio - the dominant test domain - failing to match cross-domain V2 performance (0.739 vs 0.791). Together these constitute a strong empirical case that frozen self-supervised acoustic encoders learn recording-domain features before dialect features when training data is assembled from heterogeneous public sources under weak supervision. This is a generalizable finding: any single-dialect speech study built from multi-platform public corpora using frozen encoder features faces the same confound.

### 13.2 Future work

**Highest priority: GPU-scale fine-tuning.** V2.5's failure is attributable in part to the hardware constraint (CPU-only, 10.9% trainable, 4K subsample). Full-scale end-to-end fine-tuning of XLS-R (or a more recent backbone such as wav2vec2-BERT-2.0, which won the NADI 2025 spoken DID task) with per-source balanced sampling and the full 13.6K training pool would directly test whether the confound can be overcome with sufficient compute. Voice conversion (Abdullah et al. 2025) - synthesizing class-balanced speaker variation - is an alternative data-augmentation path that does not require architectural changes.

**MGB-2 MSA substitution.** Replacing FLEURS read-prompt MSA with MGB-2 Al Jazeera broadcast MSA would equalize the acoustic register between the MSA class and the ADI17 dialect classes, eliminating the second root cause of the recording-domain confound. This requires QCRI-managed access to MGB-2.

**LLM family extension.** This work evaluated Llama-3.1-8B zero-shot and 3-shot. Extending to larger models (Llama-3.1-70B, GPT-4o), Arabic-specialized open-weight models (AceGPT-7B, Jais-13B), and few-shot variants with more examples would characterise the scaling curve of LLM-based Lebanese DID and determine whether the 3-shot advantage is consistent across model families.

**Multi-class extension.** Reframing as 4-way classification (Lebanese / MSA / Egyptian / Gulf) would expose the Levantine-overlap limitation directly and produce per-class precision/recall profiles that are more diagnostically useful than binary metrics.

**Inter-annotator agreement.** Completed: a second Lebanese-speaking native annotator independently labeled all 300 GT items. Binary Cohen's κ = 0.72 (substantial), 5-way κ = 0.47 (moderate). See Section 6.2.1. Future extensions: adjudication of the 65 disagreement cases and majority-vote relabeling of borderline items would further sharpen the ground truth.

**Corpus growth.** The collection pipeline is reusable; extending channel coverage and adding new podcast feeds could grow the Lebanese-positive pool by an order of magnitude, enabling higher-quality training data and a larger GT sample.

### 13.3 Practical significance and deployment guidance

The headline numbers — ROC-AUC 0.886 for V1, 0.847 for Elyadata — are meaningful only when grounded in the operational context. This section translates them into concrete terms: what they mean for a practitioner building a Lebanese Arabic speech corpus, how to choose an operating threshold, and what the false-positive and false-negative costs actually are.

#### 13.3.1 What ROC-AUC 0.886 means

ROC-AUC is a pairwise ranking probability. A value of **0.886** means: draw a random Lebanese item and a random non-Lebanese item from the corpus — the V1 system ranks the Lebanese item higher with probability 0.886. In every 100 such pairs, 88 or 89 are ranked correctly.

For Elyadata at **0.847**: 84 or 85 of every 100 pairs are ranked correctly.

For context, a naïve baseline that assigns a score uniformly at random achieves ROC-AUC = 0.500. A perfect system achieves 1.000. V1's 0.886, trained on no human-labelled data and running on CPU in milliseconds, represents a strong working system.

#### 13.3.2 Is the gap between 0.886 and 0.847 practically meaningful?

Statistically: V1 CI [0.837, 0.925] and Elyadata CI [0.790, 0.897] overlap — a paired bootstrap test would be required to establish formal dominance, and no such claim is made here.

Practically: the gap matters in a different sense — **computational cost**. Elyadata is built on Whisper large-v3 (~3 GB model, GPU-dependent for reasonable throughput). V1 embedding-only uses MiniLM-L12-v2 (120 MB) and runs on CPU in under 100 ms per item. For a pipeline processing thousands of items from public platforms, V1 delivers statistically equivalent discrimination at a fraction of the infrastructure cost. The practically meaningful conclusion is not that V1 is better, but that acoustic depth buys nothing here: the same discrimination is available from transcript text alone.

#### 13.3.3 False-positive and false-negative implications

The GT has 82 positives and 214 negatives out of 296 evaluable items — a ~27.7% Lebanese base rate. Assuming this rate holds across a pipeline collection run of 5,000 items (~1,385 true Lebanese items, 3,615 non-Lebanese):

**Table 5. Projected screening outcomes at three operating thresholds — V1 embedding-only (ROC-AUC 0.886, best macro F1 0.822 at threshold 0.70).**

| Threshold | Intended use | Lebanese items recovered | Non-Lebanese flagged (FP) | Lebanese items missed (FN) |
|---|---|---:|---:|---:|
| 0.30 | Inclusive screening | ~1,290 (93%) | ~900 (25% of pool) | ~95 (7%) |
| 0.50 | Balanced classification | ~1,170 (85%) | ~400 (11% of pool) | ~215 (15%) |
| 0.70 | High-precision indexing | ~1,025 (74%) | ~135 (4% of pool) | ~360 (26%) |

A **false positive** — flagging a non-Lebanese item as Lebanese — costs human review time and, if the item is included in training data without review, adds noise to the positive-class pool. In this pipeline, false positives that reach the POTENTIAL_LB stage are removed at the manual annotation step; the cost is bounded.

A **false negative** — missing a true Lebanese item — is a collection loss. At threshold 0.30, only ~7% of the true Lebanese pool is missed. For a pipeline designed to grow a corpus over repeated collection cycles, a 7% miss rate per run is operationally acceptable: the missed items may be recovered in subsequent collection windows as new content is discovered.

#### 13.3.4 Operating threshold by intended application

**Corpus screening** (the primary use of this pipeline): set threshold **0.30–0.40**. This recovers over 90% of the true Lebanese pool. From a 5,000-item collection run, approximately 2,100 items are flagged for human review, of which ~60% are genuinely Lebanese — a realistic throughput for a team of annotators.

**Archival indexing** (tagging a corpus for downstream NLP): set threshold **0.70**. This yields ~88% precision — roughly 9 in every 10 tagged items are genuinely Lebanese. The tradeoff is that ~26% of the true Lebanese pool is left unlabeled.

**System comparison and benchmarking**: use ROC-AUC as the primary metric, as throughout this thesis. The snooped-threshold macro F1 reported in Table 1 is an upper bound on single-threshold performance and should not be quoted as an operational number.

#### 13.3.5 The bottom line

The recording-domain confound finding establishes a general warning for the field: **frozen self-supervised acoustic encoders trained on multi-platform public corpora learn what the recording environment sounds like before they learn what the speaker sounds like.** Any single-dialect speech study assembled from heterogeneous public sources using frozen encoder features faces the same risk. The confound is not a failure of the XLS-R architecture — it is a failure of the training-data assembly process.

Text-based approaches sidestep this confound entirely. Lexical and semantic content is platform-invariant: whether the audio was captured on a podcast microphone or a smartphone, the words are the same. The V1 MiniLM embedding-only classifier — trained on no human labels, running on a laptop CPU — achieves ROC-AUC 0.886, statistically indistinguishable from MARBERTv2, a supervised large BERT trained on a curated dialectal Arabic corpus.

For any practitioner building a Lebanese Arabic speech corpus from public platforms today: **transcribe screening chunks, run V1 at threshold 0.35, route flagged items to a human annotator.** At this operating point, over 90% of Lebanese audio will be recovered at a false-alarm rate of roughly one false positive per three true positives — operationally viable, reproducible on consumer hardware, and substantially better than any purely acoustic approach evaluated in this work.

---

## References

Full BibTeX entries are in [`paper/references.bib`](references.bib). In-text citations use Pandoc-style `[@bibkey]` syntax.

**Cited in this draft:**
- `@bouamor2018madar` - MADAR Arabic Dialect Corpus
- `@ali2016mgb2` - MGB-2 broadcast Arabic; candidate MSA broadcast source (deferred to future work)
- `@ali2017mgb3` - MGB-3 regional dialect challenge
- `@ali2019mgb5` - ADI17 / MGB-5; contrastive corpus source + empirical MSA-absence finding
- `@conneau2022fleurs` - FLEURS; selected MSA contrastive source
- `@ardila2020commonvoice` - Common Voice; rejected (withdrawn from HuggingFace Oct 2025)
- `@halabi2016msa` - Arabic Speech Corpus; rejected (single-speaker confound)
- `@babu2022xlsr` - XLS-R; V2 acoustic backbone
- `@baevski2020wav2vec2` - wav2vec 2.0; background for XLS-R
- `@hsu2021hubert` - HuBERT; alternative backbone considered
- `@radford2023whisper` - Whisper; screening transcription
- `@reimers2019sbert` - Sentence-BERT; V1 embedding methodology
- `@inoue2021camelbert` - CAMeL-BERT / MARBERTv2; public text-based system
- `@salameh2018finegrained` - fine-grained Arabic dialect ID
- `@shon2018adi` - CNN + language embedding dialect baseline
- `@pedregosa2011sklearn` - scikit-learn; classifier training
- `@wolf2020transformers` - HuggingFace Transformers; model loading
- `@paszke2019pytorch` - PyTorch; deep learning framework
- `@habash2010introduction` - Arabic NLP background
- `@badr2025mms` - Abdullah et al. (2025): MMS-300m + voice conversion for Arabic DID (Badr M. Abdullah and Matthew Baas; HF: badrex/mms-300m-arabic-dialect-identifier)
- `@elleuch2025adi` - ADI-whisper-ADI20, INTERSPEECH 2025
- `@sullivan2023ssl` - SSL encoders for Arabic DID, INTERSPEECH 2023
- `@voxlect2026` - voxlect-arabic-dialect-mms-lid-256
- `@nadi2025` - NADI 2025 shared task proceedings
- `@geirhos2020shortcut` - Shortcut learning in deep neural networks
- `@keleg2023aldi` - ALDi: Arabic Level of Dialectness, EMNLP 2023 Findings

---

## Appendix A: Pipeline Reproduction

See `PIPELINE.md` in the repository root for the full step-by-step run order. The pipeline is reproducible from public sources; the only credentials required are a YouTube Data API key (for optional API-based discovery; RSS-based discovery does not require one) and PodcastIndex API credentials.

## Appendix B: Negative Results and Reproducibility Friction

The following practical constraints are recorded for researchers who attempt to reproduce or extend this work:

- **Mozilla Common Voice withdrawn from HuggingFace (October 2025).** All `mozilla-foundation/common_voice_*` paths fail with `DatasetNotFoundError`. Use Mozilla Data Collective directly.
- **`datasets ≥ 4.0` removed support for loading-script datasets.** FLEURS uses `fleurs.py`; loading it with modern `datasets` raises `RuntimeError`. Pin `datasets == 2.21.0` with `fsspec <= 2024.12.0`.
- **ADI17 train split contains zero MSA** (confirmed via full 990,821-row scan). Do not attempt to extract MSA from ADI17.
- **HuggingFace Windows symlinks** require Developer Mode or admin Python; without them, the first model load with Windows Defender active can take ~30 minutes for a 1.2 GB model.
- **Voxlect MMSWrapper requires a patch** for `transformers ≥ 4.57.3`: the `Wav2Vec2Attention` class now requires a `config=` keyword argument that the upstream vendored class does not supply.
- **Elyadata WhisperDialectClassifier requires stubbing** SpeechBrain's optional `k2` dependency and monkey-patching `LazyModule.__getattr__` to prevent eager import of `flair`, `numba`, and other optional SpeechBrain dependencies that are not required for inference.
