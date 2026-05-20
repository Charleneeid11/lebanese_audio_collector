# Thesis Roadmap (v2) — Active Plan

**Date created:** 2026-05-18 (after advisor approval of benchmark reframe)
**Thesis title (advisor-approved):** *Cross-domain Lebanese Arabic Dialect Identification: Benchmarking Lexical, Acoustic, and Large Language Models*
**Predecessor:** `_deprecated_docs/ROADMAP_v1_pre_advisor_reply.md` (superseded)

---

## 1. What changed

Advisor (2026-05-18 email) endorsed the benchmark reframing of the thesis. His five specific requests:

1. Position the dataset as a **core contribution** of the thesis.
2. Add a **dedicated contributions section** stating contributions explicitly.
3. Emphasize the **cross-domain confound analysis** as the principal finding.
4. Pursue **Option 1** (Lebanese cross-domain DID benchmark) as the primary extension.
5. Add **deeper error analysis and statistical analysis** — confusion matrices, per-platform analysis, failure cases, significance testing.

V2.5 fine-tuning is **complete** (FINDINGS §13): held-out ROC-AUC 0.5331, macro F1 0.4554. *Worse* than V2 frozen, which strengthens the recording-domain confound diagnosis (§12.6.3).

---

## 2. Contributions (final, advisor-approved framing)

1. **(C1) The corpus.** A reproducible, queue-driven, multi-platform (YouTube + podcast + TikTok + ADI17 + FLEURS) Lebanese-specific audio collection pipeline yielding ~14K candidate items.
2. **(C2) The Lebanese cross-domain DID test set.** 300 manually annotated items stratified across confidence tiers and platforms. The first Lebanese-specific held-out test set for ADI evaluation.
3. **(C3) The benchmark.** Side-by-side evaluation of 8–10 systems across three families (lexical, acoustic, LLM) on the same Lebanese-specific binary task, with per-platform breakdown and bootstrap-CI significance testing.
4. **(C4) The recording-domain confound — empirically validated principal finding.** Per-platform balanced training drops V2's ROC-AUC from 0.79 to 0.36; end-to-end fine-tuning under the same balanced sampling collapses to 0.53. The diagnostic ladder (V1 > V2 frozen > Hybrid > V2 balanced > V2.5 fine-tuned) is the empirical case for lexical features dominating acoustic features for Lebanese DID at this data scale.
5. **(C5) Reproducibility audit of ADI17.** Row-level verification (990,821 rows / 40 shards) that ADI17 train contains zero MSA, complementing Elleuch et al. 2025's high-level observation.

---

## 3. What's done (do not re-do)

| Item | Status | Location |
|---|---|---|
| Multi-platform corpus | Complete | `data/queue.db`, `data/raw_audio/` |
| Contrastive items (ADI17 + FLEURS) | Complete | DB, 7,798 items |
| 300-item GT | Complete | `data/annotations.csv` |
| V1 text-only baseline | Complete (macro F1 0.7397) | `models/dialect_classifier.joblib` + FINDINGS §9.0 |
| V2 frozen acoustic | Complete (macro F1 0.38 / 0.70 snooped) | `models/dialect_classifier_v2_acoustic.joblib` + FINDINGS §12.4 |
| V2 + per-source balancing | Complete (collapsed below random) | `models/dialect_classifier_v2_balanced.joblib` + FINDINGS §12.6 Exp B |
| Hybrid V1+V2 | Complete (macro F1 0.72) | `models/dialect_classifier_hybrid.joblib` + FINDINGS §12.6 Exp C |
| V2.5 end-to-end fine-tune | Complete (macro F1 0.4554) | `models/xlsr_finetuned/` + FINDINGS §13 |
| Recording-domain confound diagnosis | Complete | FINDINGS §12.6.3 |
| Phase 4 design documentation | Complete | FINDINGS §12.7 |
| Thesis vision (Phases 5-7) | Complete | FINDINGS §14 |

---

## 4. Day-by-day plan (10 working days)

### Day 1 — Evaluation harness skeleton + existing baselines ✅ DONE 2026-05-18

**Deliverable:** `scripts/20_benchmark_harness.py` + first results CSV with existing models.

Tasks:
- ✅ Built `scripts/20_benchmark_harness.py` exposing standardized predict-then-score per system with metrics: accuracy, macro/per-class F1, precision/recall, ROC-AUC, PR-AUC, confusion matrix.
- ✅ Bootstrap CIs utility: percentile, 1000 resamples, on macro F1 @ 0.5 and ROC-AUC. (Used percentile not BCa for simplicity; defensible at this scale.)
- ✅ Added existing rows: V1 text-only, V2 frozen acoustic MLP, V2 balanced, Hybrid V1+V2, V2.5 fine-tuned. All from disk; no retraining.
- ✅ Output `data/benchmark_results.csv` with all 5 baseline rows.
- ✅ Per-item predictions persisted to `data/benchmark_predictions/*.json` for downstream per-platform / failure-case analysis.
- ✅ FINDINGS Section 15 appended with full Day 1 results table, statistical-significance findings, and confusion matrices.
- ✅ CLAUDE.md updated with new script entry.

**Key Day 1 findings:**
- V2 balanced ROC-AUC 95% CI [0.291, 0.430] entirely below random — strongest possible statistical evidence for §12.6.3 confound diagnosis.
- Hybrid does NOT significantly outperform V1 (overlapping CIs) — adding acoustic features yields no reliable improvement at default threshold.
- V2.5 ROC-AUC CI [0.464, 0.601] straddles random — fine-tuning destroyed V2's residual signal without producing replacement dialect signal.

### Day 2 — Platform probe + V1 ablation + Whisper-LID ✅ DONE 2026-05-18

**Deliverables:** FINDINGS §16/17/18, 3 new benchmark rows, platform-probe diagnostic.

Tasks:
- ✅ **Platform probe (P1):** `scripts/21_platform_probe.py` — 4-way LR on V2 embeddings -> {adi17, youtube, podcast_rss, fleurs}. **Result: accuracy 0.8912 vs chance 0.25 → mechanism claim of §12.6.3 directly supported.** FINDINGS §16 appended.
- ✅ **V1 ablation (P5):** `scripts/22_v1_ablation.py` — trained lex-only and embedding-only variants on the V1 training pool. **Surprising result: V1 embedding-only (macroF1 0.7947, ROC-AUC 0.8860) outperforms V1 combined (0.6895 / 0.8477). Lexical features hurt the combined model.** FINDINGS §17 appended.
- ✅ **Whisper-LID free-baseline:** `scripts/23_whisper_lid_baseline.py`. **Result: ROC-AUC 0.5000 — exactly random as expected; all chunks predicted Arabic uniformly.** FINDINGS §18 appended.
- ✅ Shared utilities factored into `scripts/benchmark_harness_utils.py`.
- ✅ CLAUDE.md updated with new script entries.

**Key Day 2 findings:**
- **Platform probe = 89.12% accuracy** → V2 embeddings encode platform with near-airtight separability; ADI17 alone is 95% F1. This is the single most decisive experiment for §12.6.3 in the entire thesis.
- **V1 embedding-only beats V1 combined** at both macro F1 (0.79 vs 0.69) and ROC-AUC (0.886 vs 0.848). Lexical features are net-negative when added on top of MiniLM. This is a finding worth flagging in the paper.
- **Whisper-LID gives no Lebanese signal** (ROC-AUC = 0.500 exactly). All 296 GT items × 3 chunks = 886 chunks predicted Arabic. Zero discriminative power.

### Day 3 — Public HuggingFace ADI systems ✅ DONE 2026-05-20

**Deliverable:** 4 public-system rows in benchmark CSV + FINDINGS §19.

Tasks (all use `data/audio_clips_compact/` 10s clips):
- ✅ `badrex/mms-300m-arabic-dialect-identifier` — Badr 2025's voice-conversion-trained MMS-300M.
- ✅ `tiantiaf/voxlect-arabic-dialect-mms-lid-256` — Voxlect 2026 benchmark model. Required vendoring `MMSWrapper` from github.com/tiantiaf0627/voxlect + patching it for transformers 4.57.3 (`Wav2Vec2Attention` now requires `config=` kwarg).
- ✅ `Elyadata/ADI-whisper-ADI20` — Elleuch et al. 2025 Whisper-based ADI20 model. Required vendoring `WhisperDialectClassifier` from github.com/elyadata/ADI-20 + stubbing speechbrain's optional `k2` dep + monkey-patching `LazyModule.__getattr__` so Python's `inspect.hasattr(module, '__file__')` introspection short-circuits instead of forcing eager imports of `flair`, `numba`, etc.
- ✅ `IbrahimAmin/marbertv2-arabic-written-dialect-classifier` — text-based, applied to existing Whisper transcripts.

Engineering: `scripts/24_eval_public_systems.py` dispatches via a `loader` field in the SYSTEMS dict. Per-system pause/resume via `data/benchmark_predictions/<name>.json`. FINDINGS §19 write is idempotent (strips prior block before appending).

**Key Day 3 findings (full interpretation in FINDINGS §19.3):**
- **MARBERTv2 (text, regional Levantine proxy) wins overall** — ROC-AUC 0.898 (CI [0.851, 0.943]).
- **Elyadata is the best audio system** — ROC-AUC 0.847 (CI [0.790, 0.897]); the only country-level `LEB`-trained model in the comparison.
- **Country-level beats regional-Levantine proxy at the audio level** — Elyadata 0.847 > Badr 0.778 > Voxlect 0.705. The Levantine label conflates LB+SY+JO+PA and costs ~0.07-0.14 ROC-AUC.
- **Voxlect has a calibration pathology** — ROC-AUC 0.705 (above random) but at threshold 0.5 it predicts essentially nothing as positive ([[213, 1], [81, 1]]). Best-F1 sweep recovers it slightly to 0.444 @ thr=0.30. Confirms it ranks adequately but is unusable as a binary classifier off-the-shelf.
- **Public-vs-in-house parity on lexical, gap on acoustic.** V1 embedding-only (ours, 0.886) and MARBERTv2 (public, 0.898) have overlapping CIs. Elyadata (public, 0.847) significantly beats V2-frozen (ours, 0.786) on acoustic — confirming the recording-domain confound diagnosis of §12.6.3 / §16: V2 underperforms because it was trained under platform-confounded weak supervision, not because acoustic modelling is inherently broken.
- **Consolidated 12-system scoreboard** lives in FINDINGS §19.3.

### Day 4 — Arabic LLMs (open-weight + Gemini free tier)

**Deliverable:** 2-3 LLM rows in benchmark CSV.

Tasks:
- **Open-weight Arabic LLM:** AceGPT-7B int4 via `llama-cpp-python` on the 296 Whisper transcripts. Zero-shot prompt: *"Identify the dialect of this Arabic text: Lebanese, Egyptian, Gulf, MSA, or other."* Plus 3-shot variant.
- **Gemini AI Studio (free tier):** same prompts via Gemini 2.5 Flash API key (free with rate limits).
- Optional, time permitting: Jais-13b open weights if it fits CPU memory.
- Each gets zero-shot + 3-shot rows.

### Day 5 — Per-platform breakdown + code-switching analysis

**Deliverable:** Per-platform tables for every system + code-switching feature column.

Tasks:
- Per-platform breakdown utility: for each system in the CSV, compute macro F1 and ROC-AUC restricted to (a) podcast subset, (b) youtube subset, (c) ADI17 subset of GT, (d) FLEURS subset.
- Code-switching density per item: regex over existing transcripts for non-Arabic-script tokens (Latin letters, French/English markers). Compute correlation with per-system error rate.
- Output: `data/benchmark_per_platform.csv`, `data/code_switching_features.csv`.

### Day 6 — ALDi + same-source control

**Deliverable:** ALDi correlation + V2 same-source control row.

Tasks:
- **ALDi (Arabic Level of Dialectness)** continuous scoring per GT item using AMR-KELEG/ALDi (public HF model). Correlate ALDi score with per-system accuracy. Adds a sociolinguistic-vs-computational analysis dimension.
- **Same-source control (P4):** train V2 only on podcast_rss POTENTIAL_LB vs podcast_rss WEAK_NEGATIVE (same domain). Evaluate on podcast subset of GT. Disambiguates "domain shortcut" from "weak intrinsic acoustic signal." One row.

### Day 7 — Failure-case selection + final tables

**Deliverable:** Failure-case appendix + final benchmark results table + figures.

Tasks:
- For each system: pick top 10 most-confidently-wrong items (highest |probability − label|). Output: `data/failure_cases.csv` with system_name, item_id, transcript, true_label, predicted_prob, audio_path.
- Generate paper-ready figures:
  - ROC curves for all systems (one panel)
  - Per-platform bar chart (one panel per family)
  - Confusion matrices (small-multiples grid)
- Compute pairwise significance: bootstrap CI on (V1 macro F1 − each other system's macro F1).

### Day 8 — Paper rewrite: intro + abstract + related work + contributions

**Deliverable:** `paper/draft.md` v2 with new framing.

Tasks:
- New title: *Cross-domain Lebanese Arabic Dialect Identification: Benchmarking Lexical, Acoustic, and Large Language Models*
- New abstract centered on benchmark contribution + recording-domain confound principal finding
- New §1 with explicit numbered contributions (C1-C5 from §2 above)
- §2 Related work rewrite citing Sullivan 2023, Badr 2025, Elleuch 2025, NADI 2025, Casablanca 2024, Voxlect 2026, AraDiCE, ALDi, Shon 2018, Geirhos 2020. Position the thesis as Lebanese-specific replication + extension.

### Day 9 — Paper rewrite: corpus + GT + benchmark sections

**Deliverable:** `paper/draft.md` §3-6 rewritten.

Tasks:
- §3 Corpus collection (existing content, lightly edited)
- §4 GT construction (existing content, add limitations subsection)
- §5 Benchmark methodology (new — protocol, metrics, statistical setup)
- §6 Benchmark results (new — main results table + per-platform breakdown + bootstrap CIs)

### Day 10 — Paper rewrite: analysis + discussion + limitations + future work

**Deliverable:** `paper/draft.md` complete.

Tasks:
- §7 Recording-domain confound analysis (promoted from §12.6.3 in FINDINGS)
- §8 Failure-case taxonomy + code-switching + ALDi analyses
- §9 Discussion
- §10 Limitations (single annotator, podcast-heavy GT, mostly_lebanese → positive choice, CPU-only V2.5)
- §11 Future work (multi-class, IAA at scale, lexicon enrichment, voice conversion remediation following Badr 2025)
- Update `paper/references.bib`: add Sullivan 2023, Badr 2025, Elleuch 2025, NADI 2025, Casablanca 2024, Voxlect 2026, AraDiCE, ALDi, Geirhos 2020, Shon 2018.

---

## 5. Beyond Day 10 — polish & submission

- **Day 11-12:** Polish pass, formatting, figures, table of contents, abstract refinement.
- **Day 13:** Send draft to advisor for review.
- **Day 14+:** Address advisor feedback, finalize.

---

## 6. Hardware / cost summary

- **All compute local CPU**, except:
  - Gemini AI Studio (free tier, rate-limited)
  - Optionally GPT-4o (~$10-15 if used as one row; skip if budget zero)
- **Total expected out-of-pocket: $0** (open models + Gemini free tier sufficient)
- **Optional add-on: $10-15** for a GPT-4o closed-model upper-bound row
- **Network downloads needed:** Badr MMS (~1.2 GB), Voxlect (~1 GB), Elyadata ADI-whisper (~1 GB), MARBERTv2 (~600 MB), AceGPT-7B int4 (~5 GB), ALDi (~400 MB). Total ~9 GB. **Plan accordingly given data caps.**

---

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| HuggingFace model API change or HF down | Low | Cache models locally on first download |
| Some Arabic LLMs gated (require approval) | Medium | Apply early for AceGPT/Fanar/Jais; fall back to open models that work |
| Gemini free tier rate limits | Medium | Pace requests; if blocked, can buy paid token allotment (~$5) |
| AceGPT-7B int4 too slow on CPU | Medium | Fall back to MARBERTv2 + Gemini only for LLM family |
| Network data cap hit | Medium | Prioritize Badr + Voxlect + MARBERTv2; defer AceGPT if needed |
| Paper rewrite takes longer than 3 days | High | Days 11-12 are buffer; this thesis is now well-scoped enough to write quickly |

---

## 8. Out of scope (do NOT do)

These were considered and dropped from this plan:

- **Multi-class GT re-annotation** (would need ~4-6 hrs of your labeling; advisor did not request)
- **Inter-annotator agreement pilot** (advisor did not request; the listener study was a researcher's suggestion that requires human recruiting)
- **HuBERT-base V2 reproduction** (P6 in old roadmap; redundant given the four-tier V1/V2/V2.5 ladder already covers backbone variation)
- **TF-IDF + character n-gram V1 ablation** (P7 in old roadmap; V1 ablation in Day 2 already covers lexical-vs-embedding decomposition)
- **Voice conversion (kNN-VC) remediation** (Badr 2025's method; goes in future work section)
- **Phase 5 multi-class extension** (FINDINGS §14; future work)
- **Phase 6 full IAA** (FINDINGS §14; future work)
- **Phase 7 lexicon enrichment** (FINDINGS §14; future work)
- **Domain-adversarial training / DANN** (too brittle, too slow)

---

## 9. Definition of "done"

The thesis is complete when:
- All 10 days of Section 4 are executed
- `data/benchmark_results.csv` contains all 12-15 system rows with metrics + bootstrap CIs
- `data/benchmark_per_platform.csv` contains per-(system, platform) breakdown
- `paper/draft.md` is fully rewritten in the new framing, ~30-50 pages
- Advisor has approved a near-final draft

Ready-to-submit signal: the abstract reads as a benchmark contribution; §1 contributions are numbered; §7 confound analysis is the empirical centerpiece; future work names voice conversion (Badr 2025) and multi-class extension as the natural follow-ups.
