# Thesis Roadmap — Recommendations Going Forward

**Date:** 2026-05-07
**Status of V2.5 fine-tuning:** in progress; expected GT macro F1 ∈ [0.55, 0.68], P(beat V1) ≈ 0.10
**Synthesizes:** researcher critique (in-conversation) + defensive literature review (Sullivan 2023, Badr 2025, Elleuch 2025, Shon 2018, Kuparinen 2026, NADI 2025, +20 other citations)

---

## 1. The honest state of the thesis after literature review

Three of seven framed contributions are at significant risk of being seen as duplicating recent prior work:

| Contribution | Status | Risk | Why |
|---|---|---|---|
| C1 Pipeline | Intact | Low | Multi-platform, Lebanese-specific, queue-driven combination is novel |
| C2 300-item GT | Intact, fragile | Medium | Single annotator; 252/295 podcast; speaker-bias concern (Kuparinen 2026) |
| C3 ADI17 has no MSA | **Re-frame** | High | Elleuch et al. (2025, INTERSPEECH) state this; ADI-20 fixes it |
| C4 V1 baseline 0.74 | Intact | Low | Specific feature combination on Lebanese is novel |
| C5 V2 collapse | **Re-frame** | High | Sullivan 2023, Badr 2025 report similar gaps |
| C6 Recording-domain confound | **Re-frame** | High | Shon 2018, Sullivan 2023, Badr 2025, Kuparinen 2026 all overlap |
| C7 V2.5 fine-tune | Intact, contextually weakened | Medium | Badr 2025 proposes VC; NADI 2025 top systems use selective freezing |

**Net assessment:** the thesis is **defensible** as a Lebanese-specific replication and extension of an already-documented phenomenon, **not** as a novel discovery of the recording-domain confound. The strongest framing leads with C1 (corpus pipeline), C2 (Lebanese GT), and the per-platform balancing diagnostic (a sub-result inside C6 that is genuinely novel).

---

## 2. Top priority: re-read these three papers in full before writing

Order of priority — read this week:

1. **Sullivan, Elmadany, Abdul-Mageed (2023)** — "On the Robustness of Arabic Speech Dialect Identification." INTERSPEECH 2023. https://arxiv.org/abs/2306.03789
2. **Badr et al. (2025)** — "Voice Conversion Improves Cross-Domain Robustness for Spoken Arabic Dialect Identification." INTERSPEECH 2025. https://arxiv.org/abs/2505.24713
3. **Elleuch et al. (2025)** — "ADI-20: Arabic Dialect Identification Dataset and Models." INTERSPEECH 2025. https://arxiv.org/abs/2511.10070

Plus the foundational reference:

4. **Shon, Hsu, Glass (2018)** — "Unsupervised Representation Learning of Speech for Dialect Identification." IEEE SLT 2018. https://arxiv.org/abs/1809.04458

These four papers determine how the thesis introduction and related-work section must be rewritten. Without reading them in full, any re-framing attempt will be superficial.

---

## 3. Priority experiments (one focused week of work)

In strict priority order. Each closes a specific defense-grade hole.

### P1 — Platform-probe experiment (½ day, HIGHEST VALUE)

**Goal.** Directly test the mechanism claimed in §12.6.3 — that frozen XLS-R embeddings encode recording domain.

**Method.** Train a 4-way platform classifier (youtube / podcast / adi17 / fleurs) on V2's 1024-d embeddings. Report accuracy and confusion matrix on a held-out split.

**Outcome interpretation:**
- Accuracy > 90% → mechanism claim supported directly. Section 12.6.3 becomes a 5/5 contribution.
- Accuracy near chance (25%) → mechanism claim contradicted by data. Section 12.6.3 needs softening.

**Why this matters.** This is the single most important missing experiment. It is the answer to the question "what would falsify your confound claim?" — the question every defense committee asks and that the thesis currently has no crisp answer to. Sullivan and Badr argue the confound exists; this would be the first paper to directly probe it on V2-equivalent embeddings.

**Output.** `scripts/17_platform_probe.py` → new FINDINGS Section 12.8 "Direct probing of recording-domain encoding."

### P2 — Inter-annotator agreement on a 30-50 item subset (2 days)

**Goal.** Bound the noise floor on the held-out GT.

**Method.** Recruit one Lebanese L1 speaker (ideally with linguistic background) to independently re-annotate 30-50 GT items. Compute Cohen's κ.

**Cost.** ~3 hours of annotator time. $0-50 if compensated.

**Outcome.** Adds a defensibility paragraph that pre-empts the highest-priority committee attack ("single annotator"). Lifts C2 from 3/5 to 4/5.

**Output.** New FINDINGS Section 6.3 "Inter-annotator agreement (pilot)."

### P3 — Bootstrap confidence intervals (½ day)

**Goal.** Convert point estimates into inferential claims.

**Method.** BCa bootstrap, 1000 resamples, on V1 / V2 / V2-balanced / Hybrid / V2.5 held-out macro F1 and ROC-AUC. Report 95% CIs.

**Output.** Updated tables in §9.0, §12.4, §12.6, §13.

**Why.** Reviewers will ask "is the V1>V2 gap significant?" With n=296 and macro F1 0.74 vs 0.38, yes — but you need the number.

### P4 — Same-platform Experiment B variant (½ day)

**Goal.** Disambiguate "domain shortcut" from "weak intrinsic dialect signal" in V2.

**Method.** Train V2 only on podcast_rss POTENTIAL_LB vs podcast_rss WEAK_NEGATIVE (same source, no cross-domain). Evaluate on the podcast subset of held-out GT.

**Outcome interpretation:**
- ROC-AUC > 0.75 → dialect signal exists in XLS-R; the original V2 collapse was domain shortcut.
- ROC-AUC ≈ 0.50–0.60 → dialect signal is weak in frozen XLS-R independent of domain; the architecture is the limit, not the data.

**Output.** New FINDINGS Section 12.9 "Same-source control for V2."

### P5 — V1 ablation table (½ day)

**Goal.** Quantify which V1 features carry the signal.

**Method.** Train three V1 variants:
- Lexical features only (5-d)
- Embedding only (384-d)
- Combined (current, 389-d)

Evaluate each on held-out GT.

**Output.** Table in updated Section 9.0; supports paper's "lexical features are the robust signal" claim.

### P6 — HuBERT-base reproduction of V2 (1 day)

**Goal.** Show the failure is recipe-level, not XLS-R-specific.

**Method.** Re-extract embeddings with `facebook/hubert-base-ls960` (much smaller, faster). Train MLP head, evaluate on held-out GT.

**Outcome.** If HuBERT also collapses (likely), the thesis can claim "the recording-domain confound is not XLS-R-specific." If HuBERT works (unlikely), the thesis pivots to "the choice of SSL backbone matters."

### P7 — Text baseline ablation (½ day)

**Goal.** Bound V1's strength as a baseline.

**Method.** Add TF-IDF + character n-gram LR baseline, optional fastText. Evaluate on held-out GT.

**Outcome.** Either strengthens V1's "lexical features win" framing or reveals V1 is underselling text methods.

### P-Future (deferred) — Phase 5/6/7

- **Phase 5 (multi-class):** defer until P1-P7 are complete. Multi-class is a thesis-strengthening polish, not a defense-blocker.
- **Phase 6 (full IAA):** P2 covers the minimum viable version. Full IAA is post-defense / publication work.
- **Phase 7 (lexicon enrichment):** defer. Too much work for marginal thesis benefit.

---

## 4. Re-framing the contributions for the paper

### Old framing (current paper draft)
"We discover the recording-domain confound in Arabic dialect ID; V1 text-only is the strongest model."

### New framing (recommended)
"We build a reproducible Lebanese audio corpus and replicate, on Lebanese-specific data, the cross-domain generalization gap documented by Sullivan et al. (2023) and Badr et al. (2025). Our remediation experiments — per-platform balanced training (which fails) and partial end-to-end fine-tuning (V2.5) — extend the literature with empirical evidence on what does *not* recover V1-level performance under data-scarce conditions. The thesis contributes (a) a reproducible Lebanese audio collection pipeline, (b) a 300-item Lebanese held-out test set, (c) a row-level audit of the absence of MSA in ADI17 (cf. Elleuch et al. 2025), and (d) the first direct probing experiment of recording-domain encoding in frozen XLS-R embeddings on Arabic data."

The shift: from claiming **discovery** to claiming **replication + specific empirical extensions**. This is what makes the thesis defensible. The "specific empirical extensions" are P1 and P4 above — the platform probe and the same-source control.

---

## 5. Three sharp questions to answer before defense

These came from the researcher critique and must have written answers in the thesis:

1. **What experiment, if it returned the wrong answer, would falsify the recording-domain-confound claim?** Answer: P1 (platform probe). Commit to this in writing.

2. **Why is the positive class "lebanese OR mostly_lebanese"?** Either defend the choice (code-switching is realistic Lebanese speech) or run a sensitivity analysis with strict-positive labels.

3. **If V2.5 beats V1 by 5 F1 points, what does that mean? If V2.5 underperforms by 5 F1, what does that mean?** Commit to interpretations *before* the eval number lands, not after.

---

## 6. Email to advisor — what to flag now

Suggested draft outline (not the email itself — fill in tone):

> Dear Professor,
>
> Update on the dialect identification thesis (Option 2 from our last exchange).
>
> The pipeline is complete: contrastive dataset built (ADI17 EGY/Gulf/LEB + FLEURS MSA = 7,798 items), 300-item manually-annotated held-out test set built, three classifier families evaluated end-to-end.
>
> Headline finding: the text-only baseline (V1: lexical features + multilingual sentence embeddings → LogisticRegression) outperforms both the frozen acoustic classifier (V2: wav2vec2-xls-r-300m embeddings + MLP) and the hybrid text+acoustic model on held-out data. V1 macro F1 = 0.74; V2 macro F1 = 0.38.
>
> The V2 collapse is mechanistically a recording-domain confound: the frozen encoder learns to distinguish broadcast (ADI17) vs. read-prompt (FLEURS) vs. podcast (LB) audio rather than dialect features. This is consistent with recent literature (Sullivan et al. 2023; Badr et al. 2025; Elleuch et al. 2025). I am running an end-to-end fine-tuning remediation (V2.5) on CPU; first results expected within the week.
>
> Recommended re-framing: the thesis as a Lebanese-specific replication of the cross-domain dialect-ID confound, with two original empirical contributions — a Lebanese held-out GT set and a per-platform balancing diagnostic that fails to recover performance. This is academically stronger than the original "validated detection system" framing because the negative result is more generalizable.
>
> Two questions:
> 1. Are you comfortable with this re-framing toward a methodological/negative-result thesis?
> 2. Do you have a preferred Lebanese L1 speaker contact for a 30-item inter-annotator-agreement pilot?
>
> Happy to discuss when convenient.
>
> Best,
> Charlene

---

## 7. What NOT to do (saving cycles)

- **Do not start Phase 5 (multi-class), 6 (full IAA), or 7 (lexicon enrichment)** until P1-P4 are complete.
- **Do not keep tuning V2.5 hyperparameters.** The CPU constraint caps the experiment's strength; results are interpretable either way.
- **Do not rent GPU yet.** Until P1-P4 confirm the framing, GPU experiments are premature.
- **Do not re-collect data.** The corpus is sufficient.
- **Do not delete V2 / V2.5 / hybrid results.** Negative results are the thesis. They go in the paper.

---

## 8. Final sanity check

Before any further coding, the student should be able to answer in writing:

- [ ] What is the one-sentence scientific question?
- [ ] What are the four (not seven) defensible contributions, in their final framing?
- [ ] What experiment would falsify the central claim?
- [ ] What is the interpretation of V2.5 beating V1 vs. underperforming V1?
- [ ] Have Sullivan 2023, Badr 2025, and Elleuch 2025 been read end-to-end?

Until these are answered, additional experimentation will not improve the thesis.

---

## Quick-reference timeline (one focused week)

| Day | Task | Output |
|---|---|---|
| 1 | Read Sullivan 2023 + Badr 2025 + Elleuch 2025 in full | Reading notes |
| 1 | P3 Bootstrap CIs | Updated tables |
| 2 | P1 Platform probe experiment | FINDINGS §12.8 |
| 2 | P5 V1 ablation | Updated §9.0 |
| 3 | P4 Same-source control | FINDINGS §12.9 |
| 3 | P7 Text baseline ablation | Updated §9.0 |
| 4 | P6 HuBERT reproduction | FINDINGS §12.10 |
| 4 | Update paper draft with new framing | `paper/draft.md` v2 |
| 5 | Email advisor with re-framing + ask for IAA contact | Sent |
| 5-7 | P2 IAA pilot (parallel to other work) | FINDINGS §6.3 |
| 7 | V2.5 eval lands; FINDINGS §13 auto-appended | §13 |

After this week, the thesis is in a defensible state for committee submission.
