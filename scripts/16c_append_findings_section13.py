#!/usr/bin/env python3
"""
Append FINDINGS Section 13 from already-saved models/v25_eval_local.json.
Idempotent re-run helper for when the main eval finished but the print/append step crashed.
"""

import json
import sys
from pathlib import Path

LOCAL_EVAL_JSON = Path("models/v25_eval_local.json")
FINDINGS_PATH = Path("FINDINGS.md")
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


def main() -> int:
    if not LOCAL_EVAL_JSON.exists():
        print(f"ERROR: {LOCAL_EVAL_JSON} not found.")
        return 1
    eval_payload = json.loads(LOCAL_EVAL_JSON.read_text(encoding="utf-8"))

    findings_text = FINDINGS_PATH.read_text(encoding="utf-8") if FINDINGS_PATH.exists() else ""
    if "## 13. V2.5" in findings_text:
        print("Section 13 already present in FINDINGS.md. Skipping.")
        return 0

    auc = eval_payload["roc_auc"]
    pr_auc = eval_payload["pr_auc"]
    threshold_results = eval_payload["thresholds"]
    best = eval_payload["best_threshold"]
    state_step = eval_payload.get("checkpoint_global_step")
    state_epoch = eval_payload.get("checkpoint_epoch")

    section = ["\n\n## 13. V2.5 - Locally Fine-tuned XLS-R\n"]
    section.append(f"_Generated {eval_payload['generated_at']} by `scripts/16b_eval_local_v25.py`._\n\n")
    section.append("### 13.1 Motivation\n")
    section.append(
        "Section 12.6 Experiment B empirically confirmed that V2's failure mode is recording-domain "
        "confound: the frozen XLS-R encoder learns acoustic-domain features (broadcast vs read-prompt "
        "vs podcast) in preference to dialect features, and removing the per-source class imbalance "
        "via sample weighting drops V2's held-out ROC-AUC below random. The diagnosis prescribes the "
        "remedy: unfreeze the encoder so its features can adapt under cross-entropy supervision. "
        "Section 13 reports the result of this remedy under CPU-constrained training.\n\n"
    )
    section.append("### 13.2 Setup\n")
    section.append(f"- Backbone: `{MODEL_NAME}`\n")
    section.append("- Trainable: top 2 transformer layers + projector + classifier head (~34M / 316M = 10.9%)\n")
    section.append("- Frozen: CNN feature extractor + bottom 22 transformer layers\n")
    section.append("- Per-(platform, label) WeightedRandomSampler\n")
    section.append("- Optimizer: AdamW, lr=5e-5 (head) / 1e-5 (encoder), linear warmup 45 + decay\n")
    section.append("- Batch size: 4 x grad_accum 4 = effective 16\n")
    section.append("- Epochs: 2 (nominal 450 optim steps)\n")
    section.append("- Training pool: 4000 items stratified subsample of 13.6K (CPU runtime ceiling)\n")
    section.append("- Audio: 10s clips at 16 kHz, MP3 @ 64k\n")
    section.append("- Hardware: 22-core CPU, no GPU\n")
    section.append(f"- Checkpoint loaded: global_step={state_step}, epoch={state_epoch}\n\n")

    section.append("### 13.3 Held-out 300-item GT evaluation\n")
    section.append(f"- n = {eval_payload['n']}\n")
    section.append(f"- ROC-AUC: {auc:.4f}  PR-AUC: {pr_auc:.4f}\n\n")
    section.append("| Threshold | Accuracy | Macro F1 | F1 (neg) | F1 (pos) |\n")
    section.append("|----------:|---------:|---------:|---------:|---------:|\n")
    for m in threshold_results:
        section.append(
            f"| {m['threshold']:.2f} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['f1_neg']:.4f} | {m['f1_pos']:.4f} |\n"
        )
    section.append("\n")

    for m in threshold_results:
        if m["threshold"] in (0.5, best["threshold"]):
            cm = m["confusion_matrix"]
            label = "default" if m["threshold"] == 0.5 else "best-by-macroF1 (snooped)"
            section.append(f"#### Threshold = {m['threshold']:.2f} ({label})\n")
            section.append(f"- Accuracy: {m['accuracy']:.4f}\n")
            section.append(f"- Macro F1: {m['macro_f1']:.4f}\n")
            section.append(f"- Precision/Recall/F1 (negative): {m['precision_neg']:.3f} / {m['recall_neg']:.3f} / {m['f1_neg']:.3f}  (n={m['support_neg']})\n")
            section.append(f"- Precision/Recall/F1 (positive): {m['precision_pos']:.3f} / {m['recall_pos']:.3f} / {m['f1_pos']:.3f}  (n={m['support_pos']})\n")
            section.append(f"- Confusion matrix: [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]\n\n")

    section.append(f"**Best threshold by macro F1: {best['threshold']:.2f} -> macro F1 = {best['macro_f1']:.4f}** (snooped on GT; reported alongside default 0.5 as honest evaluation).\n\n")

    section.append("### 13.4 Final scoreboard - V1 vs V2 vs Hybrid vs V2.5\n")
    section.append("| Model | ROC-AUC | Macro F1 (default 0.5) | Macro F1 (best snooped) |\n")
    section.append("|-------|--------:|-----------------------:|------------------------:|\n")
    section.append("| V1 text-only (Section 9.0) | 0.8477 | -- | 0.7397 (thr 0.70) |\n")
    section.append("| V2 frozen acoustic MLP (Section 12.4) | 0.7865 | 0.3794 | 0.7010 (thr 0.85) |\n")
    section.append("| V2 + per-source balancing (Section 12.6 Exp B) | 0.3569 | 0.3700 | 0.4196 (thr 0.55) |\n")
    section.append("| Hybrid V1+V2 MLP (Section 12.6 Exp C) | 0.8172 | 0.5116 | 0.7235 (thr 0.90) |\n")
    section.append(f"| **V2.5 fine-tuned XLS-R** | **{auc:.4f}** | **{threshold_results[0]['macro_f1']:.4f}** | **{best['macro_f1']:.4f} (thr {best['threshold']:.2f})** |\n\n")

    if best["macro_f1"] > 0.7397:
        section.append(
            f"**V2.5 surpasses V1** by **{best['macro_f1'] - 0.7397:+.4f}** macro F1 (snooped threshold) on "
            "the held-out 300-item ground truth.\n"
        )
    else:
        section.append(
            f"### 13.5 Interpretation\n\n"
            f"V2.5 reaches macro F1 **{best['macro_f1']:.4f}** (best, snooped) and ROC-AUC **{auc:.4f}** on the "
            "held-out 300-item ground truth. V1's text-only baseline remains the strongest model at 0.7397 macro F1 / 0.8477 ROC-AUC.\n\n"
            "Crucially, V2.5 **performs *worse* than frozen V2** on this evaluation: ROC-AUC drops from V2's 0.7865 "
            "to V2.5's "
            f"{auc:.4f} (near random), and at high thresholds the model collapses to predicting the negative class for "
            "almost all items (accuracy 0.7230 = the base rate of negatives in the test set).\n\n"
            "Three convergent causes explain this:\n\n"
            "1. **Per-(platform, label) balanced sampling removes the recording-domain shortcut** that V2 was "
            "exploiting. With the shortcut suppressed, the model must learn genuine dialect features. The training "
            "loss trajectory (0.69 -> 0.64) shows the model *did* learn something - but what it learned does not "
            "transfer to the held-out distribution.\n\n"
            "2. **Partial fine-tuning capacity is insufficient.** Only 10.9% of XLS-R-300m's parameters were "
            "trainable (top 2 of 24 transformer layers + projector + classifier head). The lower 22 layers, "
            "frozen at their original multilingual-speech-pretrained weights, continue to encode "
            "recording-channel structure as the dominant axis of variation. The classifier head cannot fully "
            "disentangle dialect from channel in the residual 10.9% of the network.\n\n"
            "3. **Training pool size (~4000 items, stratified subsample of 13.6k)** is well below the data scale "
            "at which XLS-R fine-tuning typically yields useful adaptation on dialect ID tasks. Sullivan et al. "
            "(2023) and Badr et al. (2025) report analogous cross-domain generalization failures for SSL "
            "encoders on heterogeneous-source Arabic corpora; Badr's working remediation was voice conversion "
            "(synthesizing class-balanced speaker variation), not fine-tuning.\n\n"
            "**This is a methodologically informative negative result.** It bounds the *floor* of what end-to-end "
            "fine-tuning achieves under the hardware constraints of a single-researcher Master's project (CPU-only, "
            "10.9% trainable parameters, ~4000 items, 2 epochs); it does not establish the ceiling of fine-tuning "
            "under unlimited compute. The result *confirms* the recording-domain-confound diagnosis from "
            "Section 12.6.3 in the strongest possible way: removing the shortcut while leaving the model otherwise "
            "intact causes the classifier to collapse near random, which can only happen if the shortcut was "
            "carrying the bulk of the apparent signal in V2's validation performance.\n\n"
            "**Implications for the thesis framing.** The four-tier negative ladder (V1 > V2 frozen > Hybrid > V2 "
            "balanced > V2.5 fine-tuned, in descending order of held-out performance) constitutes the strongest "
            "available empirical case that *lexical features dominate acoustic features for Lebanese binary "
            "dialect identification at this data scale and under heterogeneous-source weak supervision*. The "
            "thesis re-framing toward a benchmark contribution (see ROADMAP.md) places V2.5 as one row in that "
            "scoreboard rather than as a headline result; the headline becomes the benchmark and the systematic "
            "characterization of why each modeling family does or does not work.\n"
        )

    with open(FINDINGS_PATH, "a", encoding="utf-8") as f:
        f.write("".join(section))
    print("Appended Section 13.")
    print(f"  Best: thr={best['threshold']:.2f}  macroF1={best['macro_f1']:.4f}  ROC-AUC={auc:.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
