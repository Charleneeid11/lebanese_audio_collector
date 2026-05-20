#!/usr/bin/env python3
"""
Evaluate the locally-fine-tuned V2.5 checkpoint on the 300-item held-out GT,
save HF-format model, and append FINDINGS Section 13.

Inputs:
  models/xlsr_finetune_ckpt/latest.pt   — local fine-tune checkpoint
  data/annotations.csv                  — held-out GT
  data/audio_clips_compact/             — 10s clips

Outputs:
  models/xlsr_finetuned/                — model in HF format (for re-use)
  models/v25_eval_local.json            — eval metrics
  FINDINGS.md                           — appended Section 13

Run: python scripts/16b_eval_local_v25.py
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
NUM_LABELS = 2
SAMPLE_RATE = 16000
MAX_LEN_SECONDS = 10

CKPT_PATH = Path("models/xlsr_finetune_ckpt/latest.pt")
OUT_DIR = Path("models/xlsr_finetuned")
LOCAL_EVAL_JSON = Path("models/v25_eval_local.json")
ANNOTATIONS_CSV = Path("data/annotations.csv")
CLIPS_DIR = Path("data/audio_clips_compact")
FINDINGS_PATH = Path("FINDINGS.md")

GT_BINARY_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}
GT_EXCLUDE = {"unclear", "skip", "", None}


def main() -> int:
    if not CKPT_PATH.exists():
        print(f"ERROR: {CKPT_PATH} not found.")
        return 1

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    import librosa

    print(f"Initializing {MODEL_NAME} architecture...", flush=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    print(f"Loading checkpoint {CKPT_PATH}...", flush=True)
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    state = ck.get("state", {})
    print(f"  resumed from global_step={state.get('global_step', '?')}, epoch={state.get('epoch', '?')}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving HF-format model to {OUT_DIR}/...", flush=True)
    model.save_pretrained(str(OUT_DIR))
    feature_extractor.save_pretrained(str(OUT_DIR))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"  device: {device}", flush=True)

    print(f"\nLoading {ANNOTATIONS_CSV}...", flush=True)
    annotations = []
    with open(ANNOTATIONS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("ground_truth") or "").strip().lower()
            if raw in GT_EXCLUDE:
                continue
            y = GT_BINARY_MAP.get(raw)
            if y is None:
                continue
            iid = int(row["item_id"])
            clip = CLIPS_DIR / f"item_{iid}.mp3"
            if not clip.exists():
                continue
            annotations.append({"item_id": iid, "y": y, "clip": clip, "raw": raw})
    print(f"  evaluable: {len(annotations)}  pos={sum(1 for a in annotations if a['y']==1)}  neg={sum(1 for a in annotations if a['y']==0)}", flush=True)

    print("\nRunning inference...", flush=True)
    probs = []
    labels = []
    for i, a in enumerate(annotations):
        y_audio, _ = librosa.load(str(a["clip"]), sr=SAMPLE_RATE, mono=True)
        if y_audio.shape[0] > SAMPLE_RATE * MAX_LEN_SECONDS:
            y_audio = y_audio[: SAMPLE_RATE * MAX_LEN_SECONDS]
        inputs = feature_extractor(y_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        iv = inputs["input_values"].to(device)
        am = inputs.get("attention_mask")
        if am is not None:
            am = am.to(device)
        with torch.no_grad():
            out = model(input_values=iv, attention_mask=am)
        p = torch.softmax(out.logits, dim=-1)[0, 1].item()
        probs.append(p)
        labels.append(a["y"])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(annotations)}", flush=True)

    y = np.array(labels)
    p = np.array(probs)

    def metrics_at(thr):
        preds = (p >= thr).astype(int)
        pr, rc, f1, support = precision_recall_fscore_support(y, preds, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        return {
            "threshold": float(thr),
            "accuracy": float(accuracy_score(y, preds)),
            "precision_neg": float(pr[0]), "recall_neg": float(rc[0]), "f1_neg": float(f1[0]),
            "support_neg": int(support[0]),
            "precision_pos": float(pr[1]), "recall_pos": float(rc[1]), "f1_pos": float(f1[1]),
            "support_pos": int(support[1]),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(),
        }

    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
    pr_auc = float(average_precision_score(y, p)) if len(set(y)) > 1 else None
    threshold_results = [metrics_at(thr) for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]]
    best = max(threshold_results, key=lambda m: m["macro_f1"])

    eval_payload = {
        "n": int(len(y)),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "thresholds": threshold_results,
        "best_threshold": best,
        "checkpoint_global_step": state.get("global_step"),
        "checkpoint_epoch": state.get("epoch"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    LOCAL_EVAL_JSON.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    print(f"\nResults saved to {LOCAL_EVAL_JSON}")
    print(f"  ROC-AUC: {auc:.4f}  PR-AUC: {pr_auc:.4f}")
    for m in threshold_results:
        print(f"  thr={m['threshold']:.2f}  acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}")
    print(f"  best by macroF1: thr={best['threshold']:.2f} -> macroF1={best['macro_f1']:.4f}")

    print(f"\nAppending Section 13 to {FINDINGS_PATH}...", flush=True)
    section = ["\n\n## 13. V2.5 — Locally Fine-tuned XLS-R\n"]
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
    section.append("- Batch size: 4 × grad_accum 4 = effective 16\n")
    section.append("- Epochs: 2 (nominal 450 optim steps)\n")
    section.append("- Training pool: 4000 items stratified subsample of 13.6K (CPU runtime ceiling)\n")
    section.append("- Audio: 10s clips at 16 kHz, MP3 @ 64k\n")
    section.append("- Hardware: 22-core CPU, no GPU\n")
    section.append(f"- Checkpoint loaded: global_step={state.get('global_step')}, epoch={state.get('epoch')}\n\n")

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

    section.append(f"**Best threshold by macro F1: {best['threshold']:.2f} → macro F1 = {best['macro_f1']:.4f}** (snooped on GT; reported alongside default 0.5 as honest evaluation).\n\n")

    section.append("### 13.4 Final scoreboard — V1 vs V2 vs Hybrid vs V2.5\n")
    section.append("| Model | ROC-AUC | Macro F1 (default 0.5) | Macro F1 (best snooped) |\n")
    section.append("|-------|--------:|-----------------------:|------------------------:|\n")
    section.append("| V1 text-only (§9.0) | 0.8477 | — | 0.7397 (thr 0.70) |\n")
    section.append("| V2 frozen acoustic MLP (§12.4) | 0.7865 | 0.3794 | 0.7010 (thr 0.85) |\n")
    section.append("| V2 + per-source balancing (§12.6 Exp B) | 0.3569 | 0.3700 | 0.4196 (thr 0.55) |\n")
    section.append("| Hybrid V1+V2 MLP (§12.6 Exp C) | 0.8172 | 0.5116 | 0.7235 (thr 0.90) |\n")
    section.append(f"| **V2.5 fine-tuned XLS-R** | **{auc:.4f}** | **{threshold_results[0]['macro_f1']:.4f}** | **{best['macro_f1']:.4f} (thr {best['threshold']:.2f})** |\n\n")

    if best["macro_f1"] > 0.7397:
        section.append(
            f"**V2.5 surpasses V1** by **{best['macro_f1'] - 0.7397:+.4f}** macro F1 (snooped threshold) on "
            "the held-out 300-item ground truth. End-to-end fine-tuning under per-source balanced sampling "
            "recovers dialect signal from the frozen-encoder collapse documented in §12.6.3.\n"
        )
    else:
        section.append(
            f"V2.5 reaches macro F1 **{best['macro_f1']:.4f}** (best snooped) on the held-out 300-item ground truth. "
            "V1's text-only baseline remains the strongest model at 0.7397 macro F1. The result is consistent with "
            "the recording-domain confound diagnosis (§12.6.3): under CPU-constrained training (top 2 of 24 transformer "
            "layers trainable, 4000-item stratified subsample, 2 epochs), partial fine-tuning is insufficient to "
            "recover V1-level performance. The result is also consistent with the broader Arabic dialect ID literature "
            "(Sullivan, Elmadany & Abdul-Mageed 2023; Badr et al. 2025), which reports analogous cross-domain "
            "generalization failures for frozen and partially fine-tuned SSL encoders on heterogeneous-source corpora. "
            "This is a methodologically informative negative result: it bounds the *floor* of what end-to-end fine-tuning "
            "achieves under the hardware constraints of a single-researcher Master's project, not the ceiling.\n"
        )

    with open(FINDINGS_PATH, "a", encoding="utf-8") as f:
        f.write("".join(section))
    print("Appended Section 13.\n", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
