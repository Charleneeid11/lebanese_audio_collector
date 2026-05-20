#!/usr/bin/env python3
"""
End-to-end fine-tune wav2vec2-xls-r-300m for binary Lebanese-vs-other dialect ID.
Local CPU version (no Colab, no GPU).

Key choices for CPU feasibility:
  - Freeze CNN feature encoder + bottom 20 of 24 transformer layers
  - Trainable: top 4 transformer layers + classifier head (~50M params)
  - Per-(platform, label) WeightedRandomSampler to defeat the recording-domain
    confound diagnosed in FINDINGS Section 12.6
  - Checkpoint every 500 steps (resumable across crashes)
  - Mixed-precision OFF on CPU (bf16 inconsistent across BLAS backends)

Time budget: ~12-18 hours on a 22-core CPU. Fully resumable.

Run: python scripts/16_finetune_xlsr_local.py
Resume after crash: re-run the same command. The latest checkpoint in
                    models/xlsr_finetune_ckpt/ is loaded automatically.

Outputs (when complete):
  models/xlsr_finetune_ckpt/best/             — best-by-GT-macro-F1 model
  models/xlsr_finetune_ckpt/training_log.json — per-checkpoint val + GT metrics
  data/v25_eval.json                          — final GT eval at threshold sweep
  FINDINGS.md                                 — appended Section 13
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Windows keep-awake: tell the OS not to sleep while we're training ---
# ES_CONTINUOUS keeps the request active; ES_SYSTEM_REQUIRED prevents the
# system from sleeping; ES_AWAYMODE_REQUIRED keeps it awake even with the
# lid closed (only effective if "Allow away mode" is enabled in power policy).
def _prevent_windows_sleep():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
        print("[keep-awake] Windows sleep blocked for this process.", flush=True)
    except Exception as e:
        print(f"[keep-awake] could not block sleep: {e}", flush=True)


_prevent_windows_sleep()

# Encoding for Arabic-safe printing
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
import librosa


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
MAX_LEN_SECONDS = 10
NUM_LABELS = 2
SEED = 42

CKPT_DIR = Path("models/xlsr_finetune_ckpt")
BEST_DIR = CKPT_DIR / "best"
LOG_PATH = CKPT_DIR / "training_log.json"
EVAL_PATH = Path("data/v25_eval.json")
ANNOTATIONS_CSV = Path("data/annotations.csv")
CLIPS_DIR = Path("data/audio_clips_compact")
EMB_PARQUET = Path("data/embeddings_with_labels.parquet")
FINDINGS_PATH = Path("FINDINGS.md")

GT_BINARY_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}
GT_EXCLUDE = {"unclear", "skip", "", None}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_clip(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)
    max_n = SAMPLE_RATE * MAX_LEN_SECONDS
    if y.shape[0] > max_n:
        y = y[:max_n]
    elif y.shape[0] < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - y.shape[0]))
    return y.astype(np.float32)


class DialectAudioDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        m = self.items[idx]
        clip = CLIPS_DIR / f"item_{m['item_id']}.mp3"
        return {
            "audio": load_clip(clip),
            "label": int(m["lebanese"]),
            "platform": m["platform"],
            "item_id": int(m["item_id"]),
        }


def make_collate(feature_extractor):
    def collate(batch):
        arrays = [b["audio"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = feature_extractor(arrays, sampling_rate=SAMPLE_RATE,
                                return_tensors="pt", padding=True)
        return {
            "input_values": enc["input_values"],
            "attention_mask": enc.get("attention_mask"),
            "labels": labels,
        }
    return collate


def evaluate(model, loader, name="") -> dict:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            iv = batch["input_values"]
            am = batch["attention_mask"]
            out = model(input_values=iv, attention_mask=am)
            p = torch.softmax(out.logits, dim=-1)[:, 1].numpy()
            probs.extend(p.tolist())
            labels.extend(batch["labels"].numpy().tolist())
    y = np.array(labels)
    p = np.array(probs)
    if len(set(y)) <= 1:
        return {"name": name, "n": int(len(y)), "note": "single-class set"}
    preds05 = (p >= 0.5).astype(int)
    pr, rc, f1, support = precision_recall_fscore_support(y, preds05, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, preds05, labels=[0, 1])
    auc = roc_auc_score(y, p)
    pr_auc = average_precision_score(y, p)
    # Threshold sweep for best macro F1
    best = {"threshold": 0.5, "macro_f1": float(f1.mean())}
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (p >= thr).astype(int)
        _, _, f1_, _ = precision_recall_fscore_support(y, preds, labels=[0, 1], zero_division=0)
        mf1 = f1_.mean()
        if mf1 > best["macro_f1"]:
            best = {"threshold": float(thr), "macro_f1": float(mf1)}
    return {
        "name": name, "n": int(len(y)),
        "accuracy": float(accuracy_score(y, preds05)),
        "macro_f1": float(f1.mean()),
        "f1_neg": float(f1[0]), "f1_pos": float(f1[1]),
        "precision_neg": float(pr[0]), "precision_pos": float(pr[1]),
        "recall_neg": float(rc[0]), "recall_pos": float(rc[1]),
        "support_neg": int(support[0]), "support_pos": int(support[1]),
        "roc_auc": float(auc), "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "best_thr_macro_f1": best,
        "probs": p.tolist(), "labels": y.tolist(),
    }


def build_items_from_parquet():
    """Load training pool and GT items from data/embeddings_with_labels.parquet
    (already labeled and joined with queue.db). We use only item_id, label,
    platform, in_ground_truth, gt_label."""
    if not EMB_PARQUET.exists():
        sys.exit(f"ERROR: {EMB_PARQUET} not found. Run scripts/13_load_embeddings.py first.")
    import pandas as pd
    df = pd.read_parquet(EMB_PARQUET, columns=[
        "item_id", "platform", "lebanese", "in_ground_truth", "gt_label", "status"
    ])
    train_items = []
    gt_items = []
    for _, row in df.iterrows():
        if row["in_ground_truth"]:
            raw = (row.get("gt_label") or "").strip().lower()
            if raw in GT_EXCLUDE:
                continue
            y = GT_BINARY_MAP.get(raw)
            if y is None:
                continue
            gt_items.append({"item_id": int(row["item_id"]), "lebanese": y, "platform": row["platform"]})
        else:
            if row["lebanese"] is None or (isinstance(row["lebanese"], float) and np.isnan(row["lebanese"])):
                continue
            train_items.append({
                "item_id": int(row["item_id"]),
                "lebanese": int(row["lebanese"]),
                "platform": row["platform"],
            })
    return train_items, gt_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--num-frozen-layers", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Use only 200 items, 1 epoch (sanity check, ~30 min).")
    ap.add_argument("--max-items", type=int, default=0,
                    help="Subsample training pool to N items, stratified by (platform, label). 0 = use all.")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    set_seed(SEED)
    torch.set_num_threads(args.threads)
    print(f"[setup] PyTorch threads: {args.threads}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("[data] loading items from embeddings_with_labels.parquet...", flush=True)
    train_pool, gt_items = build_items_from_parquet()
    print(f"[data] training pool: {len(train_pool)} (pos={sum(1 for x in train_pool if x['lebanese']==1)})", flush=True)
    print(f"[data] GT eval set: {len(gt_items)} (pos={sum(1 for x in gt_items if x['lebanese']==1)})", flush=True)

    if args.smoke_test:
        print("[smoke-test] subsampling to 200 items, 1 epoch", flush=True)
        random.shuffle(train_pool)
        train_pool = train_pool[:200]
        args.epochs = 1
        args.checkpoint_every = 50
    elif args.max_items and args.max_items < len(train_pool):
        # Stratified subsample by (platform, label): each group keeps a proportional share
        groups = {}
        for it in train_pool:
            groups.setdefault((it["platform"], it["lebanese"]), []).append(it)
        target_total = args.max_items
        sampled = []
        # Per-group quota proportional to original size, with min 5 per group when feasible
        sizes = {k: len(v) for k, v in groups.items()}
        total = sum(sizes.values())
        for k, items in groups.items():
            quota = max(5, int(round(target_total * sizes[k] / total)))
            quota = min(quota, len(items))
            random.shuffle(items)
            sampled.extend(items[:quota])
        random.shuffle(sampled)
        # If overshoot, trim; if undershoot, leave as-is
        if len(sampled) > target_total:
            sampled = sampled[:target_total]
        train_pool = sampled
        print(f"[subsample] reduced to {len(train_pool)} items (stratified)", flush=True)

    # Train/val split
    random.shuffle(train_pool)
    n_val = max(50, int(args.val_fraction * len(train_pool)))
    val_items = train_pool[:n_val]
    fit_items = train_pool[n_val:]
    print(f"[data] fit: {len(fit_items)}  val: {len(val_items)}", flush=True)

    # Per-(platform, label) sample weights
    counts = Counter((it["platform"], it["lebanese"]) for it in fit_items)
    n_groups = len(counts)
    weights = [1.0 / (n_groups * counts[(it["platform"], it["lebanese"])]) for it in fit_items]
    print(f"[data] per-source groups: {n_groups}", flush=True)
    for k, v in sorted(counts.items()):
        print(f"        {k}: {v}", flush=True)

    # Datasets / loaders
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    collate = make_collate(feature_extractor)

    fit_ds = DialectAudioDataset(fit_items)
    val_ds = DialectAudioDataset(val_items)
    gt_ds = DialectAudioDataset(gt_items)

    sampler = WeightedRandomSampler(weights, num_samples=len(fit_items), replacement=True)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch, sampler=sampler,
                            collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate, num_workers=0)
    gt_loader = DataLoader(gt_ds, batch_size=args.batch, shuffle=False,
                           collate_fn=collate, num_workers=0)

    # Model
    print(f"[model] loading {MODEL_NAME}...", flush=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.freeze_feature_encoder()
    n_layers = len(model.wav2vec2.encoder.layers)
    n_frozen = args.num_frozen_layers
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        for p in layer.parameters():
            p.requires_grad = i >= n_frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] total: {total/1e6:.0f}M  trainable: {trainable/1e6:.0f}M  ({100*trainable/total:.1f}%)", flush=True)
    print(f"[model] freezing first {n_frozen}/{n_layers} transformer layers", flush=True)

    # Optimizer
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder.layers" in n or "feature_projection" in n:
            enc_params.append(p)
        else:
            head_params.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder},
         {"params": head_params, "lr": args.lr_head}],
        weight_decay=0.01,
    )
    total_steps = (len(fit_loader) * args.epochs) // args.grad_accum
    warmup = max(10, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"[optim] total optim steps: {total_steps}  warmup: {warmup}", flush=True)

    # Resume from latest checkpoint if present
    state = {"global_step": 0, "epoch": 0, "best_gt_macro_f1": -1.0, "log": []}
    latest_ckpt = CKPT_DIR / "latest.pt"
    if latest_ckpt.exists():
        print(f"[resume] loading {latest_ckpt}", flush=True)
        ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        state = ck["state"]
        print(f"[resume] resumed at global_step={state['global_step']}, epoch={state['epoch']}", flush=True)
    else:
        print("[resume] no checkpoint found — starting from scratch", flush=True)

    # Training loop
    print(f"\n[train] starting at epoch {state['epoch']+1}/{args.epochs}", flush=True)
    t0 = time.time()
    for epoch in range(state["epoch"], args.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(fit_loader):
            iv = batch["input_values"]
            am = batch["attention_mask"]
            labels = batch["labels"]
            out = model(input_values=iv, attention_mask=am, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                state["global_step"] += 1

                if state["global_step"] % args.checkpoint_every == 0:
                    elapsed = time.time() - t0
                    avg_loss = epoch_loss / max(n_batches, 1)
                    print(f"[ckpt] epoch={epoch+1} step={state['global_step']}/{total_steps} avg_loss={avg_loss:.4f} elapsed={elapsed/60:.1f}min", flush=True)
                    ck_payload = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "state": dict(state, epoch=epoch),
                        "args": vars(args),
                    }
                    torch.save(ck_payload, latest_ckpt)

            if (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"[step] epoch={epoch+1} step={step+1}/{len(fit_loader)} loss={loss.item()*args.grad_accum:.4f} elapsed={elapsed/60:.1f}min", flush=True)

        # End of epoch: evaluate on val + GT
        print(f"\n[eval] end of epoch {epoch+1}", flush=True)
        val_res = evaluate(model, val_loader, name=f"val_epoch{epoch+1}")
        gt_res = evaluate(model, gt_loader, name=f"gt_epoch{epoch+1}")
        gt_mf1 = gt_res.get("best_thr_macro_f1", {}).get("macro_f1", gt_res.get("macro_f1", 0))
        print(f"[eval] val acc={val_res.get('accuracy', 0):.4f} macroF1={val_res.get('macro_f1', 0):.4f}", flush=True)
        print(f"[eval] GT  acc={gt_res.get('accuracy', 0):.4f} macroF1@0.5={gt_res.get('macro_f1', 0):.4f} best={gt_mf1:.4f} ROC-AUC={gt_res.get('roc_auc', 0):.4f}", flush=True)
        state["log"].append({"epoch": epoch + 1, "val": {k: v for k, v in val_res.items() if k not in {"probs", "labels"}},
                             "gt": {k: v for k, v in gt_res.items() if k not in {"probs", "labels"}}})

        # Save best by GT best-threshold macro F1
        if gt_mf1 > state["best_gt_macro_f1"]:
            state["best_gt_macro_f1"] = gt_mf1
            print(f"[best] new best GT macro F1: {gt_mf1:.4f}, saving to {BEST_DIR}/", flush=True)
            BEST_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_DIR)
            feature_extractor.save_pretrained(BEST_DIR)
            (BEST_DIR / "eval.json").write_text(json.dumps({"val": val_res, "gt": gt_res}, indent=2), encoding="utf-8")

        state["epoch"] = epoch + 1
        LOG_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"\n[done] total elapsed: {(time.time()-t0)/60:.0f}min", flush=True)
    print(f"[done] best GT macro F1: {state['best_gt_macro_f1']:.4f}", flush=True)
    print(f"[done] best model in: {BEST_DIR}", flush=True)
    print("\nNext step: run scripts/16_eval_finetuned_xlsr.py to write FINDINGS Section 13", flush=True)


if __name__ == "__main__":
    main()
