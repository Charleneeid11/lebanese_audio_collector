"""
Colab notebook script: end-to-end fine-tune wav2vec2-xls-r-300m for binary
Lebanese-vs-other Arabic dialect classification.

Run in Google Colab on T4 GPU. Paste this whole file in one cell, uncomment the
!pip / !apt-get setup, run.

Why this script exists: Section 12.6 of FINDINGS.md showed that frozen-encoder
acoustic embeddings (V2) failed because the encoder never adjusts its features
toward dialect identity. End-to-end fine-tuning lets the encoder weights update
under cross-entropy supervision; the model can learn dialect features that
survive the per-source class-imbalance shortcut.

Strategy:
  - Wav2Vec2ForSequenceClassification head on top of facebook/wav2vec2-xls-r-300m
  - Freeze first 18 of 24 encoder layers; train top 6 layers + classifier head
    (keeps ~75M trainable params, fits T4 memory; preserves general
    multilingual representations in lower layers)
  - Per-(platform, label) WeightedRandomSampler to prevent recording-domain
    shortcut learning (the same balancing that broke frozen-V2 will work HERE
    because end-to-end training can adjust the features themselves)
  - 3 epochs, AdamW with linear warmup, lr=5e-5 head / 1e-5 encoder
  - Save fine-tuned model + Wav2Vec2FeatureExtractor for reproducibility
  - Evaluate on held-out 300 GT items (item-id matched against manifest)

Audio source: GitHub release (already public)
  https://github.com/Charleneeid116/thesis-data/releases/download/v0.1-embeddings-input/audio_for_embeddings_compact.zip

Output (downloaded back from Colab): models/xlsr_finetuned.zip containing:
  - pytorch_model.bin (or safetensors)
  - config.json
  - preprocessor_config.json
  - training_log.json
  - eval_results.json
"""

# ============================================================
# Cell 1: setup + auth
# ============================================================
# !pip install -q -U transformers torch torchaudio soundfile librosa pandas pyarrow scikit-learn huggingface_hub
# !apt-get -qq install -y ffmpeg

import io
import json
import os
import random
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

# HF auth (FLEURS-era token still valid)
try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
    if _tok:
        os.environ["HF_TOKEN"] = _tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
        from huggingface_hub import login
        login(token=_tok, add_to_git_credential=False)
        print("HF auth: token loaded.", flush=True)
except Exception as _e:
    print(f"HF auth: skipped ({_e}).", flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import librosa


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cpu":
    raise SystemExit(
        "GPU required. Runtime > Change runtime type > T4 GPU."
    )

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
MAX_LEN_SECONDS = 10
NUM_LABELS = 2
SEED = 42

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4   # effective batch = 32
NUM_EPOCHS = 3
LR_HEAD = 5e-5
LR_ENCODER = 1e-5
WARMUP_RATIO = 0.1

WORK = Path("/content/finetune_work")
WORK.mkdir(exist_ok=True)
INPUT_ZIP = Path("/content/audio_for_embeddings_compact.zip")
EXTRACT_DIR = WORK / "extracted"
OUT_DIR = Path("/content/xlsr_finetuned")
OUT_DIR.mkdir(exist_ok=True)
OUT_ZIP = Path("/content/xlsr_finetuned.zip")

ASSET_URL = (
    "https://github.com/Charleneeid116/thesis-data/releases/download/"
    "v0.1-embeddings-input/audio_for_embeddings_compact.zip"
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Cell 2: download + extract audio
# ============================================================
if not INPUT_ZIP.exists():
    print(f"Fetching {ASSET_URL}...", flush=True)
    import requests
    with requests.get(ASSET_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(INPUT_ZIP, "wb") as f:
            got = 0
            next_report = 100 * 1024 * 1024
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
                if got >= next_report:
                    pct = (100 * got / total) if total else 0
                    print(f"  {got/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)", flush=True)
                    next_report += 100 * 1024 * 1024
    print(f"  done: {INPUT_ZIP.stat().st_size/1e6:.0f} MB", flush=True)

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True)
print(f"Extracting {INPUT_ZIP}...", flush=True)
with zipfile.ZipFile(INPUT_ZIP) as z:
    z.extractall(EXTRACT_DIR)

manifest_path = EXTRACT_DIR / "manifest.json"
audio_dir = EXTRACT_DIR / "audio"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(f"manifest items: {len(manifest)}", flush=True)


# ============================================================
# Cell 3: build labels (mirrors scripts/13_load_embeddings.py)
# ============================================================
NEGATIVE_ADI17_DIALECTS = {"EGY", "KSA", "KUW", "UAE", "QAT", "OMA"}

def derive_label(m: dict) -> int | None:
    status = m.get("status")
    platform = m.get("platform")
    adi17_dialect = m.get("adi17_dialect")
    if status == "POTENTIAL_LB":
        return 1
    if status == "WEAK_POSITIVE":
        return 1
    if platform == "adi17" and adi17_dialect == "LEB":
        return 1
    if status == "WEAK_NEGATIVE":
        return 0
    if platform == "adi17" and adi17_dialect in NEGATIVE_ADI17_DIALECTS:
        return 0
    if platform == "fleurs":
        return 0
    return None


GT_BINARY_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}

train_items = []
gt_items = []
skipped = 0
for m in manifest:
    gt_label_raw = (m.get("gt_label") or "").strip().lower()
    if gt_label_raw in {"lebanese", "mostly_lebanese", "not_lebanese"}:
        gt_items.append({**m, "y": GT_BINARY_MAP[gt_label_raw]})
        continue
    if gt_label_raw in {"unclear", "skip"}:
        # Excluded from both train and eval (matches v1/v2 protocol)
        skipped += 1
        continue
    y = derive_label(m)
    if y is None:
        skipped += 1
        continue
    train_items.append({**m, "y": y})

print(f"\ntrain pool: {len(train_items)}  pos={sum(1 for x in train_items if x['y']==1)}  neg={sum(1 for x in train_items if x['y']==0)}", flush=True)
print(f"GT eval set: {len(gt_items)}  pos={sum(1 for x in gt_items if x['y']==1)}  neg={sum(1 for x in gt_items if x['y']==0)}", flush=True)
print(f"skipped: {skipped}", flush=True)


# ============================================================
# Cell 4: stratified train/val split + per-(platform,label) weights
# ============================================================
set_seed(SEED)
random.shuffle(train_items)
val_size = int(0.1 * len(train_items))   # 10% val (smaller than 20% to give more train)
val_items = train_items[:val_size]
fit_items = train_items[val_size:]
print(f"\nfit: {len(fit_items)}  val: {len(val_items)}", flush=True)

group_counts = Counter((it["platform"], it["y"]) for it in fit_items)
print("(platform, label) groups in fit set:", flush=True)
for k, v in sorted(group_counts.items()):
    print(f"  {k}: {v}", flush=True)

n_groups = len(group_counts)
fit_weights = [1.0 / (n_groups * group_counts[(it["platform"], it["y"])]) for it in fit_items]


# ============================================================
# Cell 5: Dataset + DataLoader
# ============================================================
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)


def load_clip_to_array(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)
    max_n = SAMPLE_RATE * MAX_LEN_SECONDS
    if y.shape[0] > max_n:
        y = y[:max_n]
    elif y.shape[0] < SAMPLE_RATE:        # < 1 sec — pad with zeros
        y = np.pad(y, (0, SAMPLE_RATE - y.shape[0]))
    return y.astype(np.float32)


class DialectDataset(Dataset):
    def __init__(self, items, audio_dir):
        self.items = items
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        m = self.items[idx]
        path = self.audio_dir / m["audio_filename"]
        arr = load_clip_to_array(path)
        return {"audio": arr, "label": m["y"], "platform": m["platform"], "item_id": m["item_id"]}


def collate(batch):
    arrays = [b["audio"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = feature_extractor(arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    return {
        "input_values": enc["input_values"],
        "attention_mask": enc.get("attention_mask"),
        "labels": labels,
    }


fit_ds = DialectDataset(fit_items, audio_dir)
val_ds = DialectDataset(val_items, audio_dir)
gt_ds = DialectDataset(gt_items, audio_dir)

sampler = WeightedRandomSampler(fit_weights, num_samples=len(fit_items), replacement=True)
fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)
gt_loader = DataLoader(gt_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)

print(f"\nDataLoaders ready. Sampler will yield {len(fit_items)} examples per epoch.", flush=True)


# ============================================================
# Cell 6: model — partial unfreezing
# ============================================================
print(f"\nLoading {MODEL_NAME} with classification head...", flush=True)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)
model = model.to(DEVICE)

# Freeze the feature extractor (CNN frontend) entirely — these are very general.
model.freeze_feature_encoder()

# Freeze first N transformer layers; train the rest + classifier head.
N_LAYERS = len(model.wav2vec2.encoder.layers)
N_FROZEN = 18                            # train top (24-18) = 6 layers
for i, layer in enumerate(model.wav2vec2.encoder.layers):
    requires_grad = i >= N_FROZEN
    for p in layer.parameters():
        p.requires_grad = requires_grad

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  total params: {total/1e6:.0f}M  trainable: {trainable/1e6:.0f}M  ({100*trainable/total:.1f}%)", flush=True)

# Two parameter groups: encoder layers (low LR) and head/projector (high LR)
encoder_params = []
head_params = []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "encoder.layers" in n or "feature_projection" in n:
        encoder_params.append(p)
    else:
        head_params.append(p)
optimizer = torch.optim.AdamW(
    [
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD},
    ],
    weight_decay=0.01,
)
total_steps = (len(fit_loader) * NUM_EPOCHS) // GRAD_ACCUM_STEPS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
print(f"  total steps: {total_steps}  warmup: {warmup_steps}", flush=True)


# ============================================================
# Cell 7: train + eval
# ============================================================
def evaluate(model, loader, name=""):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            iv = batch["input_values"].to(DEVICE)
            am = batch["attention_mask"].to(DEVICE) if batch["attention_mask"] is not None else None
            out = model(input_values=iv, attention_mask=am)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())
    y = np.array(all_labels)
    p = np.array(all_probs)
    preds = (p >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y, preds, labels=[0, 1], zero_division=0)
    macro_f1 = f1.mean()
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else None
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return {
        "name": name,
        "n": int(len(y)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1_neg": float(f1[0]), "f1_pos": float(f1[1]),
        "precision_neg": float(pr[0]), "precision_pos": float(pr[1]),
        "recall_neg": float(rc[0]), "recall_pos": float(rc[1]),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "probs": p.tolist(),
        "labels": y.tolist(),
    }


training_log = []
print("\nStarting training...", flush=True)
t_start = time.time()

global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0.0
    n_batches = 0
    for step, batch in enumerate(fit_loader):
        iv = batch["input_values"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE) if batch["attention_mask"] is not None else None
        labels = batch["labels"].to(DEVICE)

        out = model(input_values=iv, attention_mask=am, labels=labels)
        loss = out.loss / GRAD_ACCUM_STEPS
        loss.backward()
        epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        n_batches += 1

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  epoch {epoch+1}/{NUM_EPOCHS} step {step+1}/{len(fit_loader)} loss={loss.item()*GRAD_ACCUM_STEPS:.4f} elapsed={elapsed/60:.1f}min", flush=True)

    avg_loss = epoch_loss / max(n_batches, 1)
    print(f"\n[epoch {epoch+1}] avg loss: {avg_loss:.4f}", flush=True)

    print("  evaluating on val...", flush=True)
    val_res = evaluate(model, val_loader, name=f"val_epoch{epoch+1}")
    print(f"    val acc={val_res['accuracy']:.4f} macroF1={val_res['macro_f1']:.4f} ROC-AUC={val_res['roc_auc']}", flush=True)

    print("  evaluating on GT...", flush=True)
    gt_res = evaluate(model, gt_loader, name=f"gt_epoch{epoch+1}")
    print(f"    GT  acc={gt_res['accuracy']:.4f} macroF1={gt_res['macro_f1']:.4f} ROC-AUC={gt_res['roc_auc']}", flush=True)

    training_log.append({
        "epoch": epoch + 1,
        "avg_train_loss": avg_loss,
        "val": val_res,
        "gt": gt_res,
    })

elapsed = time.time() - t_start
print(f"\nTraining done in {elapsed/60:.0f}min.", flush=True)


# ============================================================
# Cell 8: save model + results
# ============================================================
print(f"\nSaving model to {OUT_DIR}...", flush=True)
model.save_pretrained(OUT_DIR)
feature_extractor.save_pretrained(OUT_DIR)

eval_results = {
    "model_name": MODEL_NAME,
    "config": {
        "num_labels": NUM_LABELS,
        "max_len_seconds": MAX_LEN_SECONDS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "num_epochs": NUM_EPOCHS,
        "lr_head": LR_HEAD,
        "lr_encoder": LR_ENCODER,
        "warmup_ratio": WARMUP_RATIO,
        "n_frozen_layers": N_FROZEN,
        "n_trainable_params_M": trainable / 1e6,
        "per_source_balanced": True,
    },
    "training_log": training_log,
    "final_val": training_log[-1]["val"] if training_log else None,
    "final_gt": training_log[-1]["gt"] if training_log else None,
    "elapsed_seconds": round(elapsed, 1),
}
(OUT_DIR / "eval_results.json").write_text(json.dumps(eval_results, indent=2), encoding="utf-8")

# Best epoch by GT macro F1
if training_log:
    best = max(training_log, key=lambda r: r["gt"]["macro_f1"])
    print(f"\nBest epoch by GT macro F1: {best['epoch']}  GT macroF1={best['gt']['macro_f1']:.4f}  ROC-AUC={best['gt']['roc_auc']}", flush=True)

print(f"\nZipping model + results to {OUT_ZIP}...", flush=True)
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in OUT_DIR.iterdir():
        z.write(f, arcname=f.name)
print(f"DONE. {OUT_ZIP} size = {OUT_ZIP.stat().st_size/1e6:.0f} MB.", flush=True)
print("\nDownload xlsr_finetuned.zip from Colab's file panel and unpack into models/xlsr_finetuned/ locally.")
print("Then a follow-up local script can re-run the GT eval and append FINDINGS Section 13 (V2.5 fine-tuned).")
