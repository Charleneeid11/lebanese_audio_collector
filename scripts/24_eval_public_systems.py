#!/usr/bin/env python3
"""
Evaluate 4 publicly released Arabic dialect ID systems on the Lebanese GT.

Systems:
  1. badrex/mms-300m-arabic-dialect-identifier        (Badr 2025, 5-class incl. Levantine)
  2. tiantiaf/voxlect-arabic-dialect-mms-lid-256      (Voxlect 2026, 5-class incl. Levantine)
  3. Elyadata/ADI-whisper-ADI20                       (Elleuch 2025, 20-class incl. LEB)
  4. IbrahimAmin/marbertv2-arabic-written-dialect-classifier (text, 5-class incl. LEV)

Mapping to Lebanese binary:
  - Systems exposing 'LEB' (country-level): P(LEB) directly
  - Systems exposing 'Levantine'/'LEV' (region-level): P(Levantine) as proxy
    => Note: this is a generous upper bound; Levantine includes LB, SY, JO, PA

Per-system pause/resume:
  - Each system writes data/benchmark_predictions/<name>.json atomically on completion.
  - If interrupted mid-system, that system is re-run from scratch on next invocation.
  - Already-complete systems are skipped (loaded from disk).

Usage:
  python scripts/24_eval_public_systems.py
  python scripts/24_eval_public_systems.py --only badr_mms_300m_levantine
  python scripts/24_eval_public_systems.py --skip voxlect_mms_lid256_levantine --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_harness_utils import (
    PREDS_DIR, RESULTS_CSV, compute_metrics, load_gt_items, load_predictions,
    save_predictions, upsert_csv_row,
)

CLIPS_DIR = Path("data/audio_clips_compact")
FINDINGS_PATH = Path("FINDINGS.md")
SAMPLE_RATE = 16000
MAX_LEN_SECONDS = 10


# ---------------------------------------------------------------------------
# Per-system predict implementations
# ---------------------------------------------------------------------------

def _load_audio(item_id: int) -> "np.ndarray | None":
    import librosa
    clip = CLIPS_DIR / f"item_{item_id}.mp3"
    if not clip.exists():
        return None
    y, _ = librosa.load(str(clip), sr=SAMPLE_RATE, mono=True)
    if y.shape[0] > SAMPLE_RATE * MAX_LEN_SECONDS:
        y = y[: SAMPLE_RATE * MAX_LEN_SECONDS]
    return y


def predict_audio_classification(model_id: str, lebanese_label: str,
                                  items: list[dict], system_name: str) -> dict[int, float]:
    """Generic audio classifier inference via AutoModelForAudioClassification."""
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print(f"  [load] {model_id}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    id2label = model.config.id2label
    labels_list = [id2label[i] for i in sorted(id2label.keys())]
    print(f"  [labels] {labels_list}", flush=True)

    # Find the target label (case-insensitive match)
    target_idx = None
    for i, lbl in enumerate(labels_list):
        if lbl.strip().lower() == lebanese_label.strip().lower():
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError(f"Label '{lebanese_label}' not found in {labels_list}")
    print(f"  [target] '{lebanese_label}' is class index {target_idx}", flush=True)

    probs: dict[int, float] = {}
    n = len(items)
    for i, it in enumerate(items):
        iid = it["item_id"]
        audio = _load_audio(iid)
        if audio is None:
            continue
        try:
            inputs = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            p_full = torch.softmax(logits, dim=-1)[0].numpy()
            probs[iid] = float(p_full[target_idx])
        except Exception as e:
            print(f"    [warn] item {iid} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if (i + 1) % 25 == 0:
            print(f"  [{system_name}] {i+1}/{n}", flush=True)
    return probs


def predict_voxlect_mms(model_id: str, lebanese_label: str,
                         items: list[dict], system_name: str) -> dict[int, float]:
    """Voxlect MMS-LID-256 (Feng et al. 2025) — custom MMSWrapper class via PyTorchModelHubMixin."""
    import torch
    import torch.nn.functional as F
    from scripts._vendored.voxlect_mms_dialect import MMSWrapper

    print(f"  [load] {model_id} (Voxlect MMSWrapper)", flush=True)
    # HF CDN occasionally drops the connection mid-download. The cache .incomplete
    # files are resumable, so retry up to 5 times before giving up.
    import time as _time
    last_err = None
    for attempt in range(5):
        try:
            model = MMSWrapper.from_pretrained(model_id)
            break
        except Exception as e:  # noqa: BLE001 — we want to catch network errors broadly
            last_err = e
            print(f"  [retry {attempt+1}/5] from_pretrained failed: {type(e).__name__}: {e}", flush=True)
            _time.sleep(5 * (attempt + 1))
    else:
        raise last_err
    model.eval()

    # README label order:
    labels_list = ["Egyptian", "Levantine", "Maghrebi", "MSA", "Peninsular"]
    print(f"  [labels] {labels_list}", flush=True)
    target_idx = labels_list.index(lebanese_label)
    print(f"  [target] '{lebanese_label}' is class index {target_idx}", flush=True)

    probs: dict[int, float] = {}
    n = len(items)
    max_len = 15 * SAMPLE_RATE  # Voxlect README: max 15s, 16 kHz mono
    for i, it in enumerate(items):
        iid = it["item_id"]
        audio = _load_audio(iid)
        if audio is None:
            continue
        try:
            if audio.shape[0] > max_len:
                audio = audio[:max_len]
            x = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
            with torch.no_grad():
                logits = model(x)
            p_full = F.softmax(logits, dim=-1)[0].numpy()
            probs[iid] = float(p_full[target_idx])
        except Exception as e:
            print(f"    [warn] item {iid} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if (i + 1) % 10 == 0:
            print(f"  [{system_name}] {i+1}/{n}", flush=True)
    return probs


def predict_elyadata_speechbrain(model_id: str, lebanese_label: str,
                                  items: list[dict], system_name: str) -> dict[int, float]:
    """Elyadata ADI-whisper-ADI20 (Elleuch et al. 2025) — SpeechBrain Pretrained checkpoint.

    SpeechBrain's fetcher symlinks files from the HF cache into ``savedir``. On Windows
    that requires either admin rights or Developer Mode (WinError 1314). To keep this
    portable, we monkey-patch ``os.symlink`` to fall back to a real file copy on failure,
    only for the duration of this function.
    """
    import os
    import shutil
    import torch
    from scripts._vendored.elyadata_classifier_attention_pooling import WhisperDialectClassifier

    _orig_symlink = os.symlink
    def _symlink_with_copy_fallback(src, dst, target_is_directory=False):
        try:
            _orig_symlink(src, dst, target_is_directory=target_is_directory)
        except (OSError, NotImplementedError):
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    os.symlink = _symlink_with_copy_fallback

    try:
        print(f"  [load] {model_id} (SpeechBrain WhisperDialectClassifier — downloads whisper-large-v3 on first run)", flush=True)
        savedir = Path("models/_pretrained_elyadata_did")
        savedir.mkdir(parents=True, exist_ok=True)
        classifier = WhisperDialectClassifier.from_hparams(
            source=model_id,
            hparams_file="hyperparams.yaml",
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
    finally:
        os.symlink = _orig_symlink
    classifier.hparams.label_encoder.expect_len(classifier.hparams.n_languages)

    # Resolve target index from the label encoder's lab2ind dict.
    lab2ind = classifier.hparams.label_encoder.lab2ind
    labels_list = sorted(lab2ind.items(), key=lambda kv: kv[1])
    labels_list = [k for k, _ in labels_list]
    print(f"  [labels] {labels_list}", flush=True)
    target_idx = lab2ind[lebanese_label]
    print(f"  [target] '{lebanese_label}' is class index {target_idx}", flush=True)

    probs: dict[int, float] = {}
    n = len(items)
    for i, it in enumerate(items):
        iid = it["item_id"]
        audio = _load_audio(iid)
        if audio is None:
            continue
        try:
            wav = torch.from_numpy(audio).float().unsqueeze(0)
            wav_lens = torch.tensor([1.0])
            with torch.no_grad():
                out_log_prob, _, _, _ = classifier.classify_batch(wav, wav_lens)
            p_full = torch.exp(out_log_prob)[0].numpy()
            probs[iid] = float(p_full[target_idx])
        except Exception as e:
            print(f"    [warn] item {iid} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if (i + 1) % 5 == 0:
            print(f"  [{system_name}] {i+1}/{n}", flush=True)
    return probs


def predict_text_classification(model_id: str, lebanese_label: str,
                                 items: list[dict], system_name: str) -> dict[int, float]:
    """Generic text classifier inference via AutoModelForSequenceClassification."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.cfg import Settings

    settings = Settings.load()
    transcripts_dir = Path(settings.transcription.transcripts_dir)

    print(f"  [load] {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    id2label = model.config.id2label
    labels_list = [id2label[i] for i in sorted(id2label.keys())]
    print(f"  [labels] {labels_list}", flush=True)

    target_idx = None
    for i, lbl in enumerate(labels_list):
        if lbl.strip().lower() == lebanese_label.strip().lower():
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError(f"Label '{lebanese_label}' not found in {labels_list}")
    print(f"  [target] '{lebanese_label}' is class index {target_idx}", flush=True)

    probs: dict[int, float] = {}
    n = len(items)
    for i, it in enumerate(items):
        iid = it["item_id"]
        tpath = transcripts_dir / f"clip_{iid}_screening.json"
        if not tpath.exists():
            continue
        try:
            data = json.loads(tpath.read_text(encoding="utf-8"))
            text = " ".join(s.get("text", "") for s in (data.get("screening_samples") or []))
            if not text.strip():
                continue
            enc = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                logits = model(**enc).logits
            p_full = torch.softmax(logits, dim=-1)[0].numpy()
            probs[iid] = float(p_full[target_idx])
        except Exception as e:
            print(f"    [warn] item {iid} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if (i + 1) % 50 == 0:
            print(f"  [{system_name}] {i+1}/{n}", flush=True)
    return probs


# ---------------------------------------------------------------------------
# Systems registry
# ---------------------------------------------------------------------------

SYSTEMS = {
    "badr_mms_300m_levantine": {
        "model_id": "badrex/mms-300m-arabic-dialect-identifier",
        "family": "acoustic",
        "modality": "audio",
        "loader": "hf_audio_classification",
        "lebanese_label": "Levantine",
        "label_granularity": "regional (Levantine = LB+SY+JO+PA)",
        "citation": "Badr et al. 2025 INTERSPEECH",
    },
    "voxlect_mms_lid256_levantine": {
        "model_id": "tiantiaf/voxlect-arabic-dialect-mms-lid-256",
        "family": "acoustic",
        "modality": "audio",
        "loader": "voxlect_mms",
        "lebanese_label": "Levantine",
        "label_granularity": "regional (Levantine = LB+SY+JO+PA)",
        "citation": "Feng et al. 2026 KDD (Voxlect)",
    },
    "elyadata_whisper_adi20_leb": {
        "model_id": "Elyadata/ADI-whisper-ADI20",
        "family": "acoustic",
        "modality": "audio",
        "loader": "elyadata_speechbrain",
        "lebanese_label": "LEB",
        "label_granularity": "country-level",
        "citation": "Elleuch et al. 2025 INTERSPEECH",
    },
    "marbertv2_lev": {
        "model_id": "IbrahimAmin/marbertv2-arabic-written-dialect-classifier",
        "family": "lexical",
        "modality": "text",
        "loader": "hf_text_classification",
        "lebanese_label": "LEV",
        "label_granularity": "regional (Levantine = LB+SY+JO+PA)",
        "citation": "MARBERTv2 / Abdul-Mageed et al. lineage",
    },
}


LOADERS = {
    "hf_audio_classification": predict_audio_classification,
    "hf_text_classification": predict_text_classification,
    "voxlect_mms": predict_voxlect_mms,
    "elyadata_speechbrain": predict_elyadata_speechbrain,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--skip", action="append", default=[])
    args = ap.parse_args()

    items = load_gt_items()
    print(f"[gt] loaded {len(items)} GT items "
          f"(pos={sum(1 for it in items if it['y']==1)}, "
          f"neg={sum(1 for it in items if it['y']==0)})", flush=True)

    selected = list(SYSTEMS.items())
    if args.only:
        selected = [(n, v) for n, v in selected if n in args.only]
    if args.skip:
        selected = [(n, v) for n, v in selected if n not in args.skip]
    print(f"[plan] {len(selected)} system(s): {[n for n, _ in selected]}", flush=True)

    for system_name, cfg in selected:
        preds_path = PREDS_DIR / f"{system_name}.json"
        t0 = time.time()

        if preds_path.exists() and not args.force:
            print(f"\n[skip ] {system_name} - predictions on disk, loading", flush=True)
            probs = load_predictions(preds_path)
        else:
            print(f"\n[start] {system_name} ({cfg['family']} / {cfg['modality']} / {cfg['loader']}) - {cfg['model_id']}", flush=True)
            try:
                loader_fn = LOADERS.get(cfg["loader"])
                if loader_fn is None:
                    print(f"  [error] unknown loader {cfg['loader']}", flush=True)
                    continue
                probs = loader_fn(
                    cfg["model_id"], cfg["lebanese_label"], items, system_name
                )
            except Exception as e:
                print(f"[error] {system_name} crashed: {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue
            save_predictions(preds_path, system_name, items, probs)
            print(f"[save ] {preds_path}  ({len(probs)}/{len(items)} predictions, "
                  f"{time.time()-t0:.1f}s)", flush=True)

        row = compute_metrics(system_name, cfg["family"], items, probs)
        upsert_csv_row(RESULTS_CSV, row)
        print(
            f"[done ] {system_name}  n={row.get('n_evaluated')}  "
            f"macroF1@0.5={row.get('macro_f1_at_0.5'):.4f} "
            f"(CI95 {row.get('macro_f1_ci95_lo'):.3f}-{row.get('macro_f1_ci95_hi'):.3f})  "
            f"best={row.get('macro_f1_at_best'):.4f}@thr={row.get('best_threshold'):.2f}  "
            f"ROC-AUC={row.get('roc_auc'):.4f} "
            f"(CI95 {row.get('roc_auc_ci95_lo'):.3f}-{row.get('roc_auc_ci95_hi'):.3f})",
            flush=True,
        )

    # Append FINDINGS Section 19 once all systems have been processed
    print(f"\n[findings] appending Section 19 to {FINDINGS_PATH}...", flush=True)
    import csv
    rows = {}
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["system"]] = r

    def fmt(sys_name: str) -> str:
        r = rows.get(sys_name, {})
        if not r:
            return "(missing — system did not complete)"
        return (
            f"macroF1@0.5={float(r['macro_f1_at_0.5']):.4f} "
            f"(CI95 {float(r['macro_f1_ci95_lo']):.3f}-{float(r['macro_f1_ci95_hi']):.3f}); "
            f"best={float(r['macro_f1_at_best']):.4f}@thr={float(r['best_threshold']):.2f}; "
            f"ROC-AUC={float(r['roc_auc']):.4f} "
            f"(CI95 {float(r['roc_auc_ci95_lo']):.3f}-{float(r['roc_auc_ci95_hi']):.3f})"
        )

    section = ["\n\n## 19. Public Arabic Dialect Classifiers - Lebanese Cross-Domain Evaluation\n"]
    section.append(f"_Generated {datetime.now(timezone.utc).isoformat()} by `scripts/24_eval_public_systems.py`. ROADMAP v2 Day 3._\n\n")
    section.append("### 19.1 Systems evaluated\n\n")
    section.append("| System | Model | Family | Lebanese label | Granularity |\n")
    section.append("|---|---|---|---|---|\n")
    for name, cfg in SYSTEMS.items():
        section.append(f"| {name} | `{cfg['model_id']}` | {cfg['family']}/{cfg['modality']} | `{cfg['lebanese_label']}` | {cfg['label_granularity']} |\n")
    section.append("\n")
    section.append("**Important caveat:** three of four systems do not expose a country-level Lebanese class; "
                   "we use Levantine probability as a Lebanese proxy. This *upper-bounds* their Lebanese detection "
                   "ability — a system that perfectly identifies Levantine but cannot distinguish LB from SY/JO/PA "
                   "will appear strong here. The Elyadata model (`LEB`) is the only honest country-level comparison.\n\n")
    section.append("### 19.2 Results on the held-out 296-item GT\n\n")
    section.append("| System | Statistics |\n")
    section.append("|---|---|\n")
    for name in SYSTEMS.keys():
        section.append(f"| {name} | {fmt(name)} |\n")
    section.append("\n")
    section.append("### 19.3 Interpretation\n\n")
    section.append(
        "*(See updated scoreboard in ROADMAP.md and §15 for context. Numbers above place these public "
        "systems alongside V1/V2/V2.5/Hybrid for the first systematic Lebanese cross-domain DID evaluation.)*\n"
    )
    # Idempotent write: strip any prior "## 19." block(s) before appending.
    current = FINDINGS_PATH.read_text(encoding="utf-8")
    marker = "\n\n## 19. Public Arabic Dialect Classifiers"
    cut = current.find(marker)
    if cut != -1:
        current = current[:cut].rstrip() + "\n"
    new_content = current + "".join(section)
    FINDINGS_PATH.write_text(new_content, encoding="utf-8")
    print("[done] Section 19 written (idempotent — replaced any prior block).", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
