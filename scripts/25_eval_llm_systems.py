#!/usr/bin/env python3
"""
ROADMAP v2 Day 4 — LLM-family evaluation on the Lebanese cross-domain GT.

This script evaluates LLM-based dialect classifiers on the held-out 296-item
GT, using the same persistence + bootstrap-CI machinery as Day 1-3 systems.

Currently supported providers:
  - Gemini 2.5 Flash (Google AI Studio free tier) -- requires GEMINI_API_KEY in .env

Per-system pause/resume:
  - Each system writes data/benchmark_predictions/<name>.json atomically on completion.
  - Within a run, partial predictions are checkpointed to <name>.partial.json every
    20 items so an interruption only loses up to ~20 items of progress.
  - On restart, the partial file is reloaded and only missing items are queried.

Usage:
  python scripts/25_eval_llm_systems.py
  python scripts/25_eval_llm_systems.py --only gemini_flash_zeroshot
  python scripts/25_eval_llm_systems.py --only gemini_flash_3shot --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_harness_utils import (
    PREDS_DIR, RESULTS_CSV, compute_metrics, load_gt_items, load_predictions,
    save_predictions, upsert_csv_row,
)

TRANSCRIPTS_DIR = Path("data/transcripts")
FINDINGS_PATH = Path("FINDINGS.md")

# Hardcoded 3-shot example item_ids (selected from training pool, NOT in GT).
# Stable across runs so the prompt is reproducible.
THREESHOT_EXAMPLES = [
    {"item_id": 9,     "is_lebanese": True,
     "note": "Lebanese — colloquial football commentary, 'في فاول'/'برافو عليك'"},
    {"item_id": 23738, "is_lebanese": False,
     "note": "Egyptian — cinema/cultural narrative, 'في مصر والعالم العربي'/'دراما والسينما المصريتان'"},
    {"item_id": 23755, "is_lebanese": False,
     "note": "Gulf-flavored Arabic — driving safety, 'بومبة جيدة'/'شرات ضوئية'"},
]


# ---------------------------------------------------------------------------
# Transcript loader
# ---------------------------------------------------------------------------

def load_transcript(item_id: int, max_chars: int = 1500) -> str | None:
    """Load and concatenate the 3 screening chunks for an item.

    Returns None if no transcript exists or it's blank.
    """
    tpath = TRANSCRIPTS_DIR / f"clip_{item_id}_screening.json"
    if not tpath.exists():
        return None
    try:
        data = json.loads(tpath.read_text(encoding="utf-8"))
    except Exception:
        return None
    text = " ".join(s.get("text", "") for s in (data.get("screening_samples") or [])).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a careful Arabic dialect identifier. "
    "Given an Arabic transcript, you decide whether it is in **Lebanese Arabic** "
    "dialect specifically (not Syrian, Jordanian, or Palestinian Levantine; "
    "not Egyptian; not Gulf; not Moroccan/Maghrebi; not Modern Standard Arabic). "
    "Reply with ONLY a JSON object of the form "
    '{"is_lebanese": <true|false>, "confidence": <float in [0,1]>}. '
    "Use confidence to indicate how sure you are. Do not output any other text."
)


def build_user_prompt(transcript: str, fewshot_block: str | None = None) -> str:
    parts = []
    if fewshot_block:
        parts.append(fewshot_block.rstrip())
        parts.append("")
        parts.append("Now classify the following transcript:")
    else:
        parts.append("Classify the following transcript:")
    parts.append('"""')
    parts.append(transcript)
    parts.append('"""')
    parts.append('Respond with ONLY the JSON object: {"is_lebanese": <true|false>, "confidence": <float>}.')
    return "\n".join(parts)


def build_fewshot_block() -> str:
    """Build the 3-shot demonstration block from THREESHOT_EXAMPLES."""
    lines = ["Examples:"]
    for i, ex in enumerate(THREESHOT_EXAMPLES, 1):
        text = load_transcript(ex["item_id"])
        if text is None:
            raise RuntimeError(f"3-shot example item_id={ex['item_id']} has no transcript")
        ans = json.dumps({"is_lebanese": ex["is_lebanese"], "confidence": 0.92})
        lines.append("")
        lines.append(f"Example {i} ({ex['note']}):")
        lines.append('"""')
        lines.append(text[:1200])
        lines.append('"""')
        lines.append(f"Answer: {ans}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_response_to_prob_leb(raw: str) -> float | None:
    """Pull P(Lebanese) out of the model response.

    Try strict JSON first; fall back to extracting the first {...} block.
    Returns None if we cannot parse a reasonable answer.
    """
    if not raw:
        return None
    raw = raw.strip()
    # Strip Markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = None
    for candidate in [raw, *_JSON_OBJ_RE.findall(raw)]:
        try:
            data = json.loads(candidate)
            break
        except Exception:
            continue
    if not isinstance(data, dict):
        return None

    is_leb = data.get("is_lebanese")
    if is_leb is None:
        # Some models say {"dialect": "Lebanese"} — handle that too.
        dialect = (data.get("dialect") or "").strip().lower()
        if dialect:
            is_leb = "lebanese" in dialect
    if is_leb is None:
        return None
    if isinstance(is_leb, str):
        is_leb = is_leb.strip().lower() in {"true", "yes", "lebanese", "1"}
    is_leb = bool(is_leb)

    conf = data.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.9 if is_leb else 0.9  # default high confidence if unspecified

    conf = max(0.0, min(1.0, conf))
    return conf if is_leb else (1.0 - conf)


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

def _load_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GEMINI_API_KEY not set. Add to .env or environment.")


def _call_gemini_rest(model_id: str, api_key: str, system_instr: str,
                      user_prompt: str, timeout: float = 60.0) -> str:
    """Call Gemini via REST (no SDK — avoids pydantic dep). Returns raw text."""
    import requests
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_id}:generateContent")
    body = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_instr}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    r = requests.post(url, params={"key": api_key}, json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"no candidates in response: {data}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    return text


def predict_gemini(model_id: str, system_name: str, items: list[dict],
                   use_fewshot: bool) -> dict[int, float]:
    """Run Gemini Flash on the GT items. Resumable via .partial.json checkpoint."""
    api_key = _load_gemini_api_key()
    fewshot_block = build_fewshot_block() if use_fewshot else None
    print(f"  [prompt] system={len(SYSTEM_INSTRUCTION)}ch  "
          f"fewshot={'yes' if use_fewshot else 'no'} "
          f"({len(fewshot_block) if fewshot_block else 0}ch)", flush=True)

    partial_path = PREDS_DIR / f"{system_name}.partial.json"
    probs: dict[int, float] = {}
    if partial_path.exists():
        try:
            probs = {int(k): float(v) for k, v in json.loads(
                partial_path.read_text(encoding="utf-8")
            ).items()}
            print(f"  [resume] {len(probs)} predictions loaded from partial checkpoint",
                  flush=True)
        except Exception:
            probs = {}

    n = len(items)
    failures = 0
    skipped_no_text = 0
    # Gemini 2.5 Flash free tier is 10 RPM. 7.0s pause = ~8.5 RPM safely below.
    rate_limit_pause = 7.0
    last_save = time.time()

    for i, it in enumerate(items):
        iid = it["item_id"]
        if iid in probs:
            continue
        text = load_transcript(iid)
        if text is None:
            skipped_no_text += 1
            continue
        user_prompt = build_user_prompt(text, fewshot_block)

        prob = None
        for attempt in range(5):
            try:
                raw = _call_gemini_rest(model_id, api_key, SYSTEM_INSTRUCTION,
                                        user_prompt)
                prob = parse_response_to_prob_leb(raw)
                if prob is None:
                    print(f"    [warn] item {iid} unparsable response: {raw[:160]!r}",
                          flush=True)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                wait = 8.0 * (attempt + 1)
                print(f"    [retry {attempt+1}/5] item {iid}: {msg[:200]} "
                      f"(waiting {wait:.0f}s)", flush=True)
                time.sleep(wait)
        if prob is None:
            failures += 1
        else:
            probs[iid] = prob

        time.sleep(rate_limit_pause)

        if (i + 1) % 20 == 0 or (time.time() - last_save) > 60:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps({str(k): v for k, v in probs.items()}, indent=2),
                encoding="utf-8",
            )
            last_save = time.time()
            print(f"  [{system_name}] {i+1}/{n}  ok={len(probs)}  "
                  f"failed={failures}  no_text={skipped_no_text}", flush=True)

    if partial_path.exists():
        partial_path.unlink()

    print(f"  [final] ok={len(probs)}  failed={failures}  no_text={skipped_no_text}",
          flush=True)
    return probs


# ---------------------------------------------------------------------------
# Groq provider (Llama 3.1 8B via Groq's LPU API; free tier 14,400 RPD)
# ---------------------------------------------------------------------------

def _load_groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GROQ_API_KEY not set. Add to .env or environment.")


def _call_groq_rest(model_id: str, api_key: str, system_instr: str,
                    user_prompt: str, timeout: float = 60.0) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint."""
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_instr},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"no choices in response: {data}")
    return (choices[0].get("message") or {}).get("content", "").strip()


def predict_groq(model_id: str, system_name: str, items: list[dict],
                 use_fewshot: bool) -> dict[int, float]:
    """Run Groq Llama 3.1 8B on the GT items. Resumable via .partial.json checkpoint."""
    api_key = _load_groq_api_key()
    fewshot_block = build_fewshot_block() if use_fewshot else None
    print(f"  [prompt] system={len(SYSTEM_INSTRUCTION)}ch  "
          f"fewshot={'yes' if use_fewshot else 'no'} "
          f"({len(fewshot_block) if fewshot_block else 0}ch)", flush=True)

    partial_path = PREDS_DIR / f"{system_name}.partial.json"
    probs: dict[int, float] = {}
    if partial_path.exists():
        try:
            probs = {int(k): float(v) for k, v in json.loads(
                partial_path.read_text(encoding="utf-8")
            ).items()}
            print(f"  [resume] {len(probs)} predictions loaded from partial checkpoint",
                  flush=True)
        except Exception:
            probs = {}

    n = len(items)
    failures = 0
    skipped_no_text = 0
    # Groq free tier: 30 RPM on Llama-3.1-8B. 2.5s pause = ~24 RPM safely below.
    rate_limit_pause = 2.5
    last_save = time.time()

    for i, it in enumerate(items):
        iid = it["item_id"]
        if iid in probs:
            continue
        text = load_transcript(iid)
        if text is None:
            skipped_no_text += 1
            continue
        user_prompt = build_user_prompt(text, fewshot_block)

        prob = None
        for attempt in range(5):
            try:
                raw = _call_groq_rest(model_id, api_key, SYSTEM_INSTRUCTION,
                                      user_prompt)
                prob = parse_response_to_prob_leb(raw)
                if prob is None:
                    print(f"    [warn] item {iid} unparsable response: {raw[:160]!r}",
                          flush=True)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                wait = 4.0 * (attempt + 1)
                print(f"    [retry {attempt+1}/5] item {iid}: {msg[:200]} "
                      f"(waiting {wait:.0f}s)", flush=True)
                time.sleep(wait)
        if prob is None:
            failures += 1
        else:
            probs[iid] = prob

        time.sleep(rate_limit_pause)

        if (i + 1) % 25 == 0 or (time.time() - last_save) > 60:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps({str(k): v for k, v in probs.items()}, indent=2),
                encoding="utf-8",
            )
            last_save = time.time()
            print(f"  [{system_name}] {i+1}/{n}  ok={len(probs)}  "
                  f"failed={failures}  no_text={skipped_no_text}", flush=True)

    if partial_path.exists():
        partial_path.unlink()

    print(f"  [final] ok={len(probs)}  failed={failures}  no_text={skipped_no_text}",
          flush=True)
    return probs


# ---------------------------------------------------------------------------
# AceGPT-7B local provider (llama-cpp-python on CPU)
# ---------------------------------------------------------------------------

# Lazy global to avoid re-loading the 4-GB GGUF for every system
_ACEGPT_LLM = None


def _load_acegpt_llm(model_path: str):
    global _ACEGPT_LLM
    if _ACEGPT_LLM is None:
        from llama_cpp import Llama
        print(f"  [load] AceGPT GGUF: {model_path}", flush=True)
        _ACEGPT_LLM = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
    return _ACEGPT_LLM


def predict_acegpt(model_id: str, system_name: str, items: list[dict],
                   use_fewshot: bool) -> dict[int, float]:
    """Run AceGPT-7B int4 via llama-cpp-python. Resumable via .partial.json."""
    llm = _load_acegpt_llm(model_id)
    fewshot_block = build_fewshot_block() if use_fewshot else None
    print(f"  [prompt] system={len(SYSTEM_INSTRUCTION)}ch  "
          f"fewshot={'yes' if use_fewshot else 'no'} "
          f"({len(fewshot_block) if fewshot_block else 0}ch)", flush=True)

    partial_path = PREDS_DIR / f"{system_name}.partial.json"
    probs: dict[int, float] = {}
    if partial_path.exists():
        try:
            probs = {int(k): float(v) for k, v in json.loads(
                partial_path.read_text(encoding="utf-8")
            ).items()}
            print(f"  [resume] {len(probs)} predictions loaded from partial checkpoint",
                  flush=True)
        except Exception:
            probs = {}

    n = len(items)
    failures = 0
    skipped_no_text = 0
    last_save = time.time()

    for i, it in enumerate(items):
        iid = it["item_id"]
        if iid in probs:
            continue
        text = load_transcript(iid)
        if text is None:
            skipped_no_text += 1
            continue
        user_prompt = build_user_prompt(text, fewshot_block)

        # llama-cpp-python chat template handles system + user roles
        try:
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=128,
                response_format={"type": "json_object"} if False else None,  # AceGPT base may not enforce
            )
            raw = (out.get("choices", [{}])[0].get("message", {})
                   .get("content", "")).strip()
            prob = parse_response_to_prob_leb(raw)
            if prob is None:
                print(f"    [warn] item {iid} unparsable response: {raw[:160]!r}",
                      flush=True)
                failures += 1
            else:
                probs[iid] = prob
        except Exception as e:
            print(f"    [warn] item {iid} failed: {type(e).__name__}: {e}", flush=True)
            failures += 1

        if (i + 1) % 10 == 0 or (time.time() - last_save) > 60:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps({str(k): v for k, v in probs.items()}, indent=2),
                encoding="utf-8",
            )
            last_save = time.time()
            print(f"  [{system_name}] {i+1}/{n}  ok={len(probs)}  "
                  f"failed={failures}  no_text={skipped_no_text}", flush=True)

    if partial_path.exists():
        partial_path.unlink()

    print(f"  [final] ok={len(probs)}  failed={failures}  no_text={skipped_no_text}",
          flush=True)
    return probs


# ---------------------------------------------------------------------------
# Systems registry
# ---------------------------------------------------------------------------

SYSTEMS = {
    # AceGPT-7B-chat (Arabic instruction-tuned Llama-2) via llama-cpp-python.
    # Open weights, free, CPU. The Arabic-specialized LLM in the benchmark.
    "acegpt_7b_zeroshot": {
        "model_id": "models/AceGPT-7B-chat.Q4_K_M.gguf",
        "provider": "acegpt",
        "family": "llm",
        "use_fewshot": False,
        "citation": "AceGPT-7B-chat Q4_K_M (FreedomIntelligence, 2024); mradermacher GGUF",
    },
    "acegpt_7b_3shot": {
        "model_id": "models/AceGPT-7B-chat.Q4_K_M.gguf",
        "provider": "acegpt",
        "family": "llm",
        "use_fewshot": True,
        "citation": "AceGPT-7B-chat Q4_K_M (FreedomIntelligence, 2024); mradermacher GGUF",
    },
    # Groq Llama-3.1-8B-Instant — general-purpose multilingual LLM as the
    # second LLM-family row. Free tier ~14k RPD; OpenAI-compatible API.
    "groq_llama31_8b_zeroshot": {
        "model_id": "llama-3.1-8b-instant",
        "provider": "groq",
        "family": "llm",
        "use_fewshot": False,
        "citation": "Llama-3.1-8B-Instant (Meta, 2024) via Groq free tier",
    },
    "groq_llama31_8b_3shot": {
        "model_id": "llama-3.1-8b-instant",
        "provider": "groq",
        "family": "llm",
        "use_fewshot": True,
        "citation": "Llama-3.1-8B-Instant (Meta, 2024) via Groq free tier",
    },
    # Gemini 2.5 Flash-Lite — kept as a sanity row, but free-tier RPD is 20
    # so only the first ~18 GT items get scored. Document as partial.
    "gemini_flash_lite_zeroshot": {
        "model_id": "gemini-2.5-flash-lite",
        "provider": "gemini",
        "family": "llm",
        "use_fewshot": False,
        "citation": "Gemini 2.5 Flash-Lite (Google, 2025) via AI Studio free tier "
                    "(partial — 20 RPD quota)",
    },
}


PROVIDERS = {
    "gemini": predict_gemini,
    "groq": predict_groq,
    "acegpt": predict_acegpt,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if final predictions JSON exists")
    ap.add_argument("--only", action="append", default=[],
                    help="Run only the named system(s); repeatable")
    ap.add_argument("--skip", action="append", default=[],
                    help="Skip named system(s); repeatable")
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
            print(f"\n[skip ] {system_name} — predictions on disk, loading", flush=True)
            probs = load_predictions(preds_path)
        else:
            print(f"\n[start] {system_name} ({cfg['family']}/{cfg['provider']}) "
                  f"— {cfg['model_id']}  fewshot={cfg['use_fewshot']}", flush=True)
            try:
                provider_fn = PROVIDERS[cfg["provider"]]
                probs = provider_fn(
                    cfg["model_id"], system_name, items, cfg["use_fewshot"]
                )
            except Exception as e:
                print(f"[error] {system_name} crashed: {type(e).__name__}: {e}",
                      flush=True)
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

    # Append (idempotent) FINDINGS Section 20
    print(f"\n[findings] writing Section 20 to {FINDINGS_PATH}…", flush=True)
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

    section = ["\n\n## 20. LLM-family Evaluation — Gemini Flash Zero-shot and 3-shot\n"]
    section.append(f"_Generated {datetime.now(timezone.utc).isoformat()} "
                   f"by `scripts/25_eval_llm_systems.py`. ROADMAP v2 Day 4._\n\n")
    section.append("### 20.1 Systems evaluated\n\n")
    section.append("| System | Model | Family | Shots |\n")
    section.append("|---|---|---|---|\n")
    for name, cfg in SYSTEMS.items():
        shots = "3-shot" if cfg["use_fewshot"] else "zero-shot"
        section.append(f"| {name} | `{cfg['model_id']}` | {cfg['family']} | {shots} |\n")
    section.append("\n")
    section.append("Prompt design: a system instruction tells the model to decide whether "
                   "the transcript is in **Lebanese Arabic specifically** (not Syrian/"
                   "Jordanian/Palestinian Levantine, not Egyptian/Gulf/Maghrebi, not MSA). "
                   "The model returns strict JSON `{is_lebanese: bool, confidence: float}` "
                   "with temperature 0. P(Lebanese) is computed as "
                   "`confidence if is_lebanese else 1.0 - confidence`. 3-shot uses three "
                   "training-pool examples — one Lebanese, one Egyptian, one Gulf-flavored "
                   "— picked outside the GT to avoid leakage.\n\n")
    section.append("### 20.2 Results on the held-out 296-item GT\n\n")
    section.append("| System | Statistics |\n")
    section.append("|---|---|\n")
    for name in SYSTEMS.keys():
        section.append(f"| {name} | {fmt(name)} |\n")
    section.append("\n")
    section.append("### 20.3 Interpretation\n\n")
    section.append("_(Interpretation pending; this block is regenerated each run. "
                   "See ROADMAP §4 Day 4 for context.)_\n")

    # Idempotent write
    current = FINDINGS_PATH.read_text(encoding="utf-8")
    marker = "\n\n## 20. LLM-family Evaluation"
    cut = current.find(marker)
    if cut != -1:
        current = current[:cut].rstrip() + "\n"
    FINDINGS_PATH.write_text(current + "".join(section), encoding="utf-8")
    print("[done] Section 20 written (idempotent).", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
