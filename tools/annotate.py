#!/usr/bin/env python3
"""
Interactive annotation tool for building the ground-truth evaluation set.

Usage:
    python tools/annotate.py

Then open http://localhost:5000 in a browser.

Behavior:
  - On first run, picks a stratified random sample of 300 items from the DB
    and saves the list to data/annotation_sample.json (reproducible, resumable).
  - For each item, serves the middle 60 seconds of its source audio (extracted
    on demand via ffmpeg, cached in data/annotation_cache/).
  - User labels each clip via 5 options or keyboard shortcuts 1-5:
      1 = Lebanese
      2 = Mostly Lebanese / mixed
      3 = Not Lebanese
      4 = Unclear
      5 = Skip (come back later)
  - Annotations are saved to data/annotations.csv (resumable across sessions).

Sample design (300 total, justified in thesis methodology):
    45 WEAK_POSITIVE   — sanity check on metadata-based labels
    90 POTENTIAL_LB    — main precision target
    60 BORDERLINE_LB   — uncertainty band
    60 REJECTED        — false negative check
    45 WEAK_NEGATIVE   — sanity check on lexical negatives
"""

import csv
import json
import random
import sqlite3
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from flask import Flask, jsonify, request, send_file, Response
except ImportError:
    print("Flask is not installed. Install with: pip install flask")
    sys.exit(1)


# -----------------------
# Config
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "queue.db"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
SAMPLE_PATH = PROJECT_ROOT / "data" / "annotation_sample.json"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "annotations.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "annotation_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STRATA = {
    "WEAK_POSITIVE": 45,
    "POTENTIAL_LB": 90,
    "BORDERLINE_LB": 60,
    "REJECTED": 60,
    "WEAK_NEGATIVE": 45,
}
TOTAL_TARGET = sum(STRATA.values())
CLIP_DURATION = 60  # seconds of audio per clip

LABEL_OPTIONS = [
    ("lebanese", "Lebanese", "1"),
    ("mostly_lebanese", "Mostly Lebanese / mixed", "2"),
    ("not_lebanese", "Not Lebanese", "3"),
    ("unclear", "Unclear", "4"),
    ("skip", "Skip (come back later)", "5"),
]

RANDOM_SEED = 42  # deterministic sample selection


# -----------------------
# Sample construction
# -----------------------
def build_sample() -> list[dict]:
    """Construct the stratified random sample once and persist it."""
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    random.seed(RANDOM_SEED)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    sample = []
    for status, n in STRATA.items():
        cur.execute(
            "SELECT id, url, platform, audio_path, status FROM queue "
            "WHERE status = ? AND audio_path IS NOT NULL",
            (status,),
        )
        rows = cur.fetchall()
        rows = [r for r in rows if r[3] and Path(r[3]).exists()]
        if len(rows) < n:
            print(f"WARNING: only {len(rows)} items available for {status}, wanted {n}")
            chosen = rows
        else:
            chosen = random.sample(rows, n)
        for item_id, url, platform, audio_path, st in chosen:
            sample.append({
                "id": item_id,
                "url": url,
                "platform": platform,
                "audio_path": audio_path,
                "source_status": st,
            })

    conn.close()
    random.shuffle(sample)  # interleave strata so user doesn't label one bucket at a time
    SAMPLE_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sample built: {len(sample)} items saved to {SAMPLE_PATH}")
    return sample


def load_annotations() -> dict[int, dict]:
    """Load existing annotations keyed by item_id."""
    if not ANNOTATIONS_PATH.exists():
        return {}
    out = {}
    with ANNOTATIONS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row["item_id"])] = row
    return out


def save_annotation(item: dict, label: str, notes: str = "") -> None:
    """Append (or replace) an annotation row."""
    existing = load_annotations()
    existing[item["id"]] = {
        "item_id": str(item["id"]),
        "source_status": item["source_status"],
        "platform": item["platform"],
        "audio_path": item["audio_path"],
        "ground_truth": label,
        "notes": notes,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    fieldnames = ["item_id", "source_status", "platform", "audio_path",
                  "ground_truth", "notes", "timestamp"]
    with ANNOTATIONS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)


def get_transcript_excerpt(item_id: int, max_chars: int = 600) -> str:
    """Load the saved screening transcript text, truncated."""
    path = TRANSCRIPT_DIR / f"clip_{item_id}_screening.json"
    if not path.exists():
        return "(no transcript available)"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = " ".join(s.get("text", "") for s in data.get("screening_samples", []))
        return text[:max_chars].strip() or "(empty transcript)"
    except Exception as e:
        return f"(transcript read error: {e})"


def get_audio_duration(path: Path) -> float:
    """ffprobe to get duration in seconds."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ], stderr=subprocess.DEVNULL, timeout=30)
        return float(out.strip())
    except Exception:
        return 0.0


def extract_clip(source_path: Path, out_path: Path, duration: int = CLIP_DURATION) -> bool:
    """Extract a representative clip: start at 30% of the file, play for `duration` seconds."""
    total = get_audio_duration(source_path)
    if total <= 0:
        return False
    start = max(0, min(total - duration, total * 0.3))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}",
        "-t", str(duration),
        "-i", str(source_path),
        "-ac", "1", "-ar", "22050",
        "-c:a", "libmp3lame", "-b:a", "96k",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)

SAMPLE: list[dict] = []


def next_unlabeled_index() -> int:
    annotated = load_annotations()
    for i, item in enumerate(SAMPLE):
        ann = annotated.get(item["id"])
        if ann is None or ann.get("ground_truth") == "skip":
            return i
    return -1


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lebanese Audio Annotation</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #2f81f7;
    --lb: #2ea043;
    --mix: #d29922;
    --nolb: #da3633;
    --unclear: #6e7681;
    --skip: #484f58;
  }
  body { margin: 0; font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); padding: 24px; }
  .container { max-width: 820px; margin: 0 auto; }
  h1 { font-size: 18px; margin: 0 0 16px; color: var(--muted); font-weight: 500; }
  .progress-bar { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-bottom: 24px; }
  .progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
  .progress-text { font-size: 13px; color: var(--muted); margin-bottom: 16px; }
  .panel { background: var(--panel); padding: 20px; border-radius: 8px; margin-bottom: 16px; }
  .meta { display: flex; gap: 16px; font-size: 13px; color: var(--muted); margin-bottom: 12px; flex-wrap: wrap; }
  .meta .label { color: var(--text); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px;
           background: #21262d; color: var(--accent); font-weight: 600; }
  audio { width: 100%; margin: 12px 0; }
  .transcript { background: #0d1117; padding: 12px; border-radius: 6px;
                font-size: 14px; line-height: 1.6; max-height: 200px; overflow-y: auto;
                direction: rtl; font-family: "Segoe UI", Tahoma, sans-serif;
                border: 1px solid #30363d; white-space: pre-wrap; }
  .buttons { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 20px; }
  button { padding: 14px 8px; border: none; border-radius: 6px; color: white;
           font-size: 13px; font-weight: 600; cursor: pointer; transition: transform 0.1s; }
  button:active { transform: scale(0.96); }
  button .key { display: block; font-size: 10px; opacity: 0.7; margin-top: 4px; }
  .b-lebanese { background: var(--lb); }
  .b-mostly { background: var(--mix); }
  .b-notlb { background: var(--nolb); }
  .b-unclear { background: var(--unclear); }
  .b-skip { background: var(--skip); }
  .finished { text-align: center; padding: 40px; color: var(--muted); }
  .hint { font-size: 12px; color: var(--muted); margin-top: 12px; text-align: center; }
</style>
</head>
<body>
<div class="container">
  <h1>Lebanese Audio Ground-Truth Annotation</h1>
  <div class="progress-text" id="progressText">Loading...</div>
  <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>

  <div id="content"></div>
</div>

<script>
let current = null;

async function loadNext() {
  const res = await fetch('/api/next');
  const data = await res.json();
  if (data.done) {
    document.getElementById('content').innerHTML =
      '<div class="panel finished">' +
      '<h2>All done!</h2>' +
      '<p>You have labeled ' + data.annotated + ' / ' + data.total + ' items.</p>' +
      '<p>Results saved to <code>data/annotations.csv</code>.</p>' +
      '</div>';
    document.getElementById('progressFill').style.width = '100%';
    document.getElementById('progressText').textContent = 'Complete';
    return;
  }
  current = data.item;
  const pct = Math.round((data.annotated / data.total) * 100);
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent =
    'Progress: ' + data.annotated + ' / ' + data.total + ' labeled (' + pct + '%) — ' +
    'item #' + current.id + ' / ' + data.sample_index + ' in sample';

  document.getElementById('content').innerHTML = `
    <div class="panel">
      <div class="meta">
        <span><span class="label">Item:</span> ${current.id}</span>
        <span><span class="label">Platform:</span> ${current.platform}</span>
        <span><span class="label">Auto-label:</span> <span class="badge">${current.source_status}</span></span>
      </div>
      <audio controls autoplay src="/api/audio/${current.id}?t=${Date.now()}"></audio>
      <div class="transcript">${escapeHtml(current.transcript)}</div>
      <div class="buttons">
        <button class="b-lebanese"     onclick="label('lebanese')">Lebanese<span class="key">1</span></button>
        <button class="b-mostly"       onclick="label('mostly_lebanese')">Mostly LB / mixed<span class="key">2</span></button>
        <button class="b-notlb"        onclick="label('not_lebanese')">Not Lebanese<span class="key">3</span></button>
        <button class="b-unclear"      onclick="label('unclear')">Unclear<span class="key">4</span></button>
        <button class="b-skip"         onclick="label('skip')">Skip<span class="key">5</span></button>
      </div>
      <div class="hint">Keyboard shortcuts: 1 = Lebanese, 2 = Mostly LB, 3 = Not LB, 4 = Unclear, 5 = Skip &nbsp;•&nbsp; Space = replay audio</div>
    </div>`;
}

async function label(choice) {
  if (!current) return;
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({item_id: current.id, label: choice})
  });
  loadNext();
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const map = {'1':'lebanese','2':'mostly_lebanese','3':'not_lebanese','4':'unclear','5':'skip'};
  if (map[e.key]) { e.preventDefault(); label(map[e.key]); }
  if (e.key === ' ') {
    e.preventDefault();
    const a = document.querySelector('audio');
    if (a) { a.currentTime = 0; a.play(); }
  }
});

loadNext();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/next")
def api_next():
    annotated = load_annotations()
    real_annotated = sum(1 for a in annotated.values() if a.get("ground_truth") != "skip")
    idx = next_unlabeled_index()
    if idx < 0:
        return jsonify({
            "done": True,
            "annotated": real_annotated,
            "total": len(SAMPLE),
        })
    item = SAMPLE[idx]
    item_with_transcript = {
        **item,
        "transcript": get_transcript_excerpt(item["id"]),
    }
    return jsonify({
        "done": False,
        "item": item_with_transcript,
        "annotated": real_annotated,
        "total": len(SAMPLE),
        "sample_index": idx + 1,
    })


@app.route("/api/audio/<int:item_id>")
def api_audio(item_id):
    item = next((x for x in SAMPLE if x["id"] == item_id), None)
    if not item:
        return "Not found", 404
    source = Path(item["audio_path"])
    if not source.exists():
        return "Source audio missing", 404
    cache_path = CACHE_DIR / f"clip_{item_id}.mp3"
    if not cache_path.exists():
        if not extract_clip(source, cache_path):
            return "Extraction failed", 500
    return send_file(str(cache_path), mimetype="audio/mpeg")


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.get_json()
    item_id = data["item_id"]
    label = data["label"]
    item = next((x for x in SAMPLE if x["id"] == item_id), None)
    if not item:
        return jsonify({"error": "item not found"}), 404
    save_annotation(item, label)
    return jsonify({"ok": True})


def main():
    global SAMPLE
    SAMPLE = build_sample()

    annotated = load_annotations()
    real = sum(1 for a in annotated.values() if a.get("ground_truth") != "skip")
    print(f"Loaded {len(SAMPLE)} items in sample. Already labeled: {real}")
    print(f"Open http://localhost:5000 in your browser to start annotating.")
    print(f"Press Ctrl+C to stop. Progress is saved to {ANNOTATIONS_PATH}")

    try:
        webbrowser.open("http://localhost:5000")
    except Exception:
        pass

    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
