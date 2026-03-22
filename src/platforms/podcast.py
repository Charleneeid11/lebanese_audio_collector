# src/platforms/podcast.py

import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import subprocess


def _safe_filename_from_url(url: str) -> str:
    """
    Deterministic, collision-safe filename from URL.
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()

    path = urlparse(url).path
    ext = Path(path).suffix.lower()

    if ext not in {".mp3", ".m4a", ".wav", ".aac", ".ogg"}:
        ext = ".mp3"

    return f"podcast_{h}{ext}"


def download_audio(
    url: str,
    raw_dir: str,
    max_download_seconds: int,
):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = _safe_filename_from_url(url)
    out_path = raw_dir / filename

    # If already downloaded, reuse
    if out_path.exists():
        duration = _get_duration_seconds(out_path)
        return str(out_path), duration

    # Stream download (safe for large files)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    duration = _get_duration_seconds(out_path)

    # Hard cap duration
    if duration > max_download_seconds:
        raise RuntimeError(
            f"Podcast too long ({duration:.1f}s > {max_download_seconds}s)"
        )

    return str(out_path), duration


def _get_duration_seconds(path: Path) -> float:
    """
    Uses ffprobe (same as yt-dlp pipeline)
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return float(out.strip())




# This script handles controlled downloading of podcast audio from a direct URL and prepares it for later processing.
# It generates a deterministic, collision-safe filename by hashing the URL (SHA-1) and preserving only approved audio extensions, 
# defaulting to .mp3 if needed. When download_audio is called, it ensures the target directory exists and checks whether the file 
# has already been downloaded; if so, it reuses it and simply computes its duration. If not, it streams the audio in chunks using 
# requests (to safely handle large files), saves it locally, and then measures its duration using ffprobe. A hard duration limit is 
# enforced—if the file exceeds max_download_seconds, the process raises an error. The helper _get_duration_seconds extracts the audio 
# length via an ffprobe subprocess call. Overall, the module ensures safe, deduplicated storage and enforces duration constraints
# before the audio enters the rest of the pipeline.