# src/platforms/tiktok.py
import subprocess
from pathlib import Path
from urllib.parse import urlparse

def _safe_basename_from_url(url: str) -> str:
    u = urlparse(url)
    last = (u.path.strip("/").split("/")[-1] or "tiktok").replace(".", "_")
    host = (u.hostname or "tiktok").replace(".", "_")
    return f"{host}_{last}"

def _ffprobe_duration(path: str) -> float:
    p = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return float(p.stdout.strip()) if p.stdout.strip() else 0.0

def download_audio(url: str, raw_dir: str, max_download_seconds: int) -> tuple[str, float]:
    raw_dir_p = Path(raw_dir)
    raw_dir_p.mkdir(parents=True, exist_ok=True)

    base = _safe_basename_from_url(url)
    out_wav = raw_dir_p / f"tiktok_{base}.wav"

    if out_wav.exists() and out_wav.stat().st_size > 0:
        dur = _ffprobe_duration(str(out_wav))
        if dur > 1.0:
            return str(out_wav), dur

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "-o", str(raw_dir_p / f"tiktok_{base}.%(ext)s"),
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-t {int(max_download_seconds)}",
        url,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())

    candidates = list(raw_dir_p.glob(f"tiktok_{base}.*"))
    if not candidates:
        raise RuntimeError("Download finished but output file not found")

    wav = next((c for c in candidates if c.suffix.lower() == ".wav"), None)
    pick = wav or candidates[0]

    if pick != out_wav:
        pick.rename(out_wav)

    dur = _ffprobe_duration(str(out_wav))
    if dur > max_download_seconds + 1:
        trimmed = out_wav.with_name(out_wav.stem + "_trim.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(out_wav),
                "-t", str(int(max_download_seconds)),
                str(trimmed),
            ],
            check=True,
            capture_output=True,
        )
        out_wav.unlink()
        trimmed.rename(out_wav)
        dur = _ffprobe_duration(str(out_wav))

    return str(out_wav), dur
