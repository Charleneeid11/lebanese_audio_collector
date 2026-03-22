# src/platforms/youtube.py
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import subprocess

from yt_dlp import YoutubeDL

from src.cfg import Settings


def _trim_with_ffmpeg(input_wav: Path, output_wav: Path, max_seconds: int) -> None:
    """
    Use ffmpeg to cut the input file to the first max_seconds.
    Overwrites output_wav.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_wav),
        "-t", str(max_seconds),
        "-ac", "1",
        "-ar", "16000",
        "-af", "loudnorm",   # normalize volume for cleaner Whisper input
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def download_audio(
    url: str,
    out_dir: str,
    max_download_seconds: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Download audio from YouTube as WAV.
    If max_download_seconds is set, the final WAV on disk will be
    trimmed to at most that many seconds.

    Returns: (audio_path, clip_duration_seconds)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # First: download & convert full audio to WAV via yt-dlp
    tmp_template = out_path / "%(id)s_full.%(ext)s"

    # Use tv/ios client to avoid 403: YouTube's web client now uses SABR streaming
    # (unsupported by yt-dlp). See https://github.com/yt-dlp/yt-dlp/issues/12482
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp_template),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "extractor_args": {"youtube": ["player_client=tv,ios"]},
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info["id"]
    full_duration = int(info.get("duration") or 0)

    full_wav = out_path / f"{video_id}_full.wav"
    final_wav = out_path / f"{video_id}.wav"

    if max_download_seconds is not None and full_duration > max_download_seconds:
        _trim_with_ffmpeg(full_wav, final_wav, max_download_seconds)
        try:
            full_wav.unlink()
        except FileNotFoundError:
            pass
        clip_duration = max_download_seconds
    else:
        if full_wav.exists():
            full_wav.rename(final_wav)
        clip_duration = full_duration

    return str(final_wav), clip_duration


def discover_candidates(settings: Settings) -> List[Dict]:
    """
    Placeholder for YouTube discovery.
    Later: use YouTube Data API with regionCode=LB, relevanceLanguage=ar,
    search_queries and channels from settings.platforms.youtube.
    """
    yt_cfg = settings.platforms.youtube
    # For now, return empty; we'll implement this in the next step.
    return []




# This script manages YouTube audio ingestion and optional trimming before downstream processing. The download_audio function downloads
# the best available audio using yt_dlp, automatically converts it to WAV via an FFmpeg postprocessor, 
# and stores it using a temporary filename pattern based on the YouTube video ID. It retrieves the full video duration from metadata,
# then determines whether trimming is required. If max_download_seconds is set and the video exceeds that limit, it calls _trim_with_ffmpeg
# to cut the file to the first specified seconds, force mono audio (-ac 1), resample to 16 kHz (-ar 16000), and apply loudness normalization (loudnorm)
# to improve transcription quality. The original full-length file is deleted after trimming. 
# If no trimming is needed, the full WAV file is simply renamed to its final filename. The function returns the final WAV path and the effective clip duration.
# The discover_candidates function is currently a placeholder for future YouTube Data API-based content discovery but does not yet implement any logic.