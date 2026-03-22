#audio.py
#Extracts short random audio chunks from a longer WAV file using ffmpeg, used for fast dialect-screening instead of full transcription.

import subprocess
from pathlib import Path
from random import randint
from typing import List


def extract_random_chunks(
    input_wav: str,
    chunk_length: int = 20,
    num_chunks: int = 3,
    out_dir: str = "data/samples"
) -> List[str]:
    """
    Extract num_chunks random chunks of chunk_length seconds from input_wav.
    Returns a list of file paths to these chunks.
    """
    inp = Path(input_wav)
    assert inp.exists(), f"Input file does not exist: {inp}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # get duration via ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(inp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())

    # ensure duration is long enough
    if duration < chunk_length:
        # only one chunk from the beginning
        start_times = [0]
    else:
        # pick random start positions
        max_start = int(duration - chunk_length)
        start_times = [randint(0, max_start) for _ in range(num_chunks)]

    chunk_paths = []
    for idx, start in enumerate(start_times):
        out_path = Path(out_dir) / f"{inp.stem}_chunk{idx}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(chunk_length),
            "-i", str(inp),
            "-ac", "1",
            "-ar", "16000",
            "-af", "loudnorm",  # normalize audio for Whisper
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunk_paths.append(str(out_path))

    return chunk_paths
