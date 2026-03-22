#whisper_engine.py
#Load the Whisper model once (not for every clip)
#Transcribe .wav files into text
#Return a structured Python dict (text + segments + language info)

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from faster_whisper import WhisperModel
from src.cfg import Settings

_model_cache: WhisperModel | None = None


def _get_screening_model(settings: Settings) -> WhisperModel:
    """
    Load the fast model used ONLY for screening (tiny/small/medium).
    """
    global _model_cache
    if _model_cache is None:
        cfg = settings.transcription
        _model_cache = WhisperModel(
            cfg.screening_model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
    return _model_cache

def transcribe_screening(audio_path: str, settings: Settings) -> Dict[str, Any]:
    """
    Transcribe a short audio sample (20 seconds) fast.
    """
    model = _get_screening_model(settings)

    segments, info = model.transcribe(
        audio_path,
        language="ar",                
        beam_size=5,                  # MORE ACCURATE than 3
        best_of=5,                    # improves Arabic decoding
        chunk_length=30,              # keeps it fast but improves context
        vad_filter=True,
        condition_on_previous_text=False,
    )

    segs = []
    texts = []
    for s in segments:
        segs.append({
            "id": s.id,
            "start": s.start,
            "end": s.end,
            "text": s.text
        })
        texts.append(s.text)

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "text": " ".join(texts),
        "segments": segs,
    }
