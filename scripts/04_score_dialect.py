#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import joblib
import numpy as np

from src.cfg import Settings
from src.db import DB
from src.dialect.scoring import final_dialect_score
from src.embeddings.engine import embed_text


MODEL_PATH = Path("models/dialect_classifier.joblib")

POTENTIAL_THRESHOLD = 0.75
BORDERLINE_THRESHOLD = 0.50


def build_feature_vector(text: str, diagnostics: dict):
    lex = diagnostics["lexicon_details"]

    lex_features = np.array([
        lex["lb"],
        lex["msa"],
        lex["strong_lb_hits"],
        lex["msa_ratio_core"],
        diagnostics["final_score"],
    ], dtype=float)

    embedding = embed_text(text).astype(float)

    return np.concatenate([lex_features, embedding])


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Run 05_train_dialect_model.py first.")

    model = joblib.load(MODEL_PATH)

    items = db.fetch_queue(status="SCREENED", limit=20)
    if not items:
        print("No SCREENED items to score.")
        return

    for item in items:
        transcript_path = (
            Path(settings.transcription.transcripts_dir)
            / f"clip_{item.id}_screening.json"
        )

        if not transcript_path.exists():
            db.update_status(item.id, "ERROR_SCORING", "Missing transcript")
            continue

        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        text = " ".join(s["text"] for s in data["screening_samples"])

        diagnostics = final_dialect_score(text)

        try:
            features = build_feature_vector(text, diagnostics)
            proba = model.predict_proba([features])[0][1]
        except Exception as e:
            db.update_status(item.id, "ERROR_SCORING", str(e))
            continue

        print(f"[{item.id}] model_probability={proba:.3f}")

        result_payload = {
            "model_probability": float(proba),
            "diagnostics": diagnostics,
        }

        if proba >= POTENTIAL_THRESHOLD:
            db.update_status(item.id, "POTENTIAL_LB")
        elif proba >= BORDERLINE_THRESHOLD:
            db.update_status(item.id, "BORDERLINE_LB")
        else:
            db.mark_rejected(item.id, result_payload)


if __name__ == "__main__":
    main()
    
    
    
    
# This script performs machine learning–based dialect classification on screened audio clips. It retrieves items marked as SCREENED, loads their previously 
# generated screening transcripts, and concatenates the sampled text segments into a single text input. From this text, it extracts structured linguistic 
# features using the existing lexicon-based scoring function (final_dialect_score) and generates semantic embeddings using a text embedding model. 
# These features are combined into a single feature vector and passed into a pre-trained classifier, which outputs a probability representing the likelihood that 
# the clip contains Lebanese dialect speech. Based on configurable probability thresholds, each item is labeled as POTENTIAL_LB, BORDERLINE_LB, or rejected. The script
# replaces earlier rule-based thresholding with probabilistic model inference while preserving lexicon features as part of the feature engineering pipeline. This stage 
# transforms dialect identification from a heuristic decision system into a supervised classification framework.