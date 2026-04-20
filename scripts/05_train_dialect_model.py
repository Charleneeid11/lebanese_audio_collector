#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.cfg import Settings
from src.db import DB
from src.dialect.scoring import final_dialect_score
from src.embeddings.engine import embed_text


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "dialect_classifier.joblib"


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


def load_transcript_text(item_id: int, transcripts_dir: str) -> str | None:
    path = Path(transcripts_dir) / f"clip_{item_id}_screening.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return " ".join(s["text"] for s in data["screening_samples"])


# Minimum strong Lebanese markers (شو, هيك, هلق, بدي, etc.) required for a
# WEAK_POSITIVE item to be used as positive training data. Items from trusted
# Lebanese channels with zero strong markers are likely MSA/formal content.
# Threshold determined empirically: ground truth annotation of 45 WEAK_POSITIVE
# items showed precision jumps from 67% (unfiltered) to 81% (strong >= 1).
MIN_STRONG_LB_HITS = 1


def main():
    settings = Settings.load()
    db = DB(settings.db_url)

    positives = db.fetch_queue(status="WEAK_POSITIVE", limit=None)
    negatives = db.fetch_queue(status="WEAK_NEGATIVE", limit=None)
    neg_rejected = db.fetch_queue(status="REJECTED", limit=None)

    print(f"Raw counts — WEAK_POSITIVE: {len(positives)}, WEAK_NEGATIVE: {len(negatives)}, REJECTED: {len(neg_rejected)}")

    X = []
    y = []
    pos_kept, pos_filtered = 0, 0

    for item in positives:
        text = load_transcript_text(item.id, settings.transcription.transcripts_dir)
        if text is None:
            continue

        diagnostics = final_dialect_score(text)

        if diagnostics["lexicon_details"]["strong_lb_hits"] < MIN_STRONG_LB_HITS:
            pos_filtered += 1
            continue

        features = build_feature_vector(text, diagnostics)
        X.append(features)
        y.append(1)
        pos_kept += 1

    print(f"Positives — kept: {pos_kept}, filtered out (strong_lb < {MIN_STRONG_LB_HITS}): {pos_filtered}")

    neg_count = 0
    for item in negatives + neg_rejected:
        text = load_transcript_text(item.id, settings.transcription.transcripts_dir)
        if text is None:
            continue

        diagnostics = final_dialect_score(text)
        features = build_feature_vector(text, diagnostics)
        X.append(features)
        y.append(0)
        neg_count += 1

    print(f"Negatives — total: {neg_count} (WEAK_NEGATIVE + REJECTED with transcripts)")

    X = np.array(X)
    y = np.array(y)

    if len(X) < 50:
        raise RuntimeError("Not enough labeled data to train model.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    print("\nValidation Report:\n")
    print(classification_report(y_val, preds))
    print("ROC-AUC:", roc_auc_score(y_val, probs))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()




# This script trains the supervised dialect classification model using weakly labeled data from the existing pipeline. It retrieves previously classified items  
# from the database (e.g., POTENTIAL_LB as positive examples and REJECTED as negative examples), loads their screening transcripts, and extracts both lexicon-derived 
# features and semantic embeddings from the transcript text. These combined features form the training input for a logistic regression classifier. The dataset is split
# into training and validation subsets to evaluate performance using standard metrics such as classification report and ROC-AUC. After training, the model is serialized
# and saved for later inference by the scoring script. This script operationalizes the transition from rule-based filtering to supervised learning by leveraging existing 
# pipeline outputs as weak labels to train a probabilistic dialect classifier.