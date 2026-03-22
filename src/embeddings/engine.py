from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _load_model():
    """
    Load embedding model once and cache it.
    Prevents reloading for every call.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    """
    Convert input text into a fixed-size embedding vector.

    Returns:
        numpy array of shape (384,)
    """
    if not text or not text.strip():
        # Return zero vector if text is empty
        return np.zeros(384, dtype=float)

    model = _load_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,  # helpful for classification stability
    )

    return embedding