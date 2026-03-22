from src.dialect.lexicons import (
    LEBANESE_WORDS,
    MSA_WORDS,
    EGYPTIAN_WORDS,
    GULF_WORDS,
    SYRIAN_WORDS,
)

# Strong Lebanese/colloquial markers (small, high-precision set)
STRONG_LEBANESE_MARKERS = [
    "شو", "ليش", "هيك", "هلّق", "هلق", "عنجد", "عن جد",
    "بدّي", "بدي", "ما في", "مابي", "مش", "عم ", "رح ",
    "كتير", "كتير ", "إنت", "انت", "إنتي", "انتي", "كنتي",
    "وين", "هون", "هونيك", "هيدا", "هيدي", "هدول",
]


def count_matches(text: str, words):
    text = text.lower()
    return sum(1 for w in words if w in text)


def lexicon_score(text: str):
    lb = count_matches(text, LEBANESE_WORDS)
    msa = count_matches(text, MSA_WORDS)
    egy = count_matches(text, EGYPTIAN_WORDS)
    gulf = count_matches(text, GULF_WORDS)
    sy = count_matches(text, SYRIAN_WORDS)

    strong_lb_hits = count_matches(text, STRONG_LEBANESE_MARKERS)

    # Ratio that matters most: MSA vs (LB+MSA)
    core_total = lb + msa
    msa_ratio_core = (msa / core_total) if core_total > 0 else 0.0

    raw_score = (
        lb * 1.8
        - msa * 0.6
        - egy * 1.0
        - gulf * 1.0
        - sy * 0.5
    )

    # Mild extra penalty only when MSA dominates the LB/MSA core
    if msa_ratio_core >= 0.60:
        raw_score -= msa * 0.7

    norm_score = max(0.0, min(1.0, raw_score / 5.0))

    details = {
        "lb": lb,
        "msa": msa,
        "egy": egy,
        "gulf": gulf,
        "sy": sy,
        "strong_lb_hits": strong_lb_hits,
        "msa_ratio_core": round(msa_ratio_core, 3),
        "raw_score": raw_score,
        "lexicon_score": norm_score,
    }

    return norm_score, details


def final_dialect_score(text: str):
    lex_score, lex_details = lexicon_score(text)

    return {
        "final_score": lex_score,
        "lex_score": lex_score,
        "lexicon_details": lex_details,
        "adi": None,  # intentionally disabled
    }
