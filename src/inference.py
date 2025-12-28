import joblib
import numpy as np
from difflib import get_close_matches
from pathlib import Path

from src.utils import normalize_text
from src.features import build_rank_features
from src.ranker import Ranker

# -----------------------------
# Load trained artifact
# -----------------------------
MODEL_PATH = Path("models/pipeline.joblib")

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
vectors = artifact["vectors"]
titles = artifact["titles"]
title_lookup = artifact["title_lookup"]
full_df = artifact["full_df"]

tfidf = pipeline.named_steps["tfidf"]
normalizer = pipeline.named_steps["normalize"]
nn = pipeline.named_steps["nn"]

# -----------------------------
# Load champion ranker
# -----------------------------
ranker = Ranker()

# -----------------------------
# Title resolution
# -----------------------------
def resolve_title(user_input: str) -> int:
    normalized = normalize_text(user_input)

    # Exact normalized match
    if normalized in title_lookup:
        return title_lookup[normalized]

    # Fuzzy fallback
    matches = get_close_matches(
        normalized,
        title_lookup.keys(),
        n=1,
        cutoff=0.75
    )

    if matches:
        return title_lookup[matches[0]]

    raise ValueError("Movie not found in catalog")

# -----------------------------
# Hybrid recommendation
# -----------------------------
def recommend(movie_title: str, top_k: int = 5, return_scores: bool = False):
    """
    Returns movie recommendations using:
    - Cosine similarity (recall)
    - Optional ML ranker (precision)
    """

    # Resolve input title
    idx = resolve_title(movie_title)

    # -------- Recall: cosine similarity --------
    distances, indices = nn.kneighbors(
        vectors[idx].reshape(1, -1),
        n_neighbors=25  # higher recall pool
    )

    candidate_indices = indices.flatten()[1:]
    similarities = 1 - distances.flatten()[1:]

    # -------- Precision: optional ranker --------
    if ranker.available:
        try:
            candidate_df = full_df.iloc[candidate_indices]
            rank_features = build_rank_features(candidate_df)
            rank_scores = ranker.rank(rank_features)

            # Sort by ranker score (descending)
            order = np.argsort(rank_scores)[::-1]
            final_indices = candidate_indices[order][:top_k]

        except Exception as e:
            print(f"[WARN] Ranker failed, falling back to cosine-only: {e}")
            final_indices = candidate_indices[:top_k]
    else:
        final_indices = candidate_indices[:top_k]

    recommendations = [titles[i] for i in final_indices]

    if return_scores:
        return recommendations, similarities[:top_k].tolist()

    return recommendations
