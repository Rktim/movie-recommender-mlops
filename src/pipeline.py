from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors

def create_recommender_pipeline():
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2)
                ),
            ),
            ("normalize", Normalizer()),
            (
                "nn",
                NearestNeighbors(
                    metric="cosine",
                    algorithm="brute",
                    n_neighbors=10
                ),
            ),
        ]
    )
    return pipeline
