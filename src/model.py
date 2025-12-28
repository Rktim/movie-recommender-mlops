import joblib
import pandas as pd
from pathlib import Path

from features import build_text_feature
from pipeline import create_recommender_pipeline
from utils import normalize_text

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------
# Training Function
# -------------------------------
def train_model(data_path: Path):
    df = pd.read_csv(data_path)

    # Build text features
    text_features = build_text_feature(df)

    # Create pipeline
    pipeline = create_recommender_pipeline()

    # Fit TF-IDF + Normalizer
    vectors = pipeline[:-1].fit_transform(text_features)

    # Fit Nearest Neighbors on cached vectors
    pipeline.named_steps["nn"].fit(vectors)

    # Build normalized title lookup
    title_lookup = {
        normalize_text(title): idx
        for idx, title in enumerate(df["title"].astype(str).values)
    }

    # Save model artifact
    artifact = {
    "pipeline": pipeline,
    "vectors": vectors,
    "titles": df["title"].values,
    "title_lookup": title_lookup,

    # ---- ADD THIS ----
    "full_df": df[
        [
            "title",
            "popularity",
            "vote_average",
            "vote_count",
        ]
    ].reset_index(drop=True),
}


    joblib.dump(artifact, MODEL_DIR / "pipeline.joblib")

    print("Model training complete.")
    print(f"Movies indexed: {len(df)}")
    print(f"Vector dimension: {vectors.shape[1]}")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    train_model(DATA_PATH)
