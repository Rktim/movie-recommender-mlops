import pandas as pd

TEXT_COLUMNS = ["overview", "genre"]
RANK_FEATURES = [
    "popularity",
    "vote_average",
    "vote_count",
]

def build_rank_features(df):
    return df[RANK_FEATURES].fillna(0)
def build_text_feature(df: pd.DataFrame) -> pd.Series:
    """
    Build a single text feature from movie overview and genre.
    """

    missing = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.fillna("")
    combined = df[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
    return combined
