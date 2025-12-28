import joblib
from pathlib import Path

CHAMPION_MODEL_PATH = Path("models/champion/ranker.pkl")

class Ranker:
    def __init__(self):
        self.model = None
        self.available = False
        self._load()

    def _load(self):
        if CHAMPION_MODEL_PATH.exists():
            try:
                self.model = joblib.load(CHAMPION_MODEL_PATH)
                self.available = True
                print("Champion ranker loaded.")
            except Exception as e:
                print(f"Failed to load ranker: {e}")
                self.available = False

    def rank(self, features_df):
        if not self.available:
            return None
        return self.model.predict_proba(features_df)[:, 1]
