import json
from pathlib import Path
from datetime import datetime, timedelta

METRICS_PATH = Path("metrics/metrics.json")
METRICS_PATH.parent.mkdir(exist_ok=True)

class MetricsManager:
    def __init__(self):
        self.reset_window()

    def reset_window(self):
        self.window_start = datetime.utcnow()
        self.total_requests = 0
        self.cold_starts = 0
        self.similarity_sum = 0.0
        self.similarity_count = 0

    def record_request(self, cold_start: bool, similarities: list[float]):
        self.total_requests += 1

        if cold_start:
            self.cold_starts += 1

        for s in similarities:
            self.similarity_sum += s
            self.similarity_count += 1

        # ---- NEW: flush every N requests ----
        if self.total_requests % 10 == 0:
            self.flush()

        self.maybe_flush()


    def maybe_flush(self):
        if datetime.utcnow() - self.window_start > timedelta(hours=24):
            self.flush()
            self.reset_window()

    def flush(self):
        metrics = {
            "cold_start_rate": (
                self.cold_starts / self.total_requests
                if self.total_requests else 0.0
            ),
            "mean_similarity": (
                self.similarity_sum / self.similarity_count
                if self.similarity_count else 1.0
            ),
            "query_count_24h": self.total_requests,
            "last_retrain_days": self._read_last_retrain_days(),
        }

        METRICS_PATH.parent.mkdir(exist_ok=True)

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)


    def _read_last_retrain_days(self):
        if not METRICS_PATH.exists():
            return 0

        try:
            with open(METRICS_PATH) as f:
                old = json.load(f)
                return old.get("last_retrain_days", 0)
        except Exception:
            return 0
   