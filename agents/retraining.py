import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

METRICS_PATH = Path("metrics/metrics.json")
MODELS_DIR = Path("models/challengers")
REPORTS_DIR = Path("reports")

REPORTS_DIR.mkdir(exist_ok=True)

# ---- Thresholds (configurable) ----
MAX_COLD_START = 0.20
MIN_SIMILARITY = 0.50
MAX_DAYS_WITHOUT_RETRAIN = 14


def load_metrics():
    if not METRICS_PATH.exists():
        print("Metrics file not found. Initializing defaults.")
        return {
            "cold_start_rate": 0.0,
            "mean_similarity": 1.0,
            "query_count_24h": 0,
            "last_retrain_days": 0,
        }

    with open(METRICS_PATH) as f:
        return json.load(f)


def should_retrain(metrics: dict) -> tuple[bool, list[str]]:
    reasons = []

    if metrics["cold_start_rate"] > MAX_COLD_START:
        reasons.append("High cold-start rate")

    if metrics["mean_similarity"] < MIN_SIMILARITY:
        reasons.append("Low mean similarity")

    if metrics["last_retrain_days"] > MAX_DAYS_WITHOUT_RETRAIN:
        reasons.append("Model too old")

    return len(reasons) > 0, reasons


def trigger_ezyml_training():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"ranker_{timestamp}.pkl"
    report_path = REPORTS_DIR / f"ranker_{timestamp}.json"

    cmd = [
        "ezyml",
        "train",
        "--data", "data/processed/ranking_data.csv",
        "--target", "relevance_score",
        "--model", "extra_trees",
        "--output", str(model_path),
        "--report", str(report_path)
    ]

    result = subprocess.run(
    cmd,
    capture_output=True,
    text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"EZYML training failed:\n{result.stderr}"
        )

    if not model_path.exists():
        raise RuntimeError("Model file was not created")

    if not report_path.exists():
        raise RuntimeError("Report file was not created")

    return model_path, report_path



def run_agent():
    try:
        model_path, report_path = trigger_ezyml_training()
    except Exception as e:
        print("Retraining FAILED.")
        print(str(e))
        return

    print("Retraining completed successfully.")
    print(f"Model saved at: {model_path}")
    print(f"Report saved at: {report_path}")



if __name__ == "__main__":
    run_agent()
