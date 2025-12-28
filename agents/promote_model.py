import json
import shutil
from pathlib import Path

MODELS_DIR = Path("models")
CHAMPION_DIR = MODELS_DIR / "champion"
CHALLENGER_DIR = MODELS_DIR / "challengers"
REPORTS_DIR = Path("reports")

CHAMPION_DIR.mkdir(exist_ok=True)
CHALLENGER_DIR.mkdir(exist_ok=True)

CHAMPION_MODEL = CHAMPION_DIR / "ranker.pkl"


def load_report(report_path: Path) -> dict:
    with open(report_path) as f:
        return json.load(f)


def extract_score(report: dict) -> float:
    """
    Prefer F1, fallback to accuracy.
    """
    metrics = report.get("metrics", report)

    if "f1_score" in metrics:
        return metrics["f1_score"]
    if "accuracy" in metrics:
        return metrics["accuracy"]

    raise ValueError("No usable metric found in report")


def promote(challenger_model: Path, challenger_report: Path):
    challenger_score = extract_score(load_report(challenger_report))

    if CHAMPION_MODEL.exists():
        champion_report = REPORTS_DIR / "champion_report.json"
        if not champion_report.exists():
            raise RuntimeError("Champion report missing — cannot compare")

        champion_score = extract_score(load_report(champion_report))

        print(f"Champion score:   {champion_score}")
        print(f"Challenger score: {challenger_score}")

        if challenger_score <= champion_score:
            print("❌ Challenger rejected (no improvement)")
            return False
    else:
        print("No champion exists — auto-promoting first model")

    # ---- Atomic promotion ----
    backup = None
    if CHAMPION_MODEL.exists():
        backup = CHAMPION_DIR / "ranker_backup.pkl"
        shutil.copy2(CHAMPION_MODEL, backup)

    shutil.copy2(challenger_model, CHAMPION_MODEL)
    shutil.copy2(challenger_report, REPORTS_DIR / "champion_report.json")

    print("✅ Challenger promoted to champion")

    return True


if __name__ == "__main__":
    # Example usage
    challenger_model = max(CHALLENGER_DIR.glob("ranker_*.pkl"), key=lambda p: p.stat().st_mtime)
    challenger_report = REPORTS_DIR / challenger_model.with_suffix(".json").name

    promote(challenger_model, challenger_report)
