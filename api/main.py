from fastapi import FastAPI, HTTPException
from src.inference import recommend
from src.metrics import MetricsManager

app = FastAPI(title="Movie Recommendation API")
metrics = MetricsManager()

@app.get("/recommend")
def get_recommendations(movie: str, k: int = 5):
    try:
        recs, sims = recommend(movie, k, return_scores=True)

        metrics.record_request(
            cold_start=False,
            similarities=sims
        )

        return {
            "movie": movie,
            "recommendations": recs
        }

    except ValueError:
        metrics.record_request(
            cold_start=True,
            similarities=[]
        )
        raise HTTPException(status_code=404, detail="Movie not found")
