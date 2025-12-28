import random
import time
import requests

API_URL = "http://127.0.0.1:8000/recommend"

# Known movies from your dataset (add more if you want)
KNOWN_MOVIES = [
    "Iron Man",
    "ironman",
    "Iron-Man",
    "IRON_MAN",
    "Avengers",
    "The Dark Knight",
    "Inception",
    "Interstellar",
]

# Intentionally bad / cold-start inputs
UNKNOWN_MOVIES = [
    "ironman 7",
    "random_movie_123",
    "unknown film",
    "some bollywood movie",
    "test_movie_xyz",
]

TOTAL_REQUESTS = 200
SLEEP_BETWEEN = 0.05  # seconds


def send_request(movie):
    params = {"movie": movie, "k": 5}
    try:
        r = requests.get(API_URL, params=params, timeout=3)
        return r.status_code
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def run_test():
    print("Starting inference load test...")
    print(f"Total requests: {TOTAL_REQUESTS}")

    for i in range(TOTAL_REQUESTS):
        # 65% valid, 35% invalid → forces cold-start rate up
        if random.random() < 0.65:
            movie = random.choice(KNOWN_MOVIES)
        else:
            movie = random.choice(UNKNOWN_MOVIES)

        status = send_request(movie)

        print(f"[{i+1}/{TOTAL_REQUESTS}] movie='{movie}' → status={status}")

        time.sleep(SLEEP_BETWEEN)

    print("Load test completed.")
    print("Check metrics/metrics.json")
    print("Then run: python agents/retraining.py")


if __name__ == "__main__":
    run_test()
