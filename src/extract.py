from pathlib import Path
import json
import os
import time

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

load_dotenv(ROOT / ".env")

TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")

URL = "https://api.themoviedb.org/3/movie/popular" 

headers = {
    "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}",
    "accept": "application/json",
}

 
def fetch_page(page):
    try:
        response = requests.get(
            URL,
            headers=headers,
            params={"page": page},
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if not data or not data.get("results"):
            print(f"No results found on page {page}")
            return None

        return data

    except requests.RequestException as error:
        print(f"Error on page {page}: {error}")
        return None


def extract_movies(total_pages=50):
    all_movies = []

    for page in range(1, total_pages + 1):
        print(f"Fetching page {page}...")

        data = fetch_page(page)

        if data is None:
            continue

        for movie in data["results"]:
            movie_id = movie["id"]

            details = fetch_movie_details(movie_id)
            keywords_data = fetch_movie_keywords(movie_id)

            if details is None:
                continue

            movie_data = {
                "movie_id": movie_id,
                "title": details.get("title"),
                "overview": details.get("overview"),
                "release_date": details.get("release_date"),
                "runtime": details.get("runtime"),
                "original_language": details.get("original_language"),
                "genres": details.get("genres"),
                "keywords": (
                    keywords_data.get("keywords", [])
                    if keywords_data
                    else []
                ),
                "budget": details.get("budget"),
                "revenue": details.get("revenue"),
                "popularity": details.get("popularity"),
                "vote_average": details.get("vote_average"),
                "vote_count": details.get("vote_count"),
            }

            all_movies.append(movie_data)

            time.sleep(0.2)

    return all_movies


def save_raw_data(movies):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RAW_DIR / "movies.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(movies, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(movies)} movies to {output_file}")


def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        print(f"Error fetching movie {movie_id}: {error}")
        return None



def fetch_movie_keywords(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords"

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        print(f"Error fetching keywords for movie {movie_id}: {error}")
        return None


movies = extract_movies(total_pages=50)

save_raw_data(movies)
