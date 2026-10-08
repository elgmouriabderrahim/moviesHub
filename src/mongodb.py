import pandas as pd
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
import os


ROOT = Path(__file__).resolve().parents[1]
clean_movies_file = ROOT / "data" / "clean" / "movies_clean.csv"
df = pd.read_csv(clean_movies_file)

load_dotenv(ROOT / ".env")
MONGO_URI = os.getenv("MONGO_URI") 

client = MongoClient(MONGO_URI)
db = client["moviesHub_db"]
movies_collection = db["movies"]

movies_collection.delete_many({})
movies_collection.insert_many(df.to_dict(orient="records"))

print(f"{len(df)} movies inserted onto MongoDB.")

# mongodb queries 

# Query 1: English movies
english_movies = movies_collection.find(
    {"original_language": "en"},
    {
        "_id": 0,
        "title": 1,
        "original_language": 1
    }
).limit(5)

print("\nEnglish movies:")
for movie in english_movies:
    print(movie)


# Query 2: Highly rated movies
high_rated_movies = movies_collection.find(
    {
        "vote_average": {"$gte": 8}
    },
    {
        "_id": 0,
        "title": 1,
        "vote_average": 1,
        "vote_count": 1
    }
).limit(5)

print("\nHighly rated movies:")
for movie in high_rated_movies:
    print(movie)


# Query 3: Popular movies
popular_movies = movies_collection.find(
    {},
    {
        "_id": 0,
        "title": 1,
        "popularity": 1
    }
).sort("popularity", -1).limit(5)

print("\nMost popular movies:")
for movie in popular_movies:
    print(movie)


# Aggregation:
# Average rating and number of movies by original language
pipeline = [
    {
        "$match": {
            "vote_count": {"$gt": 0}
        }
    },
    {
        "$group": {
            "_id": "$original_language",
            "average_rating": {"$avg": "$vote_average"},
            "movie_count": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "movie_count": -1
        }
    },
    {
        "$project": {
            "_id": 0,
            "language": "$_id",
            "average_rating": 1,
            "movie_count": 1
        }
    }
]

aggregation_results = movies_collection.aggregate(pipeline)

print("\nMovies grouped by language:")
for result in aggregation_results:
    print(result)