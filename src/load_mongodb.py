from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data" / "features" / "movies_features.csv"

load_dotenv(ROOT / ".env")
MONGO_URI = os.getenv("MONGO_URI") 

client = MongoClient(MONGO_URI)
db = client["moviesHub_db"]
movies_collection = db["movies_features"]

# Update each movie by movie_id so the collection is not cleared on every run.
movies_collection.create_index("movie_id", unique=True)

df = pd.read_csv(FEATURE_FILE)
movies = df.to_dict(orient="records")

operations = [
    UpdateOne(
        {"movie_id": movie["movie_id"]},
        {"$set": movie},
        upsert=True,
    )
    for movie in movies
]

result = movies_collection.bulk_write(operations, ordered=False)

print(f"{result.upserted_count} movies inserted onto MongoDB.")
print(f"{result.modified_count} movies updated onto MongoDB.")

client.close()
