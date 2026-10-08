from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "movies.json"
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_FILE = CLEAN_DIR / "movies_clean.csv"


df = pd.read_json(RAW_FILE)
df = df.drop_duplicates(subset="movie_id")
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

text_columns = [
    "overview",
    "tagline",
]
for column in text_columns:
    df[column] = df[column].replace("", pd.NA)

    numeric_columns = [
    "runtime",
    "budget",
    "revenue",
]
for column in numeric_columns:
        df[column] = df[column].replace(0, pd.NA)

        list_columns = [
    "genres",
    "keywords",
    "production_companies",
    "production_countries"
]
for column in list_columns:
    df[column] = df[column].apply(
        lambda x: pd.NA
        if isinstance(x, list) and len(x) == 0
        else x
    )

numerical_columns = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "vote_count",
]

categorical_columns = [
    "original_language",
    "status",
    "adult",
    "genres",
    "production_companies",
    "production_countries",
    "belongs_to_collection",
]

text_columns = [
    "title",
    "overview",
    "tagline",
    "keywords",
]

metadata_columns = [
    "movie_id",
    "poster_path",
    "backdrop_path",
    "release_date",
]

CLEAN_DIR.mkdir(parents=True,exist_ok=True)
df.to_csv(CLEAN_FILE,index=False)
print(f"\nCleaned data saved to: {CLEAN_FILE}")