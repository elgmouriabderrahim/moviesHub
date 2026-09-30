from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "movies.json"
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_FILE = CLEAN_DIR / "movies_clean.csv"


df = pd.read_json(RAW_FILE)


# Remove duplicated movies
df = df.drop_duplicates(subset="movie_id")


# Convert release_date to datetime
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)


# Replace empty text values with missing values
text_columns = [
    "title",
    "overview",
    "original_language",
]

for column in text_columns:
    df[column] = df[column].replace("", pd.NA)


# Treat 0 as missing for fields where 0 represents unavailable information
df["runtime"] = df["runtime"].replace(0, pd.NA)
df["budget"] = df["budget"].replace(0, pd.NA)
df["revenue"] = df["revenue"].replace(0, pd.NA)


# Replace empty genre and keyword lists with missing values
df["genres"] = df["genres"].apply(
    lambda x: pd.NA if len(x) == 0 else x
)

df["keywords"] = df["keywords"].apply(
    lambda x: pd.NA if len(x) == 0 else x
)


# Separate column types
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
    "genres",
]

text_columns = [
    "title",
    "overview",
    "keywords",
]


# Final checks
print("Final shape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicated movie IDs:")
print(df["movie_id"].duplicated().sum())

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nText columns:")
print(text_columns)

print("\nData types:")
print(df.dtypes)


# Save cleaned dataset
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(
    CLEAN_FILE,
    index=False
)

print(f"\nCleaned data saved to: {CLEAN_FILE}")