from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "movies.json"
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_FILE = CLEAN_DIR / "movies_clean.csv"


df = pd.read_json(RAW_FILE)

df = df.drop_duplicates(subset="movie_id")

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")


text_columns_to_clean = [
    "title",
    "original_title",
    "overview",
    "tagline",
    "original_language",
    "status",
    "homepage",
    "poster_path",
    "backdrop_path",
]
for column in text_columns_to_clean:
    df[column] = df[column].replace("", pd.NA)


numeric_missing_columns = [
    "runtime",
    "budget",
    "revenue",
]

for column in numeric_missing_columns:
    if column in df.columns:
        df[column] = df[column].replace(0, pd.NA)


list_columns = [
    "genres",
    "keywords",
    "production_companies",
    "production_countries",
    "spoken_languages",
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
    "spoken_languages",
    "belongs_to_collection",
]

text_columns = [
    "title",
    "original_title",
    "overview",
    "tagline",
    "keywords",
]

metadata_columns = [
    "movie_id",
    "homepage",
    "poster_path",
    "backdrop_path",
    "release_date",
]

df["runtime"] = df["runtime"].astype("Int64")
df["budget"] = df["budget"].astype("Int64")
df["revenue"] = df["revenue"].astype("Int64")
 

# Final checks
print("Final shape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nText columns:")
print(text_columns)

print("\nData types:")
print(df.dtypes)

print("\nNegative runtime:")
print((df["runtime"] < 0).sum())

print("\nNegative budget:")
print((df["budget"] < 0).sum())

print("\nNegative revenue:")
print((df["revenue"] < 0).sum())

print("\nInvalid vote_average:")
print(((df["vote_average"] < 0) | (df["vote_average"] > 10)).sum())

print("\nNegative vote_count:")
print((df["vote_count"] < 0).sum())


# Save cleaned dataset
CLEAN_DIR.mkdir(parents=True,exist_ok=True)

df.to_csv(CLEAN_FILE,index=False)

print(f"\nCleaned data saved to: {CLEAN_FILE}")