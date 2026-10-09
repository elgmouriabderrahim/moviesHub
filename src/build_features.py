from pathlib import Path
import ast

import pandas as pd
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "data" / "clean" / "movies_clean.csv"

df = pd.read_csv(CLEAN_FILE)

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df["keywords"] = df["keywords"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month
df["release_decade"] = (df["release_year"] // 10) * 10

df["roi"] = df["revenue"] / df["budget"]
df["movie_age"] = datetime.now().year - df["release_year"]

df["genre_count"] = df["genres"].apply(len)
df["keyword_count"] = df["keywords"].apply(len)

df["runtime_category"] = pd.cut(
    df["runtime"],
    bins=[0, 90, 120, 150, float("inf")],
    labels=["Short", "Medium", "Long", "Very Long"],
    include_lowest=True,
)

df["overview_length"] = df["overview"].fillna("").str.len()

FEATURE_DIR = ROOT / "data" / "features"
FEATURE_FILE = FEATURE_DIR / "movies_features.csv"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(FEATURE_FILE, index=False)
print(f"Feature dataset saved to: {FEATURE_FILE}")
