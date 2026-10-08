from pathlib import Path
import ast
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Load the same feature dataset used in the classification and clustering notebooks.
ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data" / "features" / "movies_features.csv"
MODEL_DIR = ROOT / "models"

df = pd.read_csv(FEATURE_FILE)


# Create the classification target, as in 04_classification.ipynb.
threshold = df["vote_count"].median()

df["high_engagement"] = (df["vote_count"] >= threshold).astype(int)


# Use the same feature columns as 04_classification.ipynb.
num_features = [
    "runtime",
    "budget",
    "revenue",
    "vote_average",
    "release_year",
    "release_month",
    "genre_count",
    "keyword_count",
    "roi",
    "movie_age"
]

categorical_features = [
    "original_language",
    "status",
    "runtime_category"
]

lists_columns = [
    "genres",
    "keywords"
]


def parse_names(value):
    if pd.isna(value):
        return []
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else value
        return [item["name"] for item in items if isinstance(item, dict) and "name" in item]
    except (ValueError, SyntaxError, TypeError):
        return []


df["genres"] = df["genres"].apply(lambda value: " , ".join(parse_names(value)) or "no_genre")
df["keywords"] = df["keywords"].apply(lambda values: " ".join(parse_names(values)))

features = num_features + categorical_features + lists_columns

X = df[features]
y = df["high_engagement"]


# Split the data in the same way as 04_classification.ipynb.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Use the same preprocessing structure as 04_classification.ipynb.
numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_preprocessor, num_features),
    ("categorical", categorical_preprocessor, categorical_features),
    ("genres", CountVectorizer(binary=True, token_pattern=r"[^,]+"), "genres"),
    ("keywords", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "keywords"),
])


# Train the same Random Forest and use the same GridSearchCV settings as the notebook.
random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    random_forest_model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

classification_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "Best parameters": grid_search.best_params_,
    "Best CV F1": grid_search.best_score_,
    "vote_count threshold": threshold
}


# Prepare the clustering data as in 05_clustering.ipynb.
df_cluster = pd.read_csv(FEATURE_FILE)

df_cluster["genres"] = df_cluster["genres"].apply(
    lambda value: ",".join(parse_names(value)) or "no_genre"
)
df_cluster["keywords"] = df_cluster["keywords"].apply(
    lambda values: " ".join(parse_names(values))
)
df_cluster["keywords"] = df_cluster["keywords"].fillna("")
df_cluster["overview"] = df_cluster["overview"].fillna("")

cluster_features = features + ["overview"]

cluster_numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cluster_categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

cluster_preprocessor = ColumnTransformer([
    ("numeric", cluster_numeric_preprocessor, num_features),
    ("categorical", cluster_categorical_preprocessor, categorical_features),
    ("genres", CountVectorizer(binary=True, token_pattern=r"[^,]+"), "genres"),
    ("keywords", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"), "keywords"),
    ("overview", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"), "overview")
])

X_cluster = cluster_preprocessor.fit_transform(df_cluster[cluster_features])


# Test several K values and use the Silhouette Score, as in 05_clustering.ipynb.
k_values = range(2, 11)

silhouette_scores = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_cluster)

    score = silhouette_score(
        X_cluster,
        labels,
        sample_size=min(1000, X_cluster.shape[0]),
        random_state=42
    )

    silhouette_scores.append(score)

best_k = list(k_values)[
    np.argmax(silhouette_scores)
]

best_score = max(silhouette_scores)

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_final.fit_predict(X_cluster)

df_cluster["cluster"] = cluster_labels

feature_names = cluster_preprocessor.get_feature_names_out()
cluster_top_features = {}

for cluster in sorted(df_cluster["cluster"].unique()):

    cluster_indices = np.where(
        df_cluster["cluster"].values == cluster
    )[0]

    cluster_feature_values = X_cluster[cluster_indices]
    mean_values = np.asarray(cluster_feature_values.mean(axis=0)).ravel()
    top_indices = mean_values.argsort()[-10:][::-1]
    cluster_top_features[int(cluster)] = feature_names[top_indices].tolist()

cluster_assignments = df_cluster[["movie_id", "title", "cluster"]]


# Use the overview TF-IDF code from 03_ifidf.ipynb for recommendations.
recommendation_df = pd.read_csv(FEATURE_FILE)
recommendation_df["overview"] = recommendation_df["overview"].fillna("")
recommendation_df["overview"] = recommendation_df["overview"].str.lower()
recommendation_df["overview"] = recommendation_df["overview"].apply(
    lambda x: re.sub(r"[^a-z0-9\s]", " ", x)
)
recommendation_df["overview"] = recommendation_df["overview"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(recommendation_df["overview"])

recommendation_items = recommendation_df[
    [column for column in ["movie_id", "title", "release_year", "genres", "vote_average"] if column in recommendation_df]
].reset_index(drop=True)


# Save the fitted models and preprocessing objects for Streamlit.
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "metrics": classification_metrics
    },
    MODEL_DIR / "classification_pipeline.joblib"
)

joblib.dump(
    {
        "preprocessor": cluster_preprocessor,
        "model": kmeans_final,
        "assignments": cluster_assignments,
        "best_k": int(best_k),
        "silhouette": float(best_score),
        "scores": dict(zip(k_values, silhouette_scores)),
        "feature_names": feature_names,
        "cluster_top_features": cluster_top_features
    },
    MODEL_DIR / "clustering_pipeline.joblib"
)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "matrix": X_tfidf,
        "items": recommendation_items
    },
    MODEL_DIR / "recommendation_pipeline.joblib"
)

print("Classification results:")
print(classification_metrics)
print("Best K:", best_k)
print("Best Silhouette Score:", round(best_score, 4))
print(f"Models saved to: {MODEL_DIR}")
