"""Streamlit dashboard for MoviesHub classification and clustering."""

import ast
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data" / "features" / "movies_features.csv"
CLASSIFIER_FILE = ROOT / "models" / "classification_pipeline.joblib"
CLUSTER_FILE = ROOT / "models" / "clustering_pipeline.joblib"
RECOMMENDATION_FILE = ROOT / "models" / "recommendation_pipeline.joblib"
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="MoviesHub", page_icon="🎬", layout="wide")


def parse_names(value):
    if isinstance(value, list):
        items = value
    elif pd.isna(value):
        items = []
    else:
        try:
            items = ast.literal_eval(value) if isinstance(value, str) else []
        except (ValueError, SyntaxError, TypeError):
            items = []
    return [item["name"] for item in items if isinstance(item, dict) and "name" in item]


@st.cache_data(ttl=60)
def load_movies():
    host = os.getenv("MONGO_HOST")
    if host:
        try:
            client = MongoClient(
                host=host,
                port=int(os.getenv("MONGO_PORT", "27017")),
                username=os.getenv("MONGO_INITDB_ROOT_USERNAME"),
                password=os.getenv("MONGO_INITDB_ROOT_PASSWORD"),
                authSource="admin",
                serverSelectionTimeoutMS=2500,
            )
            docs = list(client["moviesHub_db"]["movies"].find({}, {"_id": 0}))
            client.close()
            if docs:
                return pd.DataFrame(docs), "MongoDB"
        except Exception as exc:
            st.warning(f"MongoDB unavailable; using feature CSV. ({exc})")
    elif os.getenv("MONGO_URI"):
        try:
            client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=2500)
            docs = list(client["moviesHub_db"]["movies"].find({}, {"_id": 0}))
            client.close()
            if docs:
                return pd.DataFrame(docs), "MongoDB"
        except Exception as exc:
            st.warning(f"MongoDB unavailable; using feature CSV. ({exc})")

    if FEATURE_FILE.exists():
        return pd.read_csv(FEATURE_FILE), "feature CSV"
    return pd.DataFrame(), "no data source"


def parse_genre_names(value):
    names = parse_names(value)
    if names:
        return names
    if isinstance(value, str):
        return [name.strip() for name in value.split(",") if name.strip()]
    return []


def dashboard_page(movies, source):
    st.title("🎬 MoviesHub dashboard")
    st.caption(f"Movie data source: {source}")
    if movies.empty:
        st.error("No movie data found. Run the Airflow pipeline first.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Movies", f"{len(movies):,}")
    b.metric("Average rating", f"{movies['vote_average'].mean():.2f}" if "vote_average" in movies else "—")
    c.metric("Average popularity", f"{movies['popularity'].mean():.1f}" if "popularity" in movies else "—")
    d.metric("Languages", str(movies["original_language"].nunique()) if "original_language" in movies else "—")

    left, right = st.columns(2)
    if "release_year" in movies:
        yearly = movies.dropna(subset=["release_year"]).groupby("release_year").size().rename("Movies").reset_index()
        left.plotly_chart(px.line(yearly, x="release_year", y="Movies", title="Movies by release year"), use_container_width=True)
    if "vote_average" in movies:
        right.plotly_chart(px.histogram(movies, x="vote_average", nbins=20, title="Vote rating distribution"), use_container_width=True)

    if "genres" in movies:
        genre_counts = movies["genres"].apply(parse_genre_names).explode().dropna().value_counts().head(12)
        if not genre_counts.empty:
            st.plotly_chart(px.bar(genre_counts.sort_values().rename_axis("Genre").reset_index(name="Movies"), x="Movies", y="Genre", orientation="h", title="Most common genres"), use_container_width=True)

    search = st.text_input("Search movies by title")
    display = movies
    if search and "title" in display:
        display = display[display["title"].astype(str).str.contains(search, case=False, na=False)]
    columns = [name for name in ["title", "release_year", "genres", "vote_average", "popularity"] if name in display]
    st.dataframe(display[columns].head(1000), use_container_width=True, hide_index=True)


def classification_page(movies):
    st.title("Engagement classification")
    if not CLASSIFIER_FILE.exists():
        st.info("Classification artifact is missing. Run the Airflow ML task to build it.")
        return
    artifact = joblib.load(CLASSIFIER_FILE)
    model = artifact["model"]
    metrics = artifact["metrics"]
    st.caption("Validation on the training script's held-out split: "
               f"accuracy {metrics['accuracy']:.3f}, F1 {metrics['f1']:.3f}.")
    st.warning("Revenue, vote average, and ROI are post-release information. This model estimates engagement from movie data, not pre-release information.")

    numeric = ["runtime", "budget", "revenue", "vote_average", "release_year", "release_month", "genre_count", "keyword_count", "roi", "movie_age"]
    categorical = ["original_language", "status", "runtime_category"]
    numeric_defaults = {}
    for column in numeric:
        value = pd.to_numeric(movies[column], errors="coerce").median() if column in movies else 0
        numeric_defaults[column] = 0.0 if pd.isna(value) else float(value)
    languages = sorted(movies["original_language"].dropna().astype(str).unique()) if "original_language" in movies else ["en"]
    statuses = sorted(movies["status"].dropna().astype(str).unique()) if "status" in movies else ["Released"]
    runtime_categories = ["Short", "Medium", "Long", "Very Long"]

    with st.form("classification_form"):
        title = st.text_input("Movie title", "New movie")
        cols = st.columns(3)
        values = {}
        for i, feature in enumerate(numeric):
            values[feature] = cols[i % 3].number_input(feature.replace("_", " ").title(), value=numeric_defaults[feature])
        values["original_language"] = cols[0].selectbox("Original language", languages, index=0)
        values["status"] = cols[1].selectbox("Status", statuses, index=0)
        values["runtime_category"] = cols[2].selectbox("Runtime category", runtime_categories, index=2)
        genres = st.text_input("Genres (comma-separated)", "Action,Drama")
        keywords = st.text_input("Keywords (space-separated)", "")
        submitted = st.form_submit_button("Predict engagement")

    if submitted:
        values["genres"] = " , ".join(part.strip() for part in genres.split(",") if part.strip()) or "no_genre"
        values["keywords"] = keywords
        row = pd.DataFrame([{feature: values[feature] for feature in model.feature_names_in_}])
        prediction = int(model.predict(row)[0])
        st.subheader(f"{title}: {'High engagement' if prediction else 'Normal engagement'}")
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row)[0, 1])
            st.progress(probability, text=f"Estimated high-engagement probability: {probability:.1%}")


def clusters_page(movies):
    st.title("Movie clusters")
    if not CLUSTER_FILE.exists():
        st.info("Clustering artifacts are missing. Run the Airflow ML task to build them.")
        return
    artifact = joblib.load(CLUSTER_FILE)
    assignments = artifact["assignments"]
    st.caption(f"K-Means selected K={artifact['best_k']}; sampled silhouette score {artifact['silhouette']:.3f}.")
    counts = assignments["cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Movies")
    st.plotly_chart(px.bar(counts, x="Cluster", y="Movies", title="Movies per cluster"), use_container_width=True)

    if "movie_id" in movies and "movie_id" in assignments:
        details = movies.merge(assignments[["movie_id", "cluster"]], on="movie_id", how="inner")
    else:
        details = assignments
    cluster = st.selectbox("Choose a cluster", sorted(assignments["cluster"].unique()))
    top_features = artifact.get("cluster_top_features", {}).get(int(cluster), [])
    if top_features:
        st.write("Strongest features in this cluster:", ", ".join(top_features))
    shown = details[details["cluster"] == cluster]
    fields = [name for name in ["title", "genres", "vote_average", "release_year", "overview"] if name in shown]
    st.dataframe(shown[fields].head(100), use_container_width=True, hide_index=True)
    feature_names = artifact.get("feature_names", [])
    if feature_names is not None:
        st.caption("This cluster was formed from numeric, categorical, genre, keyword, and overview features.")


def recommendations_page():
    st.title("Movie recommendations")
    if not RECOMMENDATION_FILE.exists():
        st.info("Recommendation artifact is missing. Run the Airflow ML task to build it.")
        return
    artifact = joblib.load(RECOMMENDATION_FILE)
    items = artifact["items"]
    if items.empty:
        st.warning("There are no movies available for recommendations yet.")
        return

    def movie_label(index):
        row = items.iloc[index]
        return f"{row['title']} (ID: {row['movie_id']})"

    selected_index = st.selectbox(
        "Choose a movie you like",
        options=list(range(len(items))),
        format_func=movie_label,
    )
    count = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    if st.button("Find similar movies"):
        similarities = cosine_similarity(
            artifact["matrix"][selected_index],
            artifact["matrix"]
        ).ravel()
        indices = sorted(
            range(len(similarities)),
            key=lambda index: similarities[index],
            reverse=True
        )
        rows = []
        for index in indices:
            if index == selected_index:
                continue
            item = items.iloc[index].to_dict()
            item["Similarity"] = f"{similarities[index]:.0%}"
            rows.append(item)
            if len(rows) == count:
                break
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


movies, source = load_movies()
page = st.sidebar.radio("Section", ["Dashboard", "Classification", "Clusters", "Recommendations"])
if page == "Dashboard":
    dashboard_page(movies, source)
elif page == "Classification":
    classification_page(movies)
elif page == "Clusters":
    clusters_page(movies)
else:
    recommendations_page()
