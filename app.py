from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="NLP Project 2",
    page_icon="🧠",
    layout="wide",
)

DATA_PATH = Path(__file__).with_name("dataset_cleaned.csv")
SEED = 42
N_CLUSTERS = 8
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "have",
    "very",
    "your",
    "you",
    "are",
    "but",
    "was",
    "were",
    "from",
    "they",
    "their",
    "them",
    "not",
    "all",
    "too",
    "has",
    "had",
    "our",
    "out",
    "can",
    "will",
    "would",
    "could",
    "there",
    "been",
    "about",
    "after",
    "before",
    "more",
    "less",
    "into",
    "when",
    "where",
    "what",
    "which",
    "also",
    "only",
    "just",
    "because",
    "still",
    "than",
    "then",
    "some",
    "much",
    "even",
    "being",
    "it's",
    "im",
    "dont",
    "didnt",
    "insurance",
    "assurance",
    "company",
    "service",
    "client",
    "customer",
}
STOPWORDS = STOPWORDS.union(set(ENGLISH_STOP_WORDS))
DOMAIN_STOPWORDS_EN = {
    "insurance",
    "insurer",
    "insurers",
    "contract",
    "service",
    "services",
    "customer",
    "customers",
    "company",
    "companies",
    "claim",
    "claims",
    "case",
    "file",
    "website",
    "phone",
    "advisor",
    "advisors",
}
NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "cannot",
    "without",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "won't",
    "wont",
    "wouldn't",
    "wouldnt",
    "shouldn't",
    "shouldnt",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
}
STRONG_NEGATIVE_WORDS = {
    "horrible",
    "terrible",
    "awful",
    "worst",
    "hate",
    "expensive",
    "disappointed",
    "useless",
    "bad",
    "problem",
    "refund",
    "refused",
    "delay",
    "delays",
    "complaint",
    "scam",
}
NEGATIVE_PHRASES = {
    "do not recommend",
    "not recommend",
    "very disappointed",
    "too expensive",
}
TOPIC_GENERIC_WORDS = {
    "satisfied",
    "good",
    "recommend",
    "thank",
    "thanks",
    "fast",
    "quick",
    "simple",
    "correct",
    "perfect",
    "friendly",
    "pleasant",
    "responsive",
    "efficient",
    "great",
    "best",
    "hello",
    "direct",
}


def clean_user_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def clean_english_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_english_for_tfidf_or_lda(text: str) -> str:
    cleaned = clean_english_text(text)
    tokens = [
        token
        for token in cleaned.split()
        if (
            ((token not in ENGLISH_STOP_WORDS) or (token in NEGATION_WORDS))
            and token not in DOMAIN_STOPWORDS_EN
            and len(token) > 1
        )
    ]
    return " ".join(tokens)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prepare_english_for_embeddings_or_bert(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("’", "'")
    text = re.sub(r"\n+", " ", text)
    text = normalize_spaces(text)
    return text.strip()


def mean_embedding(tokens: list[str], model: Word2Vec) -> np.ndarray:
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def weighted_embedding(tokens: list[str], model: Word2Vec, idf_map: dict[str, float]) -> np.ndarray:
    weighted_vectors = []
    weights = []
    for token in tokens:
        if token in model.wv:
            weight = float(idf_map.get(token, 1.0))
            weighted_vectors.append(model.wv[token] * weight)
            weights.append(weight)

    if not weighted_vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return (np.sum(weighted_vectors, axis=0) / max(np.sum(weights), 1e-8)).astype(np.float32)


def extract_cluster_keywords_from_clean_texts(
    df_topic: pd.DataFrame,
    cluster_col: str,
    text_col: str,
    top_n: int = 6,
) -> dict[int, list[str]]:
    cluster_docs = (
        df_topic.groupby(cluster_col)[text_col]
        .apply(lambda texts: " ".join(t for t in texts if isinstance(t, str)))
        .sort_index()
    )

    if cluster_docs.empty:
        return {}

    vectorizer = CountVectorizer(max_features=5000, min_df=1, max_df=0.9)
    matrix = vectorizer.fit_transform(cluster_docs.values)
    feature_names = np.array(vectorizer.get_feature_names_out())
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log((1 + len(cluster_docs)) / (1 + doc_freq)) + 1.0
    weighted = matrix.multiply(idf).tocsr()

    keywords: dict[int, list[str]] = {}
    for row_idx, cluster_id in enumerate(cluster_docs.index):
        row_scores = np.asarray(weighted[row_idx].todense()).ravel()
        top_indices = row_scores.argsort()[::-1]
        terms = [feature_names[idx] for idx in top_indices if row_scores[idx] > 0][:top_n]
        keywords[int(cluster_id)] = terms if terms else ["topic", "mixed"]

    return keywords


def extract_topic_terms_from_texts(texts: list[str], top_n: int = 6) -> list[str]:
    words: list[str] = []
    for text in texts:
        words.extend(str(text).split())

    filtered = [
        word
        for word in words
        if len(word) > 2
        and word not in TOPIC_GENERIC_WORDS
        and word not in DOMAIN_STOPWORDS_EN
        and word not in ENGLISH_STOP_WORDS
    ]
    if not filtered:
        filtered = [word for word in words if len(word) > 2]
    if not filtered:
        return ["topic", "mixed"]

    return [word for word, _ in Counter(filtered).most_common(top_n)]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["note"] = pd.to_numeric(df["note"], errors="coerce")
    df["avis_en_tfidf_lda"] = df["avis_en_tfidf_lda"].fillna("").astype(str)
    df["avis_en_embeddings"] = df["avis_en_embeddings"].fillna("").astype(str)
    df["avis"] = df["avis"].fillna("").astype(str)
    df["assureur"] = df["assureur"].fillna("Unknown").astype(str)
    df["produit"] = df["produit"].fillna("Unknown").astype(str)
    df["type"] = df["type"].fillna("unknown").astype(str)
    df["date_publication"] = pd.to_datetime(df["date_publication"], errors="coerce")
    return df


@st.cache_resource(show_spinner=True)
def train_star_rating_model():
    df = load_dataset()
    train_df = df[(df["type"] == "train") & (df["avis_en_tfidf_lda"].str.strip() != "")].copy()

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train_df["avis_en_tfidf_lda"], train_df["note"])

    predictions = np.clip(np.rint(model.predict(train_df["avis_en_tfidf_lda"])), 1, 5).astype(int)
    y_true = train_df["note"].round().astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "mae": mean_absolute_error(train_df["note"], model.predict(train_df["avis_en_tfidf_lda"])),
        "rmse": mean_squared_error(train_df["note"], model.predict(train_df["avis_en_tfidf_lda"])) ** 0.5,
        "macro_f1": f1_score(y_true, predictions, average="macro"),
    }
    return model, metrics


@st.cache_resource(show_spinner=True)
def train_topic_model():
    df = load_dataset()
    df_topic = df[
        (df["avis_en_embeddings"].str.strip() != "") & (df["avis_en_tfidf_lda"].str.strip() != "")
    ].copy()
    tokenized = [text.split() for text in df_topic["avis_en_tfidf_lda"]]

    w2v_model = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=SEED,
    )

    tfidf_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_vectorizer.fit(df_topic["avis_en_tfidf_lda"])
    idf_map = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

    embeddings = np.array(
        [weighted_embedding(tokens, w2v_model, idf_map) for tokens in tokenized],
        dtype=np.float32,
    )
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    df_topic["cluster_kmeans"] = clusters

    topic_keywords = extract_cluster_keywords_from_clean_texts(
        df_topic,
        cluster_col="cluster_kmeans",
        text_col="avis_en_tfidf_lda",
        top_n=6,
    )
    topic_labels = {
        cluster_id: " / ".join(keywords[:3]).title()
        for cluster_id, keywords in topic_keywords.items()
    }

    distances = cdist(embeddings, kmeans.cluster_centers_, metric="euclidean")
    representative_examples: dict[int, list[str]] = {}
    for cluster_id in sorted(df_topic["cluster_kmeans"].unique()):
        cluster_mask = df_topic["cluster_kmeans"] == cluster_id
        cluster_indices = np.where(cluster_mask.to_numpy())[0]
        cluster_distances = distances[cluster_indices, cluster_id]
        closest_indices = cluster_indices[np.argsort(cluster_distances)[:3]]
        representative_examples[int(cluster_id)] = df_topic.iloc[closest_indices]["avis"].tolist()

    metrics = {
        "silhouette": silhouette_score(embeddings, clusters) if len(np.unique(clusters)) > 1 else np.nan,
        "cluster_sizes": df_topic["cluster_kmeans"].value_counts().sort_index(),
        "topic_labels": topic_labels,
        "topic_keywords": topic_keywords,
        "topic_examples": representative_examples,
        "idf_map": idf_map,
        "embedding_matrix": embeddings,
        "cluster_assignments": df_topic["cluster_kmeans"].to_numpy(),
        "clean_texts": df_topic["avis_en_tfidf_lda"].tolist(),
    }
    return w2v_model, kmeans, tfidf_vectorizer, metrics


@st.cache_resource(show_spinner=True)
def load_sentiment_pipeline():
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            info = {
                "status": "error",
                "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
                "note": "PyTorch is not installed. The DistilBERT model is disabled.",
            }
            return None, info

        from transformers import pipeline

        clf = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
        )
        info = {
            "status": "ok",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "note": "DistilBERT pipeline loaded successfully.",
        }
        return clf, info
    except Exception as exc:
        info = {
            "status": "error",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "note": f"Unable to load DistilBERT: {exc}",
        }
        return None, info


def predict_star_rating(text: str) -> int:
    model, _ = train_star_rating_model()
    score = float(model.predict([text])[0])
    lowered = text.lower()
    negative_hits = sum(word in lowered.split() for word in STRONG_NEGATIVE_WORDS)
    phrase_hits = sum(phrase in lowered for phrase in NEGATIVE_PHRASES)

    if phrase_hits >= 1 and negative_hits >= 2:
        return 1
    if negative_hits >= 4:
        return 1
    if negative_hits >= 2 and score <= 2.6:
        return 1

    return int(np.clip(np.rint(score), 1, 5))


def predict_topic(text: str) -> tuple[int, str, list[str]]:
    w2v_model, kmeans, tfidf_vectorizer, metrics = train_topic_model()
    tokens = text.split()
    idf_map = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    vector = weighted_embedding(tokens, w2v_model, idf_map).astype(np.float32).reshape(1, -1)
    centroid_cluster = int(kmeans.predict(vector)[0])

    similarities = cosine_similarity(vector, metrics["embedding_matrix"])[0]
    nearest_idx = similarities.argsort()[::-1][:15]
    nearest_clusters = metrics["cluster_assignments"][nearest_idx]
    cluster_votes = Counter(int(cluster) for cluster in nearest_clusters)
    cluster_id = cluster_votes.most_common(1)[0][0] if cluster_votes else centroid_cluster

    local_texts = [metrics["clean_texts"][idx] for idx in nearest_idx[:10]]
    keywords = extract_topic_terms_from_texts(local_texts, top_n=6)
    label = " / ".join(keywords[:3]).title() if keywords else metrics["topic_labels"].get(cluster_id, f"Topic {cluster_id}")
    return cluster_id, label, keywords


def predict_sentiment(text: str) -> tuple[str, float, str]:
    sentiment_pipe, info = load_sentiment_pipeline()
    if sentiment_pipe is None:
        return "Unavailable", 0.0, info["note"]

    result = sentiment_pipe(text[:512])[0]
    label = result["label"].upper()
    sentiment = "Positive" if "POSITIVE" in label or "LABEL_1" in label else "Negative"
    confidence = float(result["score"])
    return sentiment, confidence, info["model_name"]


def render_prediction_tab():
    st.subheader("Review Prediction")
    st.write(
        "Enter a customer review to get the predicted rating, the dominant topic, "
        "and the estimated sentiment."
    )

    default_text = (
        "The advisor was helpful and the price is fair, "
        "but the website is still a bit difficult to use."
    )
    user_text = st.text_area("Review", value=default_text, height=160)

    if st.button("Run analysis", type="primary"):
        text = clean_user_text(user_text)
        if not text:
            st.warning("Please enter a review before running the analysis.")
            return

        star_input = prepare_english_for_tfidf_or_lda(text)
        topic_input = prepare_english_for_embeddings_or_bert(text)
        sentiment_input = prepare_english_for_embeddings_or_bert(text)

        predicted_rating = predict_star_rating(star_input)
        cluster_id, topic_label, topic_keywords = predict_topic(topic_input)
        sentiment, confidence, sentiment_model_name = predict_sentiment(sentiment_input)

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted rating", f"{predicted_rating}/5")
        col2.metric("Dominant topic", f"Cluster {cluster_id}")
        col3.metric("Sentiment", sentiment, f"{confidence:.1%}")

        st.markdown(f"**Topic label:** {topic_label}")
        st.markdown(f"**Topic keywords:** {', '.join(topic_keywords) if topic_keywords else 'Not available'}")
        st.caption(f"Processed text for rating model: `{star_input}`")
        st.caption(f"Processed text for topic/sentiment models: `{topic_input}`")
        st.caption(
            "Sentiment powered by DistilBERT. "
            f"Loaded model: `{sentiment_model_name}`"
        )


def render_dataset_tab():
    st.subheader("Dataset Overview")
    df = load_dataset()
    train_df = df[df["type"] == "train"].copy()
    _, star_metrics = train_star_rating_model()
    _, _, _, topic_metrics = train_topic_model()

    total_reviews = len(df)
    mean_rating = train_df["note"].mean()
    mean_length = df["longueur_avis"].mean()
    train_share = (df["type"].eq("train").mean()) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of reviews", f"{total_reviews}")
    col2.metric("Average rating", f"{mean_rating:.2f}/5")
    col3.metric("Average length", f"{mean_length:.0f} words")
    col4.metric("Train split", f"{train_share:.1f}%")

    left, right = st.columns(2)

    with left:
        st.markdown("**Rating distribution**")
        rating_counts = train_df["note"].round().astype(int).value_counts().sort_index()
        st.bar_chart(rating_counts)

        st.markdown("**Top 10 insurers**")
        insurer_counts = df["assureur"].value_counts().head(10)
        st.bar_chart(insurer_counts)

    with right:
        st.markdown("**Product distribution**")
        product_counts = df["produit"].value_counts()
        st.bar_chart(product_counts)

        st.markdown("**K-Means cluster sizes**")
        st.bar_chart(topic_metrics["cluster_sizes"])

    if df["date_publication"].notna().any():
        st.markdown("**Review volume over time**")
        monthly_reviews = (
            df.dropna(subset=["date_publication"])
            .assign(mois=lambda x: x["date_publication"].dt.to_period("M").astype(str))
            .groupby("mois")
            .size()
        )
        st.line_chart(monthly_reviews)

    st.markdown("**Selected model performance**")
    perf_df = pd.DataFrame(
        [
            {
                "Method": "Star rating - TF-IDF + Ridge",
                "Metric 1": f"Rounded accuracy: {star_metrics['accuracy']:.3f}",
                "Metric 2": f"Macro F1: {star_metrics['macro_f1']:.3f}",
            },
            {
                "Method": "Topic modelling - Embeddings + K-Means",
                "Metric 1": f"Silhouette: {topic_metrics['silhouette']:.3f}",
                "Metric 2": f"Clusters: {N_CLUSTERS}",
            },
            {
                "Method": "Sentiment analysis - DistilBERT",
                "Metric 1": "Validation accuracy: 0.910",
                "Metric 2": "Validation macro F1: 0.910",
            },
        ]
    )
    st.dataframe(perf_df, hide_index=True)

    st.markdown("**Dataset sample**")
    st.dataframe(
        df[["note", "assureur", "produit", "avis"]].head(10),
        hide_index=True,
    )


def render_conclusion_tab():
    st.subheader("Project Conclusions")
    _, star_metrics = train_star_rating_model()
    _, _, _, topic_metrics = train_topic_model()
    _, sentiment_info = load_sentiment_pipeline()

    st.markdown(
        """
        ### 1. Rating prediction
        The selected method for rating prediction is **TF-IDF + Ridge**.
        This approach remains simple, fast to train, and effective for capturing
        the overall intensity expressed in customer reviews.

        ### 2. Topic modelling
        For thematic exploration, we retain **Embeddings + K-Means**.
        Reviews are projected into a vector space using average Word2Vec embeddings,
        then grouped with K-Means to identify recurring topic families in customer feedback.

        ### 3. Sentiment analysis
        For sentiment analysis, the best selected model is **DistilBERT**.
        It is the strongest model in our experiments for distinguishing
        positive and negative reviews while remaining lighter than a full BERT model.

        ### 4. Overall conclusion
        The project shows that a combination of classical and modern methods
        can cover several business needs: predicting a rating, summarizing the
        main themes in the corpus, and estimating the overall sentiment of a review.
        This Streamlit application brings these three components together in a single,
        readable, and directly usable interface.
        """
    )

    st.markdown("**Numerical summary**")
    st.write(
        f"- Star rating: rounded accuracy {star_metrics['accuracy']:.3f}, "
        f"macro F1 {star_metrics['macro_f1']:.3f}"
    )
    st.write(
        f"- Topic modelling: K-Means silhouette {topic_metrics['silhouette']:.3f}, "
        f"{N_CLUSTERS} interpretable clusters"
    )
    st.write(
        "- Sentiment: DistilBERT selected as the best model, "
        "with validation accuracy 0.910 and macro F1 0.910"
    )
    st.caption(f"Sentiment loading info: {sentiment_info['note']}")

    with st.expander("Topics detected by Embeddings + K-Means"):
        for cluster_id, label in topic_metrics["topic_labels"].items():
            st.markdown(f"**Cluster {cluster_id}**: {label}")
            examples = topic_metrics["topic_examples"].get(cluster_id, [])
            for example in examples:
                st.write(f"- {example}")


def main():
    st.title("Streamlit Application - NLP Project 2")
    st.write(
        "This application brings together the three selected methods from the project: "
        "**TF-IDF + Ridge** for rating prediction, **Embeddings + K-Means** for topic modelling, "
        "and **DistilBERT** for sentiment analysis."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Review prediction", "Dataset overview", "Conclusions"]
    )

    with tab1:
        render_prediction_tab()

    with tab2:
        render_dataset_tab()

    with tab3:
        render_conclusion_tab()


if __name__ == "__main__":
    main()
