# =============================================================================
# CONTENT-BASED MOVIE RECOMMENDATION SYSTEM
# Dataset: IMDB Genres (jquigl/imdb-genres)
# Method: TF-IDF + Cosine Similarity (No pre-trained embeddings)
# =============================================================================


# =============================================================================
# SECTION 1: LOAD DATASET
# =============================================================================

from datasets import load_dataset

# Load the IMDB genres dataset with train/validation/test splits
dataset = load_dataset("jquigl/imdb-genres")

train_data = dataset["train"]
val_data   = dataset["validation"]
test_data  = dataset["test"]

# Quick peek at the raw data structure
print(train_data[0])


# =============================================================================
# SECTION 2: CONVERT TO DATAFRAME
# =============================================================================

import pandas as pd

df     = train_data.to_pandas()
val_df = val_data.to_pandas()

print("Train shape:", df.shape)
df.head(3)


# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (TRAIN SET)
# =============================================================================

# Check for nulls
print("Null counts:\n", df.isnull().sum())

# Check for duplicates
print("\nDuplicates:", df.duplicated().sum())

# Check for string 'NaN' values in rating
print("String NaN in rating:", (df['rating'] == 'NaN').sum())

# Dataset info
df.info()


# =============================================================================
# SECTION 4: CLEAN TRAIN DATA
# =============================================================================

# Remove duplicate rows
df = df.drop_duplicates()
print("After dedup shape:", df.shape)

# Split 'movie title - year' column into separate title and year columns
df['title'] = df['movie title - year'].str.split(' -').str[0]
df['year']  = df['movie title - year'].str.split(' -').str[1]

# Combine description + genre + expanded-genres into a single 'tags' column
# This is the core text field used for TF-IDF similarity
df['tags'] = df['description'] + df['genre'] + df['expanded-genres']

# Keep only the columns needed for the recommender
movies = df[['title', 'tags', 'rating', 'year', 'genre']].copy()

print("Sample tags before cleaning:")
print(movies['tags'][1])


# =============================================================================
# SECTION 5: TEXT PREPROCESSING (TRAIN SET)
# =============================================================================

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


# ── Fixed stemming function ──────────────────────────────────────────────────
# BUG FIX: original function called ps.stem(i) but never appended result to y
# The stemmed word must be collected into the list before joining
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))   # ← append the stemmed word (was missing before)
    return " ".join(y)


# ── Apply full preprocessing pipeline ───────────────────────────────────────
movies['tags'] = (
    movies['tags']
    .fillna("")                                     # handle nulls
    .str.lower()                                    # lowercase everything
    .str.replace(r'[^\w\s]', '', regex=True)        # remove punctuation
    .str.replace(r'\s+', ' ', regex=True)           # collapse extra spaces
    .apply(lambda x: " ".join(                      # remove stopwords
        [w for w in x.split() if w not in stop_words]
    ))
    .apply(stem)                                    # stem words (Porter)
)

print("Sample tags after cleaning:")
print(movies['tags'].iloc[0])


# =============================================================================
# SECTION 6: TF-IDF VECTORIZATION (FIT ON TRAIN SET ONLY)
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

# Fit TF-IDF on the train set tags
# max_features: vocabulary cap; min_df: ignore very rare terms; stop_words: extra safety
tf = TfidfVectorizer(
    max_features=15000,
    min_df=5,
    stop_words='english'
)

vectors = tf.fit_transform(movies['tags'])

print("TF-IDF matrix shape:", vectors.shape)
print("Sample features:", tf.get_feature_names_out()[1000:1010])


# =============================================================================
# SECTION 7: ASSIGN MOVIE IDs (TRAIN SET)
# =============================================================================

# Assign sequential IDs to each movie (used internally for indexing)
movies['movie_id'] = range(len(movies))

print(movies[['movie_id', 'title', 'tags']].head(3))


# =============================================================================
# SECTION 8: RECOMMENDATION FUNCTION
# =============================================================================

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def recommend_movies(input_text, top_k=5):
    """
    Recommend top_k movies similar to input_text.

    Accepts:
      - A movie title (exact match → uses that movie's TF-IDF vector)
      - Any free text (genre, description keywords → converts to TF-IDF vector)

    Returns:
      - List of top_k recommended movie titles from the train set
    """
    input_text_lower = input_text.lower()

    # Step 1: Try exact title match in the train set
    matched = movies[movies['title'].str.lower() == input_text_lower]

    if not matched.empty:
        # Found exact title → reuse its pre-computed vector
        movie_idx    = matched.index[0]
        input_vector = vectors[movie_idx]
        ignore_idx   = {movie_idx}          # exclude the query movie itself
    else:
        # Not a title → treat as free-text keywords and vectorize on the fly
        input_vector = tf.transform([input_text])
        ignore_idx   = set()

    # Step 2: Compute cosine similarity against all train movies
    sim_scores = cosine_similarity(input_vector, vectors).flatten()

    # Step 3: Sort indices from highest to lowest similarity
    sorted_indices = np.argsort(-sim_scores)

    # Step 4: Collect top_k unique movies, skipping ignored index
    recommended = []
    seen_titles  = set()
    for idx in sorted_indices:
        title = movies.iloc[idx]['title']
        if idx not in ignore_idx and title not in seen_titles:
            recommended.append(title)
            seen_titles.add(title)
        if len(recommended) >= top_k:
            break

    return recommended


# =============================================================================
# SECTION 9: PREPROCESS VALIDATION SET
# =============================================================================
# IMPORTANT: Apply the EXACT same preprocessing pipeline used on train.
# The TF-IDF vectorizer is NOT re-fitted — only transform() is used on val.

val_df = val_data.to_pandas()
val_df = val_df.drop_duplicates()

# Extract title and year
val_df['title'] = val_df['movie title - year'].str.split(' -').str[0]
val_df['year']  = val_df['movie title - year'].str.split(' -').str[1]

# Build tags the same way as train
val_df['tags'] = val_df['description'] + val_df['genre'] + val_df['expanded-genres']

# Apply the same cleaning + stemming pipeline
val_df['tags'] = (
    val_df['tags']
    .fillna("")
    .str.lower()
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .apply(lambda x: " ".join(
        [w for w in x.split() if w not in stop_words]
    ))
    .apply(stem)
)

val_movies = val_df[['title', 'tags', 'rating', 'year', 'genre']].copy()
val_movies['movie_id'] = range(len(val_movies))

print("Validation set shape:", val_movies.shape)
val_movies.head(3)


# =============================================================================
# SECTION 10: EVALUATION — PRECISION@K ON VALIDATION SET
# =============================================================================
# Strategy:
#   For each movie in the validation set:
#     1. Use its preprocessed tags as a query to recommend_movies()
#     2. Check how many of the top-K results share at least one genre
#        with the query movie (genre overlap = "relevant")
#   Aggregate into Mean Precision@K across all validation movies


def get_genres(genre_str):
    """Parse a comma-separated genre string into a set of lowercase genres."""
    if pd.isna(genre_str):
        return set()
    return set(g.strip().lower() for g in str(genre_str).split(','))


def precision_at_k(query_tags, query_genres, top_k=5):
    """
    Evaluate one validation query:
      - Transforms the query tags using the TRAIN-fitted TF-IDF (transform only)
      - Gets top_k recommendations from the TRAIN vectors
      - Computes precision = (# hits with genre overlap) / top_k
    """
    # Transform using train-fitted vectorizer (no re-fitting)
    input_vector = tf.transform([query_tags])
    sim_scores   = cosine_similarity(input_vector, vectors).flatten()
    sorted_indices = np.argsort(-sim_scores)

    # Collect top_k unique recommendations
    recommended_genres = []
    seen_titles = set()
    for idx in sorted_indices:
        title = movies.iloc[idx]['title']
        if title not in seen_titles:
            movie_genre = movies.iloc[idx].get('genre', "")
            recommended_genres.append(get_genres(movie_genre))
            seen_titles.add(title)
        if len(recommended_genres) >= top_k:
            break

    # Count recommendations that share at least one genre with the query
    hits = sum(1 for rg in recommended_genres if rg & query_genres)
    return hits / top_k


# ── Run evaluation on a sample of the validation set ─────────────────────────
val_eval = val_movies.dropna(subset=['genre'])

# Sample 200 movies for speed (remove .sample() to evaluate the full val set)
sample = val_eval.sample(n=min(200, len(val_eval)), random_state=42)

scores = []
for _, row in sample.iterrows():
    query_genres = get_genres(row['genre'])
    if not query_genres:
        continue
    p = precision_at_k(row['tags'], query_genres, top_k=5)
    scores.append(p)

mean_precision = np.mean(scores)
print(f"\n{'='*50}")
print(f"Mean Precision@5 on Validation Set: {mean_precision:.4f}")
print(f"Evaluated on {len(scores)} movies")
print(f"{'='*50}")


# ── Precision@5 breakdown by genre ───────────────────────────────────────────
genre_scores = {}
for _, row in sample.iterrows():
    query_genres = get_genres(row['genre'])
    if not query_genres:
        continue
    p = precision_at_k(row['tags'], query_genres, top_k=5)
    for g in query_genres:
        genre_scores.setdefault(g, []).append(p)

print("\nPrecision@5 by Genre (min 5 samples):")
print(f"{'Genre':<22} {'Score':>6}  {'Count':>6}")
print("-" * 38)
for genre, sc in sorted(genre_scores.items(), key=lambda x: -np.mean(x[1])):
    if len(sc) >= 5:
        print(f"{genre:<22} {np.mean(sc):>6.3f}  (n={len(sc)})")


# =============================================================================
# SECTION 11: TESTING — MANUAL RECOMMENDATION QUERIES
# =============================================================================

print("\n" + "="*50)
print("RECOMMENDATION TESTS")
print("="*50)

# Test 1: Free-text genre keyword
print("\n[Test 1] Query: 'romance'")
print(recommend_movies('romance'))

# Test 2: Descriptive keywords
print("\n[Test 2] Query: 'action thriller spy'")
print(recommend_movies('action thriller spy'))

# Test 3: Exact movie title (will use its own vector)
print("\n[Test 3] Query: exact movie title from train set")
sample_title = movies['title'].iloc[0]
print(f"  Title: '{sample_title}'")
print(recommend_movies(sample_title))

# Test 4: A movie from the validation set (unseen during training)
print("\n[Test 4] Query: movie title from validation set")
val_sample_title = val_movies['title'].iloc[0]
print(f"  Title: '{val_sample_title}'")
print(recommend_movies(val_sample_title))

# Test 5: Description-style query
print("\n[Test 5] Query: 'a young wizard discovers magical powers'")
print(recommend_movies('a young wizard discovers magical powers'))
