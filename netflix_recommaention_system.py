import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------
def load_data():
    file_name = "netflix_titles.xlsx"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f" Could not find {file_name}. Please put it in this folder.")

    # Explicitly use openpyxl engine
    df = pd.read_excel(file_name, engine="openpyxl")
    print(" Data loaded successfully!")
    print("Rows:", len(df), " Columns:", len(df.columns))
    return df


# ------------------------------------------------------------
# STEP 2: CLEAN DATA
# ------------------------------------------------------------
def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.fillna("", inplace=True)

    # Convert all text columns to strings to avoid type errors
    df["title"] = df["title"].astype(str)
    df["listed_in"] = df["listed_in"].astype(str)
    df["description"] = df["description"].astype(str)

    # Combine features safely
    df["combined_features"] = (
        df["title"].fillna("") + " " + df["listed_in"].fillna("") + " " + df["description"].fillna("")
    )

    print(" Data cleaned successfully! Combined text features ready.")
    return df


# ------------------------------------------------------------
# STEP 3: SIMPLE DATA ANALYSIS
# ------------------------------------------------------------
def analyze_data(df):
    print("\n Data Summary:")
    print(df["type"].value_counts())

    # Plot Movie vs TV Show count
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="type", palette="viridis")
    plt.title("Movies vs TV Shows on Netflix")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.show(block=False)

    # Top 10 genres
    plt.figure(figsize=(9, 5))
    top_genres = df["listed_in"].value_counts().head(10)
    sns.barplot(y=top_genres.index, x=top_genres.values, palette="coolwarm")
    plt.title("Top 10 Genres on Netflix")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.show(block=False)


# ------------------------------------------------------------
# STEP 4: BUILD TF-IDF MATRIX
# ------------------------------------------------------------
def build_tfidf_matrix(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    print(" TF-IDF matrix built successfully. Shape:", tfidf_matrix.shape)
    return tfidf_matrix


# ------------------------------------------------------------
# STEP 5: RECOMMEND FUNCTION
# ------------------------------------------------------------
def recommend(title, df, tfidf_matrix, top_n=5):
    title = title.strip().lower()
    matches = df[df["title"].str.lower() == title]

    if matches.empty:
        print(f" '{title}' not found in dataset. Try another title.")
        return

    idx = matches.index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1][1 : top_n + 1]

    print(f"\n 🎬 Top {top_n} recommendations similar to '{df.loc[idx, 'title']}':\n")
    for i in similar_indices:
        print(f" 👉🏻 {df.loc[i, 'title']} ({df.loc[i, 'type']}, {df.loc[i, 'release_year']})")
        print(f"   Genre: {df.loc[i, 'listed_in']}")
        print(f"   Description: {df.loc[i, 'description']}\n")


# ------------------------------------------------------------
# STEP 6: MAIN FUNCTION
# ------------------------------------------------------------
def main():
    df = load_data()
    df = clean_data(df)
    analyze_data(df)
    tfidf_matrix = build_tfidf_matrix(df)

    while True:
        user_input = input("\nEnter a movie/show title (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print(" 👋🏻 Goodbye!")
            break
        recommend(user_input, df, tfidf_matrix, top_n=5)


# ------------------------------------------------------------
# RUN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    main()


