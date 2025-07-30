# product_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV
df = pd.read_csv("product_recommendation_sample.csv")

# Check if 'product_category' column exists
if 'product_category' not in df.columns:
    raise ValueError("CSV file must contain a 'product_category' column.")

# Check for nulls
df.dropna(subset=["product_name", "product_category"], inplace=True)

# TF-IDF Vectorizer on product categories
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['product_category'])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(product_name, top_n=5):
    if product_name not in df['product_name'].values:
        print(f"'{product_name}' not found in product list.")
        return

    idx = df[df['product_name'] == product_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Remove the first item (itself)
    sim_scores = [s for s in sim_scores if s[0] != idx]
    sim_scores = sim_scores[:top_n]

    if not sim_scores:
        print("No recommendations found.")
        return

    print(f"\n🔍 Recommendations for '{product_name}':")
    for i, score in sim_scores:
        print(f"→ {df.iloc[i]['product_name']} (Score: {score:.2f})")

# Example usage
get_recommendations("Smartphone", top_n=3)
