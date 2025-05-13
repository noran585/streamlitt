import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV data
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")
    return df

# Search function
def search_books(query, documents, data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents + [query])
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    data['score'] = cosine_sim
    results = data.sort_values(by='score', ascending=False)
    return results[results['score'] > 0]  # Filter relevant results

# Streamlit UI
def main():
    st.set_page_config(page_title="📚 Smart Book Search", layout="centered")

    # Custom styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #F5F5F5;
        }
        .result {
            background-color: #ffffff;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .title {
            font-size: 20px;
            color: #0066cc;
            font-weight: bold;
        }
        .author {
            color: #444;
            font-style: italic;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 Smart Library Search")
    st.write("Search for any book using keywords ✨")

    df = load_data()
    query = st.text_input("🔍 Enter keywords or book title:")

    if query.strip() != "":
        docs = df['content'].astype(str).tolist()
        results = search_books(query, docs, df.copy())

        if not results.empty:
            st.subheader("📖 Results:")
            for _, row in results.head(5).iterrows():
                st.markdown(f"""
                    <div class="result">
                        <div class="title">{row['title']}</div>
                        <div class="author">by {row['author']}</div>
                        <p>{row['content']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No results found. Try different keywords.")

if __name__ == "__main__":
    main()
