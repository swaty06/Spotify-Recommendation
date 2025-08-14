import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

@st.cache_data
def load_and_process_data():
    """Load and preprocess the dataset with error handling."""
    try:
        if not os.path.exists("Spotify_Youtube.csv"):
            st.error("Dataset file 'Spotify_Youtube.csv' not found!")
            return None
        
        df_train = pd.read_csv("Spotify_Youtube.csv")
        
        # Validate required column exists
        if 'Title' not in df_train.columns:
            st.error("'Title' column not found in dataset!")
            return None
        
        # Fill missing text fields
        df_train['Title'] = df_train['Title'].fillna('')
        
        # Remove empty titles
        df_train = df_train[df_train['Title'].str.strip() != '']
        
        return df_train
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_data
def create_similarity_matrix(df_train):
    """Create TF-IDF matrix and compute similarity."""
    try:
        # Create TF-IDF matrix
        tfv = TfidfVectorizer(
            min_df=3, 
            max_features=5000,  # Limit features for performance
            strip_accents='unicode',
            analyzer='word', 
            token_pattern=r'\w{1,}',
            ngram_range=(1, 3), 
            stop_words='english'
        )
        
        tfidf_matrix = tfv.fit_transform(df_train['Title'])
        
        # Compute similarity matrix
        sig = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Create mapping from title to index
        indices = pd.Series(df_train.index, index=df_train['Title']).drop_duplicates()
        
        return sig, indices
        
    except Exception as e:
        st.error(f"Error creating similarity matrix: {str(e)}")
        return None, None

def give_rec(title, df_train, sig, indices, num_recommendations=10):
    """Generate recommendations for a given song title."""
    if title not in indices:
        return []
    
    try:
        idx = indices[title]
        sig_scores = list(enumerate(sig[idx]))
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[1:num_recommendations+1]  # Exclude itself
        song_indices = [i[0] for i in sig_scores]
        
        # Return both titles and similarity scores
        recommendations = []
        for i in song_indices:
            recommendations.append({
                'title': df_train['Title'].iloc[i],
                'similarity': sig[idx][i]
            })
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# Streamlit App
def main():
    st.title("🎵 Spotify Song Recommendation System")
    st.write("Find similar songs using TF-IDF content-based filtering")
    
    # Load data
    df_train = load_and_process_data()
    if df_train is None:
        st.stop()
    
    # Create similarity matrix
    with st.spinner("Building recommendation model..."):
        sig, indices = create_similarity_matrix(df_train)
        if sig is None:
            st.stop()
    
    st.success(f"✅ Model ready! Dataset contains {len(df_train)} songs.")
    
    # Song selection
    st.subheader("🎯 Select a Song")
    
    # Search functionality
    search_term = st.text_input("🔍 Search for a song:", "")
    
    if search_term:
        # Filter songs based on search
        filtered_songs = df_train[df_train['Title'].str.contains(search_term, case=False, na=False)]['Title'].unique()
        if len(filtered_songs) > 0:
            song_choice = st.selectbox("Choose from search results:", filtered_songs)
        else:
            st.warning("No songs found matching your search.")
            song_choice = st.selectbox("Or choose from all songs:", df_train['Title'].unique())
    else:
        song_choice = st.selectbox("Choose a song:", df_train['Title'].unique())
    
    # Number of recommendations
    num_recs = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)
    
    # Recommend button
    if st.button("🎵 Get Recommendations", type="primary"):
        with st.spinner("Finding similar songs..."):
            recommendations = give_rec(song_choice, df_train, sig, indices, num_recs)
            
            if recommendations:
                st.subheader(f"🎧 Songs Similar to '{song_choice}':")
                
                for i, rec in enumerate(recommendations, 1):
                    similarity_percentage = round(rec['similarity'] * 100, 1)
                    st.write(f"{i}. **{rec['title']}** (Similarity: {similarity_percentage}%)")
                    
            else:
                st.warning("No recommendations found for this song.")
    
    # Dataset info
    with st.expander("ℹ️ Dataset Information"):
        st.write(f"**Total songs:** {len(df_train)}")
        st.write(f"**Unique titles:** {df_train['Title'].nunique()}")
        if 'Artist' in df_train.columns:
            st.write(f"**Unique artists:** {df_train['Artist'].nunique()}")

if __name__ == "__main__":
    main()
