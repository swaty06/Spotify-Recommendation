import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

# Page configuration
st.set_page_config(
    page_title="🎵 Song Recommender",
    page_icon="🎵",
    layout="wide"
)

@st.cache_data
def load_and_process_data():
    """Load and preprocess the dataset with error handling."""
    try:
        # Try different possible file locations
        possible_files = ["Spotify_Youtube.csv", "data/Spotify_Youtube.csv", "./Spotify_Youtube.csv"]
        df_train = None
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                df_train = pd.read_csv(file_path)
                break
        
        if df_train is None:
            st.error("❌ Dataset file 'Spotify_Youtube.csv' not found!")
            st.info("Please make sure the CSV file is uploaded to your repository.")
            return None
        
        # Validate required column exists
        if 'Title' not in df_train.columns:
            st.error("❌ 'Title' column not found in dataset!")
            st.info("Available columns: " + ", ".join(df_train.columns.tolist()))
            return None
        
        # Memory optimization: sample data if too large
        if len(df_train) > 10000:
            st.warning(f"⚠️ Dataset has {len(df_train)} rows. Sampling 10,000 for performance.")
            df_train = df_train.sample(n=10000, random_state=42).reset_index(drop=True)
        
        # Fill missing text fields
        df_train['Title'] = df_train['Title'].fillna('')
        
        # Remove empty titles and duplicates
        df_train = df_train[df_train['Title'].str.strip() != '']
        df_train = df_train.drop_duplicates(subset=['Title']).reset_index(drop=True)
        
        return df_train
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

@st.cache_data
def create_similarity_matrix(df_train):
    """Create TF-IDF matrix and compute similarity."""
    try:
        # Reduced parameters for cloud deployment
        tfv = TfidfVectorizer(
            min_df=2,  # Reduced from 3
            max_features=3000,  # Reduced from 5000 for memory efficiency
            strip_accents='unicode',
            analyzer='word', 
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),  # Reduced from (1,3) for memory
            stop_words='english'
        )
        
        tfidf_matrix = tfv.fit_transform(df_train['Title'])
        
        # Compute similarity matrix
        sig = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Create mapping from title to index
        indices = pd.Series(df_train.index, index=df_train['Title']).drop_duplicates()
        
        return sig, indices
        
    except MemoryError:
        st.error("❌ Out of memory! Dataset might be too large for deployment.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error creating similarity matrix: {str(e)}")
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
        
        recommendations = []
        for i in song_indices:
            recommendations.append({
                'title': df_train['Title'].iloc[i],
                'similarity': sig[idx][i]
            })
        
        return recommendations
        
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
        return []

# Streamlit App
def main():
    # Header
    st.title("🎵 Song Recommendation System")
    st.markdown("*Discover similar songs*")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app uses TF-IDF (Term Frequency-Inverse Document Frequency) to find songs with similar titles.")
        st.write("Simply select a song to see instant recommendations!")
        
        st.header("🔧 Settings")
        num_recs = st.slider("Number of recommendations:", min_value=5, max_value=15, value=10)
        show_scores = st.checkbox("Show similarity scores", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load data
        with st.spinner("🔄 Loading dataset..."):
            df_train = load_and_process_data()
            
        if df_train is None:
            st.stop()
        
        # Create similarity matrix
        with st.spinner("🤖 Building AI model..."):
            sig, indices = create_similarity_matrix(df_train)
            
        if sig is None:
            st.stop()
        
        st.success(f"✅ Ready! Dataset contains {len(df_train)} unique songs.")
        
        # Song selection
        st.subheader("🎯 Select a Song")
        
        # Show all songs in selectbox
        all_songs = sorted(df_train['Title'].unique())
        song_choice = st.selectbox("Choose a song to get recommendations:", all_songs, key="song_selector")
        
        # Auto-generate recommendations when song is selected
        if song_choice:
            with st.spinner("🎯 Finding similar songs..."):
                recommendations = give_rec(song_choice, df_train, sig, indices, num_recs)
                
                if recommendations:
                    st.subheader(f"🎧 Songs Similar to: **{song_choice}**")
                    
                    for i, rec in enumerate(recommendations, 1):
                        if show_scores:
                            similarity_percentage = round(rec['similarity'] * 100, 1)
                            st.write(f"**{i}.** {rec['title']} `({similarity_percentage}% similar)`")
                        else:
                            st.write(f"**{i}.** {rec['title']}")
                        
                else:
                    st.warning("❌ No recommendations found for this song.")
    
    with col2:
        st.subheader("📊 Dataset Info")
        if df_train is not None:
            st.metric("Total Songs", len(df_train))
            st.metric("Unique Titles", df_train['Title'].nunique())
            
            # Show sample data
            with st.expander("👁️ Sample Data"):
                st.dataframe(df_train[['Title']].head(), use_container_width=True)

if __name__ == "__main__":
    main()


