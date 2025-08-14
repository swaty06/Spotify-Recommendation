# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 21:34:02 2025

@author: rampr
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
df_train = pd.read_csv("Spotify_Youtube.csv")

# Fill missing text fields
df_train['Title'] = df_train['Title'].fillna('')

# Create TF-IDF matrix
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                      analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), stop_words='english')
tfidf_matrix = tfv.fit_transform(df_train['Title'])

# Compute similarity matrix
sig = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create mapping from title to index
indices = pd.Series(df_train.index, index=df_train['Title']).drop_duplicates()

def give_rec(title, sig=sig):
    if title not in indices:
        return []
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]  # top 10 excluding itself
    song_indices = [i[0] for i in sig_scores]
    return df_train['Title'].iloc[song_indices].tolist()

import streamlit as st

st.title("🎵 Spotify Song Recommendation (TF-IDF)")
st.write("Select a song to find similar tracks.")

# Song selection dropdown
song_choice = st.selectbox("Choose a song:", df_train['Title'].unique())

# Recommend button
if st.button("Recommend"):
    recommendations = give_rec(song_choice)
    if recommendations:
        st.subheader("Recommended Songs:")
        for i, song in enumerate(recommendations, 1):
            st.write(f"{i}. 🎧 {song}")
    else:
        st.warning("No recommendations found for this song.")

