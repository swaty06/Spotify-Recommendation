
# Song Recommendation System 🎵

A simple **content-based song recommendation system** built using a Spotify-inspired dataset from Kaggle.  
The system recommends songs based on similarity of song titles using **TF-IDF vectorization** and **sigmoid kernel** for similarity scoring.

---

## Dataset

The dataset used is from Kaggle and contains song details such as:
- Title
- Artist
- Album
- Other metadata



---

## Features

- TF-IDF vectorization on song titles to convert text into numerical features.
- Similarity calculation using `sigmoid_kernel` from `sklearn.metrics.pairwise`.
- Functionality to select a song and get a list of **similar songs ranked by similarity score**.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/song-recommendation.git ```

2. Navigate to the project folder:
```bash
cd song-recommendation ```

