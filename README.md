# 🎵 Song Recommendation System

An AI-powered music recommendation web application that discovers similar songs using TF-IDF content-based filtering. Simply select a song and get instant recommendations with similarity scores!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Live Demo

🔗 **[Try the App Here](https://spotify-recommendation-ntxriehfkbsyrw8qtjwmfa.streamlit.app/)**



## ✨ Features

- **🎯 Instant Recommendations**: Select a song and get similar tracks immediately
- **📊 Similarity Scores**: See percentage similarity between songs
- **🎛️ Customizable**: Adjust number of recommendations (5-15)
- **⚡ Fast Performance**: Optimized for quick response times
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🎨 Clean UI**: Intuitive interface with sidebar controls

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn (TF-IDF Vectorization)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Community Cloud

## 🧠 How It Works

1. **Data Preprocessing**: Cleans and processes song titles from the dataset
2. **TF-IDF Vectorization**: Converts song titles into numerical vectors
3. **Similarity Calculation**: Uses cosine similarity to find similar songs
4. **Real-time Recommendations**: Instantly displays results when user selects a song

### Algorithm Details

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to analyze song titles:

- **TF**: How frequently a term appears in a song title
- **IDF**: How rare or common a term is across all song titles
- **Cosine Similarity**: Measures similarity between song vectors

## 📊 Dataset

The application uses the Spotify-YouTube dataset containing:
- Song titles
- Artist information
- Various music metadata

*Note: Make sure to have `Spotify_Youtube.csv` in your project directory*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/song-recommendation-system.git
   cd song-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your dataset**
   - Place `Spotify_Youtube.csv` in the project root directory

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`

## 📁 Project Structure

```
song-recommendation-system/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Spotify_Youtube.csv    # Dataset (not included in repo)
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## 🔧 Configuration

### Customizable Parameters

In the sidebar, you can adjust:
- **Number of recommendations**: 5-15 songs
- **Show similarity scores**: Toggle on/off
- **Dataset information**: View stats about loaded data

### Performance Settings

For large datasets, the app automatically:
- Limits TF-IDF features to 3000 for memory efficiency
- Samples datasets larger than 10,000 songs
- Uses caching for faster subsequent loads

## 📈 Performance Metrics

- **Response Time**: < 2 seconds for recommendations
- **Memory Usage**: Optimized for cloud deployment
- **Accuracy**: Content-based filtering with similarity scores
- **Scalability**: Handles datasets up to 10K songs efficiently

## 🚀 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click!

### Local Development

```bash
# Install in development mode
pip install -e .

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Ideas for Contributions

- [ ] Add collaborative filtering
- [ ] Integrate Spotify Web API
- [ ] Add genre-based filtering
- [ ] Implement user ratings
- [ ] Add song preview functionality
- [ ] Create playlist generation feature

## 🐛 Known Issues

- Large datasets (>10K songs) are automatically sampled for performance
- Non-English song titles may appear in recommendations
- Similarity is based only on title text, not audio features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@swathy06]
- LinkedIn: [https://www.linkedin.com/in/swathy-ramakrishnan/]
- Email: sujswa@gmail.com

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
- [Spotify-YouTube Dataset](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube) for the music data
- Open source community for inspiration and tools

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/song-recommendation-system?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/song-recommendation-system?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/song-recommendation-system)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/song-recommendation-system)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

## 🔮 Future Enhancements

- **Audio Feature Analysis**: Integrate Spotify's audio features (tempo, energy, valence)
- **Deep Learning**: Implement neural collaborative filtering
- **User Profiles**: Add personalized recommendations based on listening history
- **Social Features**: Share playlists and recommendations with friends
- **Mobile App**: Create React Native version
- **Real-time Updates**: Live sync with user's Spotify account

---

*Built with ❤️ using Python and Streamlit*


