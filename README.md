# 🎬 Sentiment Analysis on Movie Reviews

This project performs sentiment analysis on movie reviews using NLTK's built-in dataset and a Naive Bayes classifier.

## 📁 Dataset
Uses the `movie_reviews` corpus from NLTK — 2000 reviews labeled as positive or negative.

## 🛠️ Technologies Used
- Python
- NLTK
- Scikit-learn

## 🧪 How to Run

1. Clone this repo:
   bash:
   git clone https://github.com/Anishtaa/Sentiment-analysis-on-movie-reviews.git
   cd Sentiment-analysis-on-movie-reviews

2. Set up The environment:
    bash:
    pip install -r requirements.txt
    python nltk_data_setup.py

3. Run the script 
    bash:
    python sentiment_analysis.py

## 📊 Sample Output:

Accuracy on test set: 0.82

Sample Predictions:
Text: 'This movie was thrilling and full of suspense!' → pos
Text: 'The acting was terrible and the plot made no sense.' → neg

## Future improvements:
    Use TF-IDF + Logistic Regression

    Build a Streamlit-based web app



