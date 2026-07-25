import pickle
import os

from preprocessing import clean_text


# -------------------------------------------------
# Load Saved Model
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)


# -------------------------------------------------
# Prediction Function
# -------------------------------------------------

def predict_sentiment(review):
    """
    Predicts sentiment for a movie review.
    """

    # Clean text
    cleaned_review = clean_text(review)

    # Convert to TF-IDF
    review_vector = vectorizer.transform([cleaned_review])

    # Predict class
    prediction = model.predict(review_vector)[0]

    # Predict probability
    probability = model.predict_proba(review_vector).max()

    return prediction, probability