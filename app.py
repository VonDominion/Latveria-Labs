import streamlit as st
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.abspath("src"))

from src.predictor import predict_sentiment



# ---------------- Sidebar ----------------

st.sidebar.title("📌 Project Overview")

st.sidebar.markdown("""
### 🤖 Model
Logistic Regression

### 📊 Model Accuracy
**≈89%**

### 🔤 Feature Extraction
TF-IDF Vectorizer

### 🎬 Dataset
IMDb Movie Reviews

50,000 Reviews

Binary Classification

### 💻 Technologies

- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- Streamlit
---

Mini Internship Project
by Naitik Tiwari
""")


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.caption(
    "Predict whether a movie review is Positive or Negative using Machine Learning."
)

st.markdown("---")

st.write(
    """
This application uses Natural Language Processing (NLP) and Machine Learning to classify IMDb movie reviews as Positive or Negative.
"""
)

st.divider()


with st.expander("ℹ️ How does this work?"):

    st.write("""
1. Your review is cleaned using NLP preprocessing.
2. The cleaned text is converted into numerical features using TF-IDF.
3. A Logistic Regression model predicts the sentiment.
4. The app displays the predicted sentiment and the confidence score.
""")
    

# --------------------------------------------------
# User Input
# --------------------------------------------------

review = st.text_area(
    "📝 Enter Movie Review",
    height=220,
    placeholder="Example: This movie was absolutely fantastic. The acting and story were brilliant..."
)

st.caption(f"Characters: {len(review)}")
st.caption("Minimum: 5 words")
# --------------------------------------------------
# Prediction
# --------------------------------------------------

predict = st.button(
    "🔍 Predict Sentiment",
    use_container_width=True
)

if predict:

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    elif len(review.split()) < 5:
        st.warning("Please enter at least 5 words for a better prediction.")

    else:

            with st.spinner("Analyzing your review..."):

                prediction, confidence = predict_sentiment(review)

            st.divider()

            st.subheader("Prediction Result")

            if prediction.lower() == "positive":

               st.success("😊 Positive Review")

            else:

                st.error("😞 Negative Review")

            # Progress Bar
            st.progress(float(confidence))

            # Confidence Percentage
            st.metric(
                label="Confidence",
                value=f"{confidence*100:.2f}%"
            )

            if confidence >= 0.90:

                st.success("Confidence Level: Very High")

            elif confidence >= 0.75:

                st.info("Confidence Level: High")

                with st.expander("📄 View Submitted Review"):

                    st.write(review)

            elif confidence >= 0.60:

                st.warning("Confidence Level: Moderate")

            else:

                st.error("Confidence Level: Low")

            if prediction.lower() == "positive":

                st.info(
                    "The model predicts that this review expresses a **positive opinion**."
                )

                with st.expander("📄 View Submitted Review"):
                    st.write(review)

            else:

                st.info(
                    "The model predicts that this review expresses a **negative opinion**."
                )

                with st.expander("📄 View Submitted Review"):
                    st.write(review)

st.markdown("---")

st.subheader("💡Example Reviews")

st.markdown("""
### 😊 Positive

> This movie was absolutely amazing. The acting was brilliant and the story kept me engaged until the very end.

### 😞 Negative

> The movie was boring, predictable and a complete waste of time. I would never recommend it.

### 😐 Mixed

> I am a huge fan of this actor and I loved his previous movies. However, this film felt disappointing because the story became confusing and the ending was weak.
""")

st.markdown("---")

st.caption(
    """
Built by **Naitik Tiwari**

Python • Streamlit • Scikit-Learn • Logistic Regression
"""
)