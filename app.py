import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit App UI
st.set_page_config(page_title="Flipkart Review Sentiment", layout="centered")
st.title("🛍️ Flipkart Product Review Sentiment Analyzer")
st.write("Enter a product review below and find out if the sentiment is **Positive** or **Negative**! 🔍")

# Text input
review = st.text_area("💬 Enter Review Here:", height=150)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a valid review!")
    else:
        cleaned_review = clean_text(review)
        vector = tfidf.transform([cleaned_review])
        prediction = model.predict(vector.toarray())[0]

        if prediction == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")
