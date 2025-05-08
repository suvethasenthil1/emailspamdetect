import streamlit as st
import pickle
import string
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure nltk data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Load model and vectorizer using correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Update the paths to the model and vectorizer files
    with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Required model/vectorizer files not found. Make sure 'vectorizer.pkl' and 'model.pkl' are present.")
    st.stop()  # Stop execution if the files are not found

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        st.subheader("🧠 Prediction:")
        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
