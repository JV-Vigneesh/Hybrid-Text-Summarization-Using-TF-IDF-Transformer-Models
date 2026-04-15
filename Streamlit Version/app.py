import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# -----------------------------
# Setup
# -----------------------------
nltk.download('punkt')

st.set_page_config(page_title="Hybrid Text Summarizer", layout="wide")

st.title("🧠 Hybrid Text Summarization")
st.write("TF-IDF + Transformer (BART)")

# -----------------------------
# Load Model (once)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

summarizer = load_model()

# -----------------------------
# Functions
# -----------------------------
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s) > 20]

def rank_sentences_tfidf(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    ranked = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )
    return ranked

def abstractive_summary(text):
    input_len = len(text.split())

    max_len = max(30, min(int(input_len * 0.6), 120))
    min_len = max(10, min(int(input_len * 0.3), 60))

    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return result[0]['summary_text']

def hybrid_summarize(text, k=3):
    sentences = preprocess_text(text)

    if len(sentences) == 0:
        return "Text too short."

    ranked = rank_sentences_tfidf(sentences)
    extracted = " ".join([s for _, s in ranked[:k]])

    return abstractive_summary(extracted)

# -----------------------------
# Input Options
# -----------------------------
option = st.radio(
    "Choose Input Method:",
    ["Paste Text", "Upload File"]
)

text = ""

if option == "Paste Text":
    text = st.text_area("Enter your text:", height=300)

elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

# -----------------------------
# Summarization
# -----------------------------
if st.button("Generate Summary"):
    if text.strip() == "":
        st.warning("Please provide input text.")
    else:
        with st.spinner("Processing..."):
            tfidf_sum = " ".join([s for _, s in rank_sentences_tfidf(preprocess_text(text))[:3]])
            bart_sum = abstractive_summary(text)
            hybrid_sum = hybrid_summarize(text)

        st.subheader("📊 Model Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### TF-IDF")
            st.info(tfidf_sum)

        with col2:
            st.markdown("### BART")
            st.info(bart_sum)

        with col3:
            st.markdown("### HYBRID ⭐")
            st.success(hybrid_sum)