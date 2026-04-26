import nltk
import numpy as np
import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bertscore

nltk.download('punkt')

# -----------------------------
# Models
# -----------------------------
device = 0 if torch.cuda.is_available() else -1

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# MODEL_NAME = "facebook/bart-large-cnn"

summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    device=device
)

embedder = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


# -----------------------------
# REMOVE HEADER (CRITICAL FIX)
# -----------------------------
def remove_header(text):
    if "Experience" in text:
        text = text.split("Experience", 1)[1]
    return text


# -----------------------------
# STRONG NOISE REMOVAL (FINAL)
# -----------------------------
def remove_noise(text):
    # remove metadata lines
    patterns = [
        r'(?i)id\s*no.*',
        r'(?i)name\s.*',
        r'(?i)course\s*name.*',
        r'(?i)instructor\s*name.*',
        r'(?i)department.*',
        r'(?i)branch.*',
        r'(?i)year.*term.*',
        r'(?i)assessment\s*type.*',
        r'(?i)journal\s*entry.*',
        r'(?i)module\s*number.*'
    ]

    for p in patterns:
        text = re.sub(p, '', text)

    # remove ID-like codes (241P1R100)
    text = re.sub(r'\b[A-Z0-9]{6,}\b', '', text)

    # remove numbers/bullets
    text = re.sub(r'\b\d+\.\s*', '', text)

    return text


# -----------------------------
# Normalize text
# -----------------------------
def normalize_text(text):
    text = re.sub(r'•', '. ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def normalize_output(text):
    import re

    # remove space before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # fix hyphen spacing
    text = re.sub(r'\s*-\s*', '-', text)

    # remove trailing broken sentence
    if not text.endswith(('.', '!', '?')):
        text = text.rsplit(' ', 1)[0]

    return text.strip()


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_text(text):
    text = remove_header(text)
    text = remove_noise(text)
    text = normalize_text(text)

    sentences = nltk.sent_tokenize(text)

    return [s.strip() for s in sentences if len(s.split()) > 6]


# -----------------------------
# TF-IDF
# -----------------------------
def rank_sentences_tfidf(sentences):
    if not sentences:
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(sentences)
        scores = np.array(tfidf.sum(axis=1)).flatten()

        return sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    except ValueError:
        return [(1, s) for s in sentences]


def tfidf_summary(text):
    sentences = preprocess_text(text)

    if not sentences:
        return "No meaningful textual content found."

    ranked = rank_sentences_tfidf(sentences)
    cleaned = []
    for _, s in ranked[:5]:
        s = re.sub(r'\(.*?\)', '', s)  # remove (class content)
        s = s.strip()
        cleaned.append(s)

    summary = " ".join(cleaned)
    summary = re.sub(r'\s+([.,!?])', r'\1', summary)
    summary = re.sub(r'[^A-Za-z0-9.,!? ]+', '', summary)

    summary = summary.replace("..", ".")

    return normalize_output(summary)



# -----------------------------
# Key Points
# -----------------------------
def key_points(text):
    sentences = preprocess_text(text)
    ranked = rank_sentences_tfidf(sentences)

    points = []
    for _, s in ranked[:5]:
        s = re.sub(r'\(.*?\)', '', s)   # remove brackets like (class content)
        s = s.strip()

        if len(s.split()) > 20:
            continue

        s = s.capitalize()

        if not s.endswith('.'):
            s += '.'

        s = normalize_output(s)

        points.append(s)

    return points


# -----------------------------
# Redundancy Removal
# -----------------------------
def remove_redundancy(text, threshold=0.75):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        return text

    sentences = sentences[:25]

    embeddings = embedder.encode(sentences, convert_to_tensor=True)

    selected = []
    used = set()

    for i in range(len(sentences)):
        if i in used:
            continue

        selected.append(sentences[i])

        for j in range(i + 1, len(sentences)):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > threshold:
                used.add(j)

    return " ".join(selected)


# -----------------------------
# Clean Output
# -----------------------------
def clean_and_format(text):
    sentences = nltk.sent_tokenize(text)

    cleaned = []
    for s in sentences:
        s = s.strip()

        if len(s.split()) < 4:
            continue

        s = s.capitalize()

        if not s.endswith(('.', '!', '?')):
            s += '.'

        cleaned.append(s)

    text = " ".join(cleaned)

    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'\s+-\s+', '-', text)  # fix part -of-speech

    return text

def fix_incomplete_sentence(text):
    if not text.endswith(('.', '!', '?')):
        text = text.rsplit(' ', 1)[0]
    return text

# -----------------------------
# BART
# -----------------------------
def bart_summary(text):
    text = remove_header(text)
    text = remove_noise(text)
    text = normalize_text(text)

    max_chunk = 500
    words = text.split()

    chunks = [
        " ".join(words[i:i + max_chunk])
        for i in range(0, len(words), max_chunk)
    ]

    summaries = []

    for chunk in chunks:
        input_len = len(chunk.split())

        max_len = max(80, min(int(input_len * 0.8), 220))
        min_len = max(50, int(max_len * 0.7))

        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )

        summaries.append(result[0]['summary_text'])

    combined = " ".join(summaries)

    final = remove_redundancy(combined)
    final = clean_and_format(final)
    final = fix_incomplete_sentence(final)
    final = normalize_output(final)

    final = final.replace(" ,", ",").replace(" .", ".")

    return final


# -----------------------------
# Hybrid (STRUCTURE-AWARE)
# -----------------------------
def hybrid_summary(text):
    text = remove_header(text)
    text = remove_noise(text)

    sentences = preprocess_text(text)

    if not sentences:
        return "Text too short."

    ranked = rank_sentences_tfidf(sentences)

    # take more sentences to avoid missing concepts
    extracted = " ".join([s for _, s in ranked[:10]])

    summary = bart_summary(extracted)
    return normalize_output(summary)


# -----------------------------
# ROUGE (FIXED)
# -----------------------------
def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


# -----------------------------
# Validation
# -----------------------------
def is_valid_text(text):
    words = text.split()

    if len(words) < 30:
        return False

    alpha_words = [w for w in words if w.isalpha()]

    if len(alpha_words) / len(words) < 0.5:
        return False

    return True


def compute_bertscore(reference, generated):
    P, R, F1 = bertscore([generated], [reference], lang="en", verbose=False)
    return round(F1.mean().item(), 4)