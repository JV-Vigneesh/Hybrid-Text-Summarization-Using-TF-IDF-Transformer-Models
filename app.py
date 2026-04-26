from flask import Flask, render_template, request, send_file
import os
from utils import *
import PyPDF2
import docx

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def limit_text_safely(text, max_words=2000):
    words = text.split()
    return " ".join(words[:max_words])

# -----------------------------
# Extract file text
# -----------------------------
def extract_text(file):
    filename = file.filename.lower()
    text = ""  # ✅ ALWAYS initialize

    try:
        if filename.endswith(".txt"):
            text = file.read().decode("utf-8")

        elif filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + " "

        elif filename.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + " "

    except Exception as e:
        print("File read error:", e)
        return ""


    return text.strip()


# -----------------------------
# Main route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    text = ""

    if request.method == "POST":
        try:
            text = request.form.get("text")

            file = request.files.get("file")
            if file and file.filename != "":
                text = extract_text(file)

            # limit size
            if text:
                text = limit_text_safely(text)

            if not text:
                result = {"hybrid": "No readable text found."}
                return render_template("index.html", result=result)

            if not is_valid_text(text):
                result = {
                    "tfidf": "Input not suitable (invoice/table detected).",
                    "bart": "Please upload paragraph-based text.",
                    "hybrid": "Model works best on descriptive content.",
                    "points": [],
                    "original_len": len(text.split()),
                    "summary_len": 0,
                    "reduction": 0,
                    "rouge": {
                        "TF-IDF": {"rouge1": 0, "rouge2": 0, "rougeL": 0},
                        "BART": {"rouge1": 0, "rouge2": 0, "rougeL": 0},
                        "HYBRID": {"rouge1": 0, "rouge2": 0, "rougeL": 0}
                    }
                }
            else:
                tfidf = tfidf_summary(text)
                bart = bart_summary(text)
                hybrid = hybrid_summary(text)
                points = key_points(text)

                original_len = len(text.split())
                summary_len = len(hybrid.split())
                if original_len > 0:
                    reduction = round((1 - summary_len / original_len) * 100, 2)
                else:
                    reduction = 0

                scores = {
                    "TF-IDF": compute_rouge(text, tfidf),
                    "BART": compute_rouge(text, bart),
                    "HYBRID": compute_rouge(text, hybrid)
                }

                try:
                    berts = {
                        "TF-IDF": compute_bertscore(text, tfidf),
                        "BART": compute_bertscore(text, bart),
                        "HYBRID": compute_bertscore(text, hybrid)
                    }
                except:
                    berts = {"TF-IDF": 0, "BART": 0, "HYBRID": 0}

                result = {
                    "tfidf": tfidf,
                    "bart": bart,
                    "hybrid": hybrid,
                    "points": points,
                    "original_len": original_len,
                    "summary_len": summary_len,
                    "reduction": reduction,
                    "rouge": scores,
                    "bertscore": berts
                }

        except Exception as e:
            print("ERROR:", e)

            result = {
                "hybrid": "⚠️ Error processing file. Try smaller or cleaner document."
            }

    return render_template("index.html", result=result)


# -----------------------------
# Download
# -----------------------------
@app.route("/download", methods=["POST"])
def download():
    summary = request.form.get("summary")

    path = f"summary_{os.getpid()}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)