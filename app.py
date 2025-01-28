from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
import spacy
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import nltk
import os
import re

# Initialize Flask app
app = Flask(__name__)

# Download required resources
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# PDF text extractor
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"
def gen_sim_matrix(sentences, stop_words):
    """
    Generate a similarity matrix for sentences.
    Ensures all sentences have a minimal connection with epsilon to avoid disconnected nodes.
    """
    epsilon = 1e-6  # Small value to ensure connectivity
    similarity_matrix = np.zeros((len(sentences), len(sentences))) + epsilon

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(
                    sentences[idx1], sentences[idx2], stop_words
                )

    # Normalize the similarity matrix (row-wise normalization)
    row_sums = similarity_matrix.sum(axis=1)
    similarity_matrix = similarity_matrix / row_sums[:, np.newaxis]

    return similarity_matrix

# Text summarizer
def summarize_text(text, maxN=5):
    stopWords = stopwords.words('english')

    # Split the text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Clean sentences
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
        if cleaned_sentence.strip():
            cleaned_sentences.append(cleaned_sentence.split())

    if len(cleaned_sentences) == 0:
        return "No sentences found in the text."

    # Ensure maxN doesn't exceed the number of sentences
    maxN = min(maxN, len(cleaned_sentences))

    # Generate similarity matrix
    similarity_matrix = gen_sim_matrix(cleaned_sentences, stopWords)

    # Create a similarity graph
    similarity_graph = nx.from_numpy_array(similarity_matrix)

    # Compute PageRank scores
    try:
        scores = nx.pagerank(
            similarity_graph,
            max_iter=300,  # Increased iterations
            tol=1e-4,      # Adjusted tolerance
            alpha=0.85
        )
    except nx.PowerIterationFailedConvergence:
        # Fallback if PageRank fails
        scores = {i: sum(similarity_matrix[i]) for i in range(len(cleaned_sentences))}

    # Rank sentences and handle missing scores
    ranked_sentences = sorted(
        ((scores.get(i, 0), s) for i, s in enumerate(sentences)), reverse=True
    )

    # Generate the summary
    summary = " ".join([ranked_sentences[i][1] for i in range(maxN)])
    return summary

# Sentence similarity function
def sentence_similarity(sen1, sen2, stopwords=None):
    if not sen1 or not sen2:
        return 0.0

    sen1 = [w.lower() for w in sen1 if w.lower() not in stopwords]
    sen2 = [w.lower() for w in sen2 if w.lower() not in stopwords]

    if not sen1 or not sen2:
        return 0.0

    allWords = list(set(sen1 + sen2))
    vector1 = [0] * len(allWords)
    vector2 = [0] * len(allWords)

    for w in sen1:
        vector1[allWords.index(w)] += 1
    for w in sen2:
        vector2[allWords.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'pdf_file' not in request.files:
        return "No file uploaded."

    pdf_file = request.files['pdf_file']

    # Save uploaded PDF locally
    pdf_path = os.path.join("uploads", pdf_file.filename)
    os.makedirs("uploads", exist_ok=True)
    pdf_file.save(pdf_path)

    # Extract text and summarize
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text or extracted_text.startswith("Error"):
        return extracted_text

    summary = summarize_text(extracted_text)

    # Save summary to a file
    summary_path = os.path.join("uploads", "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return render_template('result.html', summary=summary, file_path=summary_path)

@app.route('/download')
def download():
    file_path = request.args.get('file_path', default='uploads/summary.txt')
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
