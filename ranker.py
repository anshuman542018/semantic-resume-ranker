
import fitz  
import docx
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK stopwords on first run
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


# all-MiniLM-L6-v2 is fast, lightweight, and great for semantic similarity
MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(file) -> str:
    """Extract raw text from a PDF file object."""
    text = ""
    # fitz.open can take a stream (file bytes) directly
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_docx(file) -> str:
    """Extract raw text from a .docx file object."""
    document = docx.Document(file)
    return "\n".join([para.text for para in document.paragraphs])


def clean_text(text: str) -> str:
    """
    Preprocessing pipeline:
    1. Lowercase everything
    2. Remove special characters and extra whitespace
    3. Remove stopwords (words like 'the', 'is', 'at' add noise)
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # keep only alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace
    
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return ' '.join(tokens)


def get_embedding(text: str) -> np.ndarray:
    """
    Convert text to a high-dimensional vector using a transformer model.
    The model understands meaning — 'built APIs' and 'developed REST services'
    will have similar vectors even though the words differ.
    """
    return MODEL.encode(text)


def rank_resumes(job_description: str, resumes: list[dict]) -> list[dict]:
    """
    Core ranking function.
    
    Args:
        job_description: Raw JD text from the recruiter
        resumes: List of dicts [{"name": filename, "text": raw_text}, ...]
    
    Returns:
        Sorted list of resumes with similarity scores (highest first)
    """
    # Step 1: Clean and embed the Job Description
    clean_jd = clean_text(job_description)
    jd_vector = get_embedding(clean_jd)

    results = []
    for resume in resumes:
        # Step 2: Clean and embed each resume
        clean_resume = clean_text(resume['text'])
        resume_vector = get_embedding(clean_resume)

        # Step 3: Cosine Similarity — measures the angle between two vectors
        # Score of 1.0 = identical meaning, 0.0 = completely unrelated
        score = cosine_similarity(
            jd_vector.reshape(1, -1),
            resume_vector.reshape(1, -1)
        )[0][0]

        results.append({
            "name": resume['name'],
            "score": round(float(score), 4),
            "preview": resume['text'][:300]  # first 300 chars for preview
        })

    # Sort by score descending
    return sorted(results, key=lambda x: x['score'], reverse=True)
