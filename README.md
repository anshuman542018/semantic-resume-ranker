# 🎯 Semantic Resume Ranker — ATS 2.0

<div align="center">

### ▶ [Live Demo](https://semantic-resume-ranker.streamlit.app/) &nbsp;|&nbsp; Built with Python · Sentence Transformers · Streamlit

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://YOUR-APP-URL.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/Model-all--MiniLM--L6--v2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<img width="1919" height="921" alt="image" src="https://github.com/user-attachments/assets/3a50b5f1-28c1-45a0-acd5-3d8f22911797" />
<img width="1848" height="796" alt="image" src="https://github.com/user-attachments/assets/4396482b-7622-4712-b12b-5ada343d9b8f" />
<img width="1858" height="349" alt="image" src="https://github.com/user-attachments/assets/ba3a9749-4173-41b9-9c00-cfa003845d1c" />



</div>

---

## 📌 The Problem with Traditional ATS

Traditional Applicant Tracking Systems (ATS) rely on **exact keyword matching**. A candidate who writes *"built REST APIs"* gets filtered out by a system looking for *"API development"* — even though the experience is identical.

**Semantic Resume Ranker solves this.** Instead of matching words, it matches *meaning*.

---

## 💡 The ML Solution

By converting both the Job Description and each Resume into **high-dimensional semantic vectors** using a pre-trained Transformer model, the system ranks candidates based on the **conceptual similarity** of their experience — not the exact words they used.

| Traditional ATS | Semantic Resume Ranker |
|---|---|
| Keyword matching | Meaning-based matching |
| Misses synonyms & paraphrasing | Understands context & intent |
| Binary pass/fail | Continuous similarity score (0–1) |
| No ranking logic | Ranked list with visual scores |

---

## 🏗️ Architecture & How It Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  Job Description│────▶│  Text Preprocessing  │────▶│  Sentence Embedding│
│  (Recruiter)    │     │  (NLTK + Regex)      │     │  all-MiniLM-L6-v2  │
└─────────────────┘     └──────────────────────┘     └────────┬───────────┘
                                                               │
┌─────────────────┐     ┌──────────────────────┐              ▼
│  PDF/DOCX       │────▶│  Text Extraction     │     ┌────────────────────┐
│  Resumes        │     │  (PyMuPDF / docx)    │────▶│  Cosine Similarity │
└─────────────────┘     └──────────────────────┘     │  Scoring & Ranking │
                                                      └────────┬───────────┘
                                                               │
                                                               ▼
                                                      ┌────────────────────┐
                                                      │  Ranked Results    │
                                                      │  Streamlit UI      │
                                                      └────────────────────┘
```

### Step-by-Step Pipeline

1. **Data Extraction** — `PyMuPDF` reads binary PDF files; `python-docx` handles `.docx`. Both return clean plain text.

2. **Preprocessing** — Text is lowercased, special characters are stripped, and stopwords (`the`, `is`, `at`) are removed using `NLTK`. This reduces noise so the model focuses on meaningful tokens.

3. **Embedding Generation** — Both the JD and each resume are passed through `all-MiniLM-L6-v2`, a lightweight Sentence Transformer that outputs a **384-dimensional vector** capturing semantic meaning.

4. **Similarity Scoring** — `scikit-learn`'s `cosine_similarity` measures the angular distance between the JD vector and each resume vector. A score of `1.0` = identical meaning; `0.0` = completely unrelated.

5. **Ranked Output** — Results are sorted by score and displayed with color-coded badges, progress bars, and a summary table.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | Interactive web interface |
| **Embeddings** | `sentence-transformers` + `all-MiniLM-L6-v2` | Semantic vector generation |
| **Similarity** | `scikit-learn` | Cosine similarity scoring |
| **PDF Parsing** | `PyMuPDF (fitz)` | PDF text extraction |
| **DOCX Parsing** | `python-docx` | Word document extraction |
| **NLP Preprocessing** | `NLTK` | Stopword removal |
| **Data Handling** | `pandas` | Results table rendering |
| **Hosting** | Streamlit Community Cloud | Free live deployment |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/anshuman542018/semantic-resume-ranker.git
cd semantic-resume-ranker

# 2. Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 Usage Guide

1. **Paste a Job Description** in the left panel — any role, any industry
2. **Upload Resumes** (PDF or DOCX) — bulk upload supported
3. Click **🚀 Rank Resumes**
4. View:
   - 🟢 High match (≥ 0.75) — Strong candidate
   - 🟡 Moderate match (≥ 0.50) — Review manually
   - 🔴 Low match (< 0.50) — Likely misaligned

---

## 📁 Project Structure

```
semantic-resume-ranker/
│
├── app.py               # Streamlit UI — layout, file upload, results display
├── ranker.py            # Core ML pipeline — extraction, cleaning, embedding, scoring
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes venv, cache, and system files
└── README.md            # This file
```

---

## 🧠 Key ML Concepts

**Why Cosine Similarity?**
When comparing text embeddings, we care about the *direction* of the vector, not its magnitude. Two resumes of different lengths can be equally relevant. Cosine similarity normalizes for length and purely measures directional alignment — making it ideal for semantic text comparison.

**Why `all-MiniLM-L6-v2`?**
It's a distilled model trained specifically for semantic similarity tasks. At just 80MB, it outperforms much larger models on sentence-pair tasks and runs fast enough for a real-time Streamlit demo — the right balance of accuracy and speed for a portfolio project.

---

## 🔮 Future Improvements

- [ ] Add support for bulk CSV export of ranked results
- [ ] Integrate skills-gap analysis (highlight what the resume is missing)
- [ ] Add a recruiter dashboard with session history
- [ ] Fine-tune the model on domain-specific (tech/finance/healthcare) resume datasets
- [ ] Add multi-JD comparison mode

---

## 👤 Author

**Anshuman Pandey**
- GitHub: [@anshuman542018](https://github.com/anshuman542018)
- Project: [Semantic Resume Ranker](https://github.com/anshuman542018/semantic-resume-ranker)

---

## 📄 License

This project is licensed under the MIT License — use it, learn from it, build on it.

---

<div align="center">

**If this project helped you, consider giving it a ⭐ on GitHub.**

</div>
