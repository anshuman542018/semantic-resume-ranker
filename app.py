import streamlit as st
import pandas as pd
from ranker import extract_text_from_pdf, extract_text_from_docx, rank_resumes

st.set_page_config(
    page_title="Semantic Resume Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Semantic Resume Ranker")
st.markdown("**ATS 2.0** — Ranks resumes by *meaning*, not just keywords.")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="e.g. We're looking for a backend engineer with experience in REST APIs, Python, and cloud deployments..."
    )

with col2:
    st.subheader("📄 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

st.divider()

if st.button("🚀 Rank Resumes", use_container_width=True, type="primary"):

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    resumes = []
    with st.spinner("Extracting text from resumes..."):
        for file in uploaded_files:
            try:
                if file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(file)
                else:
                    text = extract_text_from_docx(file)
                resumes.append({"name": file.name, "text": text})
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")

    with st.spinner("Generating embeddings and ranking... (first run may take ~30s)"):
        ranked = rank_resumes(job_description, resumes)

    st.subheader("🏆 Ranked Results")

    def score_color(score):
        if score >= 0.75: return "🟢"
        if score >= 0.50: return "🟡"
        return "🔴"

    for i, result in enumerate(ranked):
        with st.expander(
            f"{score_color(result['score'])}  #{i+1}  {result['name']}  —  Score: {result['score']}",
            expanded=(i == 0)
        ):
            st.markdown(f"**Similarity Score:** `{result['score']}`")
            st.progress(result['score'])
            st.markdown("**Resume Preview:**")
            st.text(result['preview'] + "...")

    st.subheader("📊 Summary Table")
    df = pd.DataFrame([
        {"Rank": i+1, "Resume": r['name'], "Score": r['score']}
        for i, r in enumerate(ranked)
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)