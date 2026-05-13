
import streamlit as st
from resume_parser import extract_text
from job_matcher import match_jobs

st.title("🤖 AI Resume Analyzer & Job Matcher")

uploaded_file = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)
    st.subheader("📄 Extracted Resume Text:")
    st.write(resume_text)

    st.subheader("🎯 Top Job Matches:")
    matches = match_jobs(resume_text)
    st.table(matches)
