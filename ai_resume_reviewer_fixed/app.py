import streamlit as st
import PyPDF2
import os
import base64
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up OpenAI client (use secret key or environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_feedback_from_openai(resume_text):
    prompt = f"You're an HR expert. Please review this resume and suggest improvements:\n\n{resume_text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

def get_ats_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = similarity[0][0] * 100
    return round(score, 2)

def create_download_button(content, filename):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Feedback Report</a>'
    return href

st.set_page_config(page_title="AI Resume Reviewer & ATS Ranker", layout="centered")
st.title("📄 AI Resume Reviewer & ATS Match Checker")
st.write("Upload your resume and job description to get AI-based feedback and ATS match score.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the Job Description here")

if resume_file and job_description:
    with st.spinner("🔍 Analyzing..."):
        resume_text = extract_text_from_pdf(resume_file)
        feedback = get_feedback_from_openai(resume_text)
        ats_score = get_ats_score(resume_text, job_description)

    st.success("✅ Analysis Complete!")

    st.subheader("🧠 Resume Feedback (from AI):")
    st.write(feedback)

    st.subheader("📊 ATS Match Score:")
    st.metric(label="Match Percentage", value=f"{ats_score}%")

    if ats_score < 60:
        st.warning("❌ Low score! Resume might not pass ATS screening.")
    else:
        st.success("✅ Good score! Resume likely to pass ATS.")

    report = f"AI Feedback:\n{feedback}\n\nATS Score: {ats_score}%\n\nJob Description:\n{job_description}"
    st.markdown(create_download_button(report, "resume_feedback.txt"), unsafe_allow_html=True)
