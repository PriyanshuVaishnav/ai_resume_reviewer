# 🤖 AI Resume Reviewer & ATS Ranker

This project is an AI-powered tool that helps job seekers improve their resumes by:
- Reviewing resume content (via OpenAI)
- Matching resume with job descriptions
- Scoring ATS (Applicant Tracking System) match %
- Suggesting improvements with actionable feedback

## 🚀 How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Add your OpenAI API key inside `app.py`:

```python
openai.api_key = "your_openai_api_key"
```

3. Run the app:

```
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud

Push this folder to GitHub and deploy via [streamlit.io/cloud](https://streamlit.io/cloud)
