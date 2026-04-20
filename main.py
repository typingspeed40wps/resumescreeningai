import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_text_from_pdf, preprocess
from skills import extract_skills

def load_job_description(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def rank_resumes(jd, folder):
    resumes = []
    names = []
    raw_texts = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            text = extract_text_from_pdf(path)
            if text.strip():
                resumes.append(preprocess(text))
                raw_texts.append(text)
                names.append(file)

    if not resumes:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocess(jd)] + resumes)

    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    results = []

    jd_skills = extract_skills(jd)

    for i, score in enumerate(scores):
        resume_skills = extract_skills(raw_texts[i])
        matched = list(set(jd_skills) & set(resume_skills))

        results.append({
            "Resume": names[i],
            "Score": round(score, 3),
            "Matched Skills": matched
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)

    return df

if __name__ == "__main__":
    jd = load_job_description("sample_job_description.txt")
    df = rank_resumes(jd, "resumes")

    if df.empty:
        print("No resumes found.")
    else:
        print(df)
        df.to_csv("results.csv", index=False)
