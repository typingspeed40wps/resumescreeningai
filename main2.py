import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from utils import extract_text_from_pdf, preprocess
from skills import extract_skills

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_job_description(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def score_resume(jd_text, resume_text):
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    semantic_score = util.cos_sim(jd_embedding, resume_embedding).item()

    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    skill_match = len(set(jd_skills) & set(resume_skills))
    skill_score = skill_match / (len(jd_skills) + 1)

    final_score = (0.7 * semantic_score) + (0.3 * skill_score)

    return final_score, semantic_score, skill_score, jd_skills, resume_skills

def rank_resumes(jd, folder):
    results = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            text = extract_text_from_pdf(path)

            if not text.strip():
                continue

            processed = preprocess(text)

            final, semantic, skill, jd_sk, res_sk = score_resume(jd, processed)

            results.append({
                "Resume": file,
                "Final Score": round(final, 3),
                "Semantic Score": round(semantic, 3),
                "Skill Score": round(skill, 3),
                "Matched Skills": list(set(jd_sk) & set(res_sk))
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Final Score", ascending=False)

    return df

if __name__ == "__main__":
    jd = load_job_description("sample_job_description.txt")
    df = rank_resumes(jd, "resumes")

    if df.empty:
        print("No resumes found.")
    else:
        print(df)
        df.to_csv("results.csv", index=False)
