SKILLS_DB = [
    "python", "java", "c++", "machine learning", "deep learning",
    "nlp", "data analysis", "sql", "pandas", "numpy",
    "tensorflow", "pytorch", "excel", "communication"
]

def extract_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)
    return list(set(found))
