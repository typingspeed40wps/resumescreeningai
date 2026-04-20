import PyPDF2
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
    except:
        pass
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
    return " ".join(tokens)
