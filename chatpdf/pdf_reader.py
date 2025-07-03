
import pdfplumber

def extract_text(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)
