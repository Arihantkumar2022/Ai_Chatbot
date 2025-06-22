from PyPDF2 import PdfReader
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def save_text_to_file(text: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    pdf_path = r"C:\Users\ARIHANT KUMAR\Desktop\TAsk\project\data\AI Training Document.pdf"
    output_path = r"C:\Users\ARIHANT KUMAR\Desktop\TAsk\project\data\raw_document.txt"

    if os.path.exists(pdf_path):
        print("Extracting text from:", pdf_path)
        text = extract_text_from_pdf(pdf_path)
        save_text_to_file(text, output_path)
        print("Saved to:", output_path)
    else:
        print("PDF not found:", pdf_path)
