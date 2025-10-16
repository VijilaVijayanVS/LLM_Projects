import fitz  # PyMuPDF

def extract_text(upload_file) -> str:
    content = ""
    with fitz.open(stream=upload_file.file.read(), filetype="pdf") as doc:
        for page in doc:
            content += page.get_text()
    return content