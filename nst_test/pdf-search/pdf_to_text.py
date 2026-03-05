from pathlib import Path
from pypdf import PdfReader

#PDF_PATH = Path.cwd() / ("../docs/Redefining-the-Modern-Semantic-Layer-Databricks.pdf")
#TXT_PATH = Path.cwd() / ("../docs/Redefining-the-Modern-Semantic-Layer-Databricks.txt")
PDF_PATH = Path.cwd() / ("../docs/Speech and Language Processing - Daniel Jurafsky.pdf")
TXT_PATH = Path.cwd() / ("../docs/Speech and Language Processing - Daniel Jurafsky.txt")

pdf_reader = PdfReader(PDF_PATH)
txt_file = Path(TXT_PATH)

content = [
    #f"{pdf_reader.metadata.title}",
    f"Number of pages: {len(pdf_reader.pages)}"
]

print("Reading PDF content: {}".format(PDF_PATH))
print(pdf_reader.metadata)

for page in pdf_reader.pages:
    content.append(page.extract_text())

txt_file.write_text("\n".join(content))