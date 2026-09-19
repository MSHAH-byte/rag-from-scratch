from pypdf import PdfReader

pdf_path = "EDU-AI_FINAL_THESIS.pdf"

reader = PdfReader(pdf_path)

page_number = 78
text = reader.pages[page_number - 1].extract_text()

print(f"\nTotal characters: {len(text)}")

print("\nRAW REPRESENTATION:")
print(repr(text[:3000]))