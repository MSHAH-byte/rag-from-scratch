from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import re

pdf_path = "EDU-AI_FINAL_THESIS.pdf"
reader = PdfReader(pdf_path)

# --------------------------------------------------
# 1. Extract pages
# --------------------------------------------------

pages = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if text:
        pages.append({
            "page": page_number,
            "text": text
        })

print("Pages extracted:", len(pages))


# --------------------------------------------------
# 2. Clean extracted PDF text
# --------------------------------------------------

def clean_text(text):

    # Remove page number at the beginning
    text = re.sub(r"^\s*\d+\s*\n", "", text)

    # Fix words split across lines with a hyphen.
    # Example:
    # fine-
    # tuned
    # becomes:
    # fine-tuned
    text = re.sub(r"-\s*\n\s*", "-", text)

    # Replace normal line breaks with spaces
    text = re.sub(r"\n+", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# 3. Chunk text with section metadata
# --------------------------------------------------

chunk_size = 1000
overlap = 200

chunks = []
metadatas = []


def detect_section(text, current_section):
    """
    Detect numbered thesis headings such as:

    4.6 AI Technologies Applied and Model Selection:
    4.6.1 Model Selection Strategy:
    """

    match = re.search(
        r"\b(\d+(?:\.\d+)+)\s+([^:]{3,100}):",
        text
    )

    if match:
        return match.group(0).strip()

    return current_section


for page in pages:

    page_number = page["page"]
    text = clean_text(page["text"])

    current_section = "Unknown"
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        # Avoid cutting through a word.
        if end < len(text):

            last_space = chunk.rfind(" ")

            if last_space > 500:
                chunk = chunk[:last_space]
                end = start + last_space

        # Detect section from the chunk.
        current_section = detect_section(
            chunk,
            current_section
        )

        chunks.append(chunk)

        metadatas.append({
            "page": page_number,
            "section": current_section
        })

        start = end - overlap


# --------------------------------------------------
# 4. Inspect chunks
# --------------------------------------------------

print("Number of chunks:", len(chunks))

print("\nSample chunks:")

for i in range(min(3, len(chunks))):

    print(f"\n{'=' * 70}")

    print(
        f"CHUNK {i + 1} | "
        f"Page {metadatas[i]['page']} | "
        f"Section: {metadatas[i]['section']}"
    )

    print("=" * 70)

    print(chunks[i])


# --------------------------------------------------
# 5. Create embeddings
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")

print("Creating embeddings...")

embeddings = model.encode(chunks)

print("Embeddings created.")

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 6. Store in ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Delete previous collection so old chunks don't remain.
try:

    client.delete_collection(name="eduai_thesis")

    print("\nOld collection deleted.")

except Exception:

    pass


collection = client.create_collection(
    name="eduai_thesis"
)


collection.upsert(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


print("\nStored in ChromaDB.")

print("Documents in database:", collection.count())