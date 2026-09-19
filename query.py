from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import ollama


# --------------------------------------------------
# 1. Load models
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")

print("Loading reranker...")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Reranker loaded.")


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="eduai_thesis"
)


# --------------------------------------------------
# 3. Get question
# --------------------------------------------------

query = input("\nAsk a question about your thesis: ")

query_embedding = embedding_model.encode([query])


# --------------------------------------------------
# 4. Initial vector search
# --------------------------------------------------

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=10,
    include=["documents", "metadatas", "distances"]
)

candidate_chunks = results["documents"][0]
candidate_metadatas = results["metadatas"][0]
candidate_distances = results["distances"][0]


print("\nInitial vector search:")

for i, (document, metadata, distance) in enumerate(
    zip(
        candidate_chunks,
        candidate_metadatas,
        candidate_distances
    ),
    start=1
):
    print(
        f"Result {i} | "
        f"Page {metadata['page']} | "
        f"Distance: {distance:.4f}"
    )


# --------------------------------------------------
# 5. Rerank the candidates
# --------------------------------------------------

pairs = [
    [query, document]
    for document in candidate_chunks
]

reranker_scores = reranker.predict(pairs)


reranked = sorted(
    zip(
        candidate_chunks,
        candidate_metadatas,
        reranker_scores
    ),
    key=lambda x: x[2],
    reverse=True
)


# Keep the best 5 chunks after reranking.
top_chunks = reranked[:5]


print("\nReranked results:")

for i, (document, metadata, score) in enumerate(
    top_chunks,
    start=1
):
    print(
        f"\n--- Reranked Result {i} | "
        f"Page {metadata['page']} ---"
    )

    print(f"Reranker score: {score:.4f}")

    print(document)


# --------------------------------------------------
# 6. Build context
# --------------------------------------------------

context_parts = []

for document, metadata, score in top_chunks:

    context_parts.append(
        f"[Source: Page {metadata['page']}]\n"
        f"{document}"
    )

context = "\n\n".join(context_parts)


# --------------------------------------------------
# 7. Generate answer
# --------------------------------------------------

prompt = f"""
You are answering a question about the EduAI final-year project.

Use ONLY the information provided in the retrieved context.

Your job is to synthesize the answer from ALL relevant retrieved
chunks. Do not copy an unrelated sentence just because it appears
in the context.

If the question asks about technologies, identify the actual
technologies, frameworks, models, databases, and tools mentioned
in the context.

If the answer cannot be found in the retrieved context, say:

"I don't have enough information in the retrieved context."

Be concise and directly answer the question.

Do NOT generate a Sources section.
Do NOT invent page numbers.

Retrieved context:

{context}

Question:

{query}

Answer:
"""


print("\nGenerating answer...")

response = ollama.chat(
    model="eduai-finetuned:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]


# --------------------------------------------------
# 8. Generate sources automatically with Python
# --------------------------------------------------

source_pages = sorted(
    set(
        metadata["page"]
        for document, metadata, score in top_chunks
    )
)


# --------------------------------------------------
# 9. Display final answer
# --------------------------------------------------

print("\n--- RAG ANSWER ---")
print(answer)

print("\n--- SOURCES ---")

for page in source_pages:
    print(f"- Thesis Page {page}")