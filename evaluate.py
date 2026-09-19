from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import ollama


# --------------------------------------------------
# 1. Load models
# --------------------------------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Models loaded.")


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
# 3. Evaluation questions
# --------------------------------------------------

test_questions = [
    {
        "question": "What is the main objective of EduAI?",
        "expected": "automate the creation of academic content"
    },
    {
        "question": "What technologies were used to develop EduAI?",
        "expected": "Flutter"
    },
    {
        "question": "What language model was selected for EduAI?",
        "expected": "Gemma 3"
    },
    {
        "question": "What database technology was used for semantic search?",
        "expected": "vector database"
    },
    {
        "question": "Who was the first person to walk on Mars?",
        "expected": "I don't have enough information"
    }
]


# --------------------------------------------------
# 4. Run evaluation
# --------------------------------------------------

retrieval_passed = 0
generation_passed = 0

print("\n")
print("=" * 70)
print("RAG EVALUATION")
print("=" * 70)


for number, test in enumerate(test_questions, start=1):

    question = test["question"]
    expected = test["expected"]

    print(f"\nTest {number}: {question}")

    # ----------------------------------------------
    # Vector search
    # ----------------------------------------------

    query_embedding = embedding_model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=10,
        include=["documents", "metadatas"]
    )

    candidate_chunks = results["documents"][0]
    candidate_metadatas = results["metadatas"][0]

    # ----------------------------------------------
    # Reranking
    # ----------------------------------------------

    pairs = [
        [question, document]
        for document in candidate_chunks
    ]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(
            candidate_chunks,
            candidate_metadatas,
            scores
        ),
        key=lambda x: x[2],
        reverse=True
    )

    top_chunks = reranked[:5]

    # ----------------------------------------------
    # Check retrieval
    # ----------------------------------------------

    retrieved_text = " ".join(
        document
        for document, metadata, score in top_chunks
    ).lower()

    expected_lower = expected.lower()

    if expected_lower in retrieved_text:
        print("Retrieval: PASS")
        retrieval_passed += 1
    else:
        print("Retrieval: REVIEW")

    # ----------------------------------------------
    # Build context
    # ----------------------------------------------

    context_parts = []

    for document, metadata, score in top_chunks:

        context_parts.append(
            f"[Source: Page {metadata['page']}]\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)

    # ----------------------------------------------
    # Generate answer
    # ----------------------------------------------

    prompt = f"""
You are answering a question about the EduAI final-year project.

Use ONLY the information provided in the retrieved context.

If the answer cannot be found in the retrieved context, say:

"I don't have enough information in the retrieved context."

Be concise and directly answer the question.

Retrieved context:

{context}

Question:

{question}

Answer:
"""

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

    print(f"Answer: {answer}")

    # ----------------------------------------------
    # Check generation
    # ----------------------------------------------

    answer_lower = answer.lower()

    if expected_lower in answer_lower:
        print("Generation: PASS")
        generation_passed += 1
    else:
        print("Generation: REVIEW")


# --------------------------------------------------
# 5. Final results
# --------------------------------------------------

total = len(test_questions)

print("\n")
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Retrieval: "
    f"{retrieval_passed}/{total} "
    f"({retrieval_passed / total * 100:.1f}%)"
)

print(
    f"Generation: "
    f"{generation_passed}/{total} "
    f"({generation_passed / total * 100:.1f}%)"
)