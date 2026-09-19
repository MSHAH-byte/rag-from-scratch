\# RAG From Scratch



A Retrieval-Augmented Generation (RAG) system built from scratch in Python using an academic thesis as the knowledge source.



The project demonstrates the core RAG pipeline without using LangChain or other RAG frameworks.



\## Architecture



```text

PDF Document

&#x20;    ↓

Text Extraction

&#x20;    ↓

Text Cleaning

&#x20;    ↓

Chunking + Overlap

&#x20;    ↓

Sentence Embeddings

&#x20;    ↓

ChromaDB Vector Database

&#x20;    ↓

Semantic Search

&#x20;    ↓

Top 10 Candidates

&#x20;    ↓

Cross-Encoder Reranking

&#x20;    ↓

Top 5 Chunks

&#x20;    ↓

Context Construction

&#x20;    ↓

Ollama Local LLM

&#x20;    ↓

Generated Answer

&#x20;    ↓

Source Pages

## Technologies

* Python
* PyPDF
* Sentence Transformers
* ChromaDB
* Cross-Encoder
* Ollama
* Gemma 3 (1B)
* PowerShell / VS Code

## How It Works

### 1. Document Ingestion

`index.py` extracts text from the PDF using PyPDF.

The extracted text is cleaned and divided into overlapping chunks.

### 2. Embeddings

Each chunk is converted into a vector representation using:

```text
all-MiniLM-L6-v2
```

These embeddings allow the system to perform semantic similarity search.

### 3. Vector Database

The embeddings and document chunks are stored in ChromaDB.

### 4. Retrieval

When a user asks a question, the question is converted into an embedding and compared against the stored vectors.

The system initially retrieves the top 10 candidate chunks.

### 5. Reranking

The 10 retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The 5 highest-scoring chunks are then used as the final context.

### 6. Generation

The retrieved context and user question are sent to a locally running Ollama model:

```text
eduai-finetuned:latest
```

The model generates an answer using the retrieved context.

### 7. Source Attribution

The system automatically displays the thesis pages associated with the retrieved context, allowing the user to see where the information came from.

## Project Structure

```text
rag-from-scratch/
│
├── index.py
├── query.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── EDUAI_FINAL_THESIS.pdf   # Not included in repository
└── chroma_db/               # Generated locally and ignored
```

## Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-from-scratch
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add a PDF

Place your PDF document in the project directory and update the PDF filename in `index.py` if necessary.

The thesis used during development is not included in the repository.

### 5. Build the Vector Database

```bash
python index.py
```

This extracts the document, creates chunks and embeddings, and stores them in ChromaDB.

### 6. Set Up Ollama

Make sure Ollama is installed and the required local model is available.

The project was developed using:

```text
eduai-finetuned:latest
```

If using a different Ollama model, update the model name in `query.py` and `evaluate.py`.

### 7. Ask Questions

```bash
python query.py
```

The system will retrieve relevant document sections, rerank them, generate an answer, and display the associated source pages.

### 8. Run Evaluation

```bash
python evaluate.py
```

## Evaluation

A basic evaluation was performed using five test questions covering:

* Questions answerable from the document
* Technology identification
* Model identification
* Database identification
* An out-of-scope question

Final evaluation results:

```text
Retrieval: 4/5 (80%)
Generation: 4/5 (80%)
```

The evaluation is intentionally simple and is meant to demonstrate basic retrieval and generation validation rather than provide a research-grade benchmark.

## What I Learned

This project was built to understand RAG at the implementation level before using higher-level frameworks such as LangChain.

Key concepts implemented:

* Document ingestion
* PDF text extraction
* Text preprocessing
* Chunking and overlap
* Embeddings
* Semantic similarity search
* Vector databases
* Metadata
* Reranking
* Context construction
* Local LLM generation
* Source attribution
* Retrieval evaluation
* Generation evaluation

## Future Work

A separate project will implement RAG using LangChain to explore higher-level abstractions and compare the framework-based approach with this from-scratch implementation.
