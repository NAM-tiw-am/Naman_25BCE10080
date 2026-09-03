# RAG Pipeline — Naman_25BCE10080

A **Retrieval-Augmented Generation (RAG)** pipeline built from scratch for the **SDC Technical Task Round 2026–27 (AI/ML Track)**.  
The system loads documents, chunks them, embeds them into a vector store, retrieves the most relevant chunks for a user query, and generates a grounded answer via an LLM.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Documents  │────▶│  Chunking    │────▶│   Embedding    │
│  (.pdf .md  │     │  (Recursive  │     │ (HuggingFace   │
│   .txt)     │     │   Splitter)  │     │  MiniLM-L6-v2) │
└─────────────┘     └──────────────┘     └───────┬────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  LLM Answer │◀────│  Retrieved   │◀────│  ChromaDB      │
│  (Groq API) │     │  Top-k Chunks│     │  Vector Store  │
└─────────────┘     └──────────────┘     └────────────────┘
```

**Data Flow:**  
`Documents → Loader → Chunker → Embedder → ChromaDB → Retriever → LLM → Answer`

---

## Design Justifications

### Why `chunk_size = 400` and `chunk_overlap = 50`?

- **400 tokens** is small enough to keep each chunk semantically focused (one idea per chunk) while being large enough to preserve meaningful context.
- **50-token overlap** ensures that sentences sitting on a chunk boundary aren't lost — the retriever can still find them in at least one chunk.
- These values are configurable in [`pipeline.py`](Sourcecode/pipeline.py) (line 65) if you want to experiment.

### Why `sentence-transformers/all-MiniLM-L6-v2`?

- It is one of the most widely used open-source embedding models — lightweight (~80 MB), fast on CPU, and produces 384-dimensional vectors.
- It delivers strong semantic similarity scores on general-purpose English text, which matches our document types (knowledge-base articles, reference guides).
- Being a local HuggingFace model, it runs entirely offline — no API key or network calls needed for embedding.

### Why ChromaDB?

- ChromaDB is a lightweight, file-based vector database that persists to disk out of the box — no server setup, no Docker, no external dependencies.
- Perfect for a project of this scale: it stores embeddings in `chroma_db/` and reloads them on subsequent runs without re-embedding.
- Supports the standard LangChain `VectorStore` interface, making retrieval and integration straightforward.

### Why Groq (LLM Provider)?

- Groq provides a **free-tier API** with extremely fast inference, making it accessible for demo and evaluation without cost barriers.
- The pipeline uses the `openai/gpt-oss-20b` model via Groq's API for generation.
- The LLM provider is easily swappable — any LangChain-compatible chat model (OpenAI, Ollama, etc.) can replace it by editing [`generation.py`](Sourcecode/generation.py).

---

## Project Structure

```
Naman_25BCE10080/
├── Sourcecode/
│   ├── pipeline.py           # Main entry point — orchestrates the full RAG flow
│   ├── documents_loaders.py  # Loads .pdf, .md, and .txt files from the data/ folder
│   ├── text_splliter.py      # Chunks documents using RecursiveCharacterTextSplitter
│   ├── vector_store.py       # Creates/loads ChromaDB vector store with HuggingFace embeddings
│   ├── retriever.py          # Retrieves top-k relevant chunks via similarity search
│   └── generation.py         # Formats prompt, calls the LLM, returns the answer
├── data/
│   ├── project_nimbus_knowledge_base.pdf
│   ├── project_nimbus_knowledge_base.md
│   ├── project_nimbus_knowledge_base.txt
│   ├── machine_learning_fundamentals.md
│   ├── python_quick_reference.txt
│   └── rag_architecture_guide.md
├── .env                      # API key (not committed — see .env.example)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/NAM-tiw-am/Naman_25BCE10080.git
   cd Naman_25BCE10080
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**  
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the pipeline**
   ```bash
   cd Sourcecode
   python pipeline.py
   ```

   On first run, the pipeline will:
   - Load all documents from `data/`
   - Chunk them (400 chars, 50 overlap)
   - Embed and store them in ChromaDB (`chroma_db/`)

   On subsequent runs, it loads the existing vector store from disk.

6. **Ask questions**
   ```
   > You: What is Project Nimbus?
   > You: Explain the RAG architecture
   > You: quit
   ```

---

## How Each Module Works

| Module | Responsibility |
|---|---|
| **`documents_loaders.py`** | Uses LangChain's `DirectoryLoader` to load `.txt`, `.md`, and `.pdf` files from the `data/` folder. Each format uses the appropriate loader class (`TextLoader` for text/md, `PyPDFLoader` for PDF). |
| **`text_splliter.py`** | Splits loaded documents into smaller chunks using `RecursiveCharacterTextSplitter` with configurable `chunk_size` and `chunk_overlap`. Uses hierarchical separators (`\n\n`, `\n`, ` `, `""`) to split at natural boundaries. |
| **`vector_store.py`** | Embeds chunks using `all-MiniLM-L6-v2` and stores them in a ChromaDB collection. Supports creating, loading, and appending to the vector store. |
| **`retriever.py`** | Wraps the vector store's similarity search into a LangChain retriever. Returns the top-k most relevant chunks for a given query. |
| **`generation.py`** | Constructs a grounded prompt (context + question), sends it to the Groq LLM, and returns the generated answer. The prompt explicitly instructs the model to answer only from the provided context. |
| **`pipeline.py`** | Orchestrates the entire flow: build/load the vector store → accept user queries in a loop → retrieve → generate → display the answer with source attribution. |

---

## Supported Document Formats

The pipeline ingests three document formats, demonstrating that loaders are not PDF-specific:

| Format | Loader | Example in `data/` |
|---|---|---|
| `.pdf` | `PyPDFLoader` | `project_nimbus_knowledge_base.pdf` |
| `.md` | `TextLoader` | `rag_architecture_guide.md`, `machine_learning_fundamentals.md` |
| `.txt` | `TextLoader` | `python_quick_reference.txt`, `project_nimbus_knowledge_base.txt` |

---

## Technologies Used

- **Python 3.9+**
- **LangChain** — orchestration framework for document loading, splitting, retrieval, and LLM integration
- **ChromaDB** — lightweight persistent vector database
- **HuggingFace Transformers** (`sentence-transformers/all-MiniLM-L6-v2`) — local embedding model
- **Groq API** — fast LLM inference (free tier)
- **PyPDF** — PDF parsing

---

## Author

**Naman Tiwari**  
Registration No: 25BCE10080  
SDC Technical Task Round 2026–27 — AI/ML Track
