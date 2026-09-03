# RAG Pipeline Architecture Guide

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture that combines information retrieval with large language model (LLM) text generation. Instead of relying solely on an LLM's training data, RAG retrieves relevant documents from an external knowledge base and passes them as context to the LLM. This produces more accurate, up-to-date, and verifiable answers.

RAG was introduced by Meta AI (formerly Facebook AI Research) in 2020 in the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."

## Why Use RAG?

LLMs have three key limitations that RAG solves:

1. Knowledge Cutoff: LLMs only know what was in their training data. RAG provides access to current, domain-specific information.
2. Hallucination: LLMs sometimes generate plausible but incorrect answers. RAG grounds responses in actual documents, reducing hallucinations.
3. Source Attribution: With RAG, you can trace every answer back to its source document, enabling verification and trust.

## The 5 Stages of a RAG Pipeline

### Stage 1: Document Loading
Raw documents (PDFs, text files, markdown, web pages, databases) are loaded into the system. Tools like LangChain provide loaders for 100+ formats. Each document is represented as a Document object with page_content (the text) and metadata (source file, page number, etc.).

### Stage 2: Text Splitting (Chunking)
Documents are split into smaller chunks, typically 500-1000 characters with 50-200 character overlap. Overlap ensures that context is not lost at chunk boundaries. The RecursiveCharacterTextSplitter splits on natural boundaries like paragraphs, sentences, and words. Chunk size affects retrieval quality: too small loses context, too large dilutes relevance.

### Stage 3: Embedding Generation
Each chunk is converted into a numerical vector (embedding) that captures its semantic meaning. Similar texts produce similar vectors. Popular embedding models include sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, fast, good quality), OpenAI text-embedding-3-small (1536 dimensions), and Cohere embed-english-v3.0.

### Stage 4: Vector Store
Embeddings are stored in a vector database optimized for similarity search. When a query comes in, its embedding is compared against all stored embeddings to find the most similar chunks. Popular vector stores include Chroma (lightweight, local), FAISS (Facebook AI, very fast), Pinecone (cloud-hosted, scalable), and Weaviate (open-source, feature-rich). Distance metrics used: cosine similarity (most common), Euclidean distance, and dot product.

### Stage 5: Generation
The retrieved chunks and the user's query are combined into a prompt template and sent to an LLM. The LLM generates an answer grounded in the retrieved context. Temperature should be kept low (0.0-0.3) for factual RAG answers.

## Advanced RAG Techniques

### Hybrid Search
Combines dense vector search (semantic similarity) with sparse keyword search (BM25/TF-IDF). This handles cases where exact keyword matches are important alongside semantic understanding.

### Re-ranking
After initial retrieval, a cross-encoder model re-scores the results for better relevance. Models like cross-encoder/ms-marco-MiniLM-L-6-v2 are commonly used. This adds latency but significantly improves retrieval quality.

### Multi-Query Retrieval
The original query is rephrased into multiple variations by an LLM. Each variation retrieves documents independently. Results are merged and deduplicated. This captures different aspects of the user's intent.

### Contextual Compression
Retrieved chunks are compressed or filtered to include only the most relevant portions. This reduces noise in the context window and improves answer quality.

## Evaluation Metrics for RAG

- Context Relevance: Are the retrieved chunks relevant to the query?
- Faithfulness: Is the generated answer faithful to the retrieved context (no hallucination)?
- Answer Relevance: Does the answer actually address the user's question?
- Tools: RAGAS framework, LangSmith, and TruLens provide automated RAG evaluation.

## Common Pitfalls

1. Chunk size too large: Retrieval returns broadly relevant but not precise passages.
2. Chunk size too small: Context is fragmented and incomplete.
3. No overlap between chunks: Important information at boundaries is lost.
4. Wrong embedding model: Using a general model for domain-specific content.
5. Too few retrieved chunks (k too low): Missing important context.
6. Too many retrieved chunks (k too high): Diluting relevant information with noise.
7. Not cleaning source data: Garbage in, garbage out applies to RAG.
