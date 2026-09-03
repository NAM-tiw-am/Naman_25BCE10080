import os
import sys

# Fix Windows terminal encoding for special characters in LLM responses
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
from documents_loaders import docs_loader
from text_splliter import chunking
from vector_store import get_vectorstore, create_vectorstore
from retriever import retrieve
from generation import generate

# Load .env from the project root (one level up from Sourcecode/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, "..", ".env")
load_dotenv(_env_path)


def display_welcome():
    print("\n" + "=" * 60)
    print("  [RAG Pipeline] - Ask Questions From Your Documents")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.\n")


def display_response(query, answer, sources, num_chunks):
    print("\n" + "-" * 60)
    print(f"  [Query]:  {query}")
    print("-" * 60)
    print(f"\n  [Answer]:\n")
    for line in answer.split("\n"):
        print(f"     {line}")

    print(f"\n  [Sources used]: {num_chunks} chunk(s) from:")
    for src in sources:
        print(f"     - {src}")

    print("-" * 60 + "\n")


def generate_with_sources(query, k=5):
    retrieved_docs = retrieve(query, k=k)

    answer = generate(query, k=k)
    sources = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in sources:
            sources.append(source)

    return answer, sources, len(retrieved_docs)


def build_vectorstore():
    print("\n[*] Checking vector store...")
    vectorstore = get_vectorstore()

    if vectorstore is None:
        print("[*] No existing store found. Building from documents...")
        docs = docs_loader("data")
        print(f"    [OK] Loaded {len(docs)} document(s)")

        chunks = chunking(docs, 400, 50)
        print(f"    [OK] Split into {len(chunks)} chunk(s)")

        vectorstore = create_vectorstore(chunks)
        print("    [OK] Vector store created and saved to disk!\n")
    else:
        print("[OK] Vector store loaded from disk!\n")

    return vectorstore


def main():
    build_vectorstore()
    display_welcome()
    while True:
        try:
            query = input("  > You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        print("\n  [Thinking...]")
        try:
            answer, sources, num_chunks = generate_with_sources(query, k=5)
            display_response(query, answer, sources, num_chunks)
        except Exception as e:
            print(f"\n  [ERROR]: {e}\n")


if __name__ == "__main__":
    main()