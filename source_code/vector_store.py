from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Constants
PERSIST_DIR = "../chroma_db"
COLLECTION_NAME = "rag_collection"


def get_embedding_model():
    """Return the embedding model (reuse same one everywhere)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def get_vectorstore():
    """
    Load an EXISTING vector store from disk.
    Returns None if the store doesn't exist yet.
    """
    if os.path.exists(PERSIST_DIR):
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=get_embedding_model(),
        )
    return None


def create_vectorstore(chunks):
    """
    Create a NEW vector store from chunks (List[Document]).
    If the store already exists on disk, it appends to it.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


def add_documents(chunks):
    """
    Add new documents to an EXISTING vector store.
    If no store exists, creates one.
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return create_vectorstore(chunks)

    vectorstore.add_documents(chunks)
    return vectorstore