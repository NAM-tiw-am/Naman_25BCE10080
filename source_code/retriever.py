from vector_store import get_vectorstore


def create_retriever(k=5, search_type="similarity"):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("Vector store not found! Run the pipeline first to create it.")
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    return retriever


def retrieve(query, k=5):
    """
    Retrieve relevant chunks for a given query.
    
    Args:
        query: The user's question/query string
        k: Number of chunks to return
    
    Returns:
        List[Document] - the most relevant chunks
    """
    retriever = create_retriever(k=k)
    results = retriever.invoke(query)
    return results