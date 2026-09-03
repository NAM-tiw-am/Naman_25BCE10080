from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(docs,size,overlap):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=size,      
    chunk_overlap=overlap,
    separators=["/n/n","/n"," ",""]    
    )

    chunks=splitter.split_documents(docs)

    return chunks
