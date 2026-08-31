from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFLoader

def docs_loader(folder):
    Text_data=DirectoryLoader(
        path=folder,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"},
        ).load()

    md_data=DirectoryLoader(
        path=folder,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"},
        ).load()

    pdf_data=DirectoryLoader(
        path=folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        ).load()

    docs=md_data+pdf_data+Text_data
    return docs
