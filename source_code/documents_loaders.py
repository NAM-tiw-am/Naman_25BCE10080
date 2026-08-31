from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFLoader

def docs_loader(folder):
    Text_data=DirectoryLoader(
        path=folder,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"},
        ).load()


    return docs

