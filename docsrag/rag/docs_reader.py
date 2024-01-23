from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def read_docs(docs_path, glob="**/*.md", loader_cls=TextLoader, chunk_size=4000, chunk_overlap=500):
    loader = DirectoryLoader(docs_path, glob=glob, loader_cls=loader_cls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits
