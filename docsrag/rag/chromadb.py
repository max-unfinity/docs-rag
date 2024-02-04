from typing import Optional
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_db(docs, persist_directory, device: Optional[int] = None) -> Chroma:
    model_kwargs={}

    if device is not None:
        model_kwargs["device"] = device

    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-base",
        model_kwargs=model_kwargs,
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.add_documents(docs)
    return vectorstore


def read_db(persist_directory, device: Optional[int] = None) -> Chroma:
    model_kwargs={}

    if device is not None:
        model_kwargs["device"] = device

    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-base",
        model_kwargs=model_kwargs,
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def get_retriever(vectorstore, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever
