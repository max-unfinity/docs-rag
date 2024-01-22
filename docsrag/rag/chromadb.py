from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma


def build_db(docs, persist_directory):
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-base",
        model_kwargs={"device": 0},  # Comment out to use CPU
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.add_documents(docs)
    return vectorstore


def read_db(persist_directory):
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-base",
        model_kwargs={"device": 0},  # Comment out to use CPU
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def get_retriever(vectorstore, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever