from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from chromadb import PersistentClient


def build_db(docs, collection_name, db_dir, hf_model_name="intfloat/multilingual-e5-base"):
    embeddings = HuggingFaceEmbeddings(
        model_name=hf_model_name,
        model_kwargs={"device": 0},  # Comment out to use CPU
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_dir,
    )
    vectorstore.add_documents(docs)
    return vectorstore


def read_db(collection_name, db_dir, hf_model_name="intfloat/multilingual-e5-base"):
    embeddings = HuggingFaceEmbeddings(
        model_name=hf_model_name,
        model_kwargs={"device": 0},  # Comment out to use CPU
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_dir,
    )
    return vectorstore


def get_retriever(vectorstore, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


def list_collections(db_dir):
    client = PersistentClient(db_dir)
    return client.list_collections()