from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma


def build_db(docs, db_name, db_dir, hf_model_name="intfloat/multilingual-e5-base"):
    embeddings = HuggingFaceEmbeddings(
        model_name=hf_model_name,
        model_kwargs={"device": 0},  # Comment out to use CPU
    )
    vectorstore = Chroma(
        collection_name=db_name,
        embedding_function=embeddings,
        persist_directory=db_dir,
    )
    vectorstore.add_documents(docs)
    return vectorstore


def read_db(db_name, db_dir, hf_model_name="intfloat/multilingual-e5-base"):
    embeddings = HuggingFaceEmbeddings(
        collection_name=db_name,
        model_name=hf_model_name,
        model_kwargs={"device": 0},  # Comment out to use CPU
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=db_dir,
    )
    return vectorstore


def get_retriever(vectorstore, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever