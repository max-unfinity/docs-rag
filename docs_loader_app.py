import settings
import streamlit as st
from docsrag.chromadb import build_db
from docsrag import docs_loader


st.title("Docs Loader")
repo_url = st.text_input("Repository URL:")
base_url = st.text_input("Docs website URL:")
hf_model_name = st.text_input("Hugging Face model name:", settings.DEFAULT_EMBEDDING_MODEL)


if st.button("Load"):
    with st.spinner("Cloning repository..."):
        path = docs_loader.clone_repo(repo_url, settings.APP_DATA)
    docs = docs_loader.read_docs(path)
    st.info(f"Loaded {len(docs)} document chunks.")
    collection_name = ".".join(repo_url.split('/')[-2:])
    with st.spinner(f'Building vectorstore "{collection_name}", it may take a while...'):
        vectorstore = build_db(docs, collection_name, settings.CHROMA_DB_DIR, hf_model_name)
    metadata = {
        "repo_url": repo_url,
        "base_url": base_url,
        "embedding_model": hf_model_name,
    }
    docs_loader.update_collection_metadata(f"{settings.APP_DATA}/collections.json", collection_name, metadata)
    st.success("Done!")