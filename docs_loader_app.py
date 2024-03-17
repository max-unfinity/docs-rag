import settings
import streamlit as st
from docsrag.chromadb import build_db
from docsrag import docs_loader

from git import Repo, RemoteProgress

class CloneProgress(RemoteProgress):
    def __init__(self, progress_bar):
        super().__init__()
        self.progress_bar = progress_bar

    def update(self, op_code, cur_count, max_count=None, message=''):
        if max_count:
            self.progress_bar.progress(cur_count / max_count)


def prepare_docs(url, progress_fn=None):
    path = docs_loader.clone_repo(url, settings.APP_DATA, progress_fn)
    docs = docs_loader.read_docs(path)
    print(f"Loaded {len(docs)} documents")

    repo_name = url.split('/')[-1]
    vectorstore = build_db(docs, repo_name, settings.CHROMA_DB_DIR, settings.EMBEDDING_MODEL_NAME)
    print(f"Built vectorstore for '{repo_name}'")

st.title("Docs Loader")
progress_bar = st.progress(0, "Cloning repository...")
url = st.text_input("Repository URL")

if st.button("Load"):
    prepare_docs(url)
    st.success("Done")