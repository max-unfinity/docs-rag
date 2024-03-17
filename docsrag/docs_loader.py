from git import Repo
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def read_docs(docs_path, glob="**/*.md", loader_cls=TextLoader, chunk_size=4000, chunk_overlap=500):
    loader = DirectoryLoader(docs_path, glob=glob, loader_cls=loader_cls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits


def clone_repo(url, repo_dir, progress=None):
    repo_name = url.split('/')[-1]
    local_repo_path = f'{repo_dir}/{repo_name}'
    repo = Repo.clone_from(url, local_repo_path, progress=progress)
    print(f"Repository cloned to '{local_repo_path}'")
    return local_repo_path
