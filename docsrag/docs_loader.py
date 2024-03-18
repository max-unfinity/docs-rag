import os
from git import Repo
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


def read_docs(docs_path, glob="**/*.md", loader_cls=TextLoader, chunk_size=4000, chunk_overlap=500):
    loader = DirectoryLoader(docs_path, glob=glob, loader_cls=loader_cls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits


def clone_repo(url, repo_dir, progress=None):
    repo_name = url.split('/')[-1]
    local_repo_path = f'{repo_dir}/{repo_name}'
    if os.path.exists(local_repo_path):
        return local_repo_path
    repo = Repo.clone_from(url, local_repo_path, progress=progress)
    return local_repo_path


def update_collection_metadata(json_file, collection_name, metadata: dict):
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[collection_name] = metadata
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)