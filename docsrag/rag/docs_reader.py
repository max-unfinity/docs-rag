import os

from pathlib import Path
from typing import List, Union
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def read_docs(docs_path, glob="**/*.md", loader_cls=TextLoader, chunk_size=4000, chunk_overlap=500):
    loader = DirectoryLoader(docs_path, glob=glob, loader_cls=loader_cls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits


def docs_tree(url: str, repo_path: Union[str, Path], file_suffix: str = ".md"):
    if not os.path.exists(repo_path):
        Repo.clone_from(url, repo_path)
    return _make_tree(repo_path, file_suffix)


def _make_tree(cur_path: Union[str, Path], suffix: str) -> List[dict]:
    tree = []
    cur_path = Path(cur_path)
    for element in cur_path.iterdir():
        abs_path = str(element.absolute())
        if element.is_dir():
            tree.append(
                {
                    "label": element.name, 
                    "value": abs_path,
                    "children": _make_tree(element, suffix)
                }
            )
        elif element.suffix == suffix:
            tree.append(
                {
                    "label": element.stem,
                    "value": abs_path,
                }
            )
    
    return tree
