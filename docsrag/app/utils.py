import os
from pathlib import Path
import sys
import hashlib
from importlib import import_module
from typing import List, Optional
from langchain.vectorstores.chroma import Chroma

from docsrag.rag.docs_reader import read_docs
from docsrag.rag.chromadb import read_db, build_db


def upd_sqlite_version() -> None:
    import_module("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def prepare_vectorstore(name: str, docs_path: Optional[str] = None, device: Optional[int] = None) -> Chroma:
    if (Path(".") / name / "chroma.sqlite3").exists():
        return read_db(persist_directory=name, device=device)
    elif docs_path is None:
        raise ValueError(f"docs_path can't be None: vectorstore {name} doesn't exists.")
    
    splits = read_docs(docs_path)
    return build_db(splits, name, device)


def hash_from_str_list(strings: List[str]) -> str:
    enc_strings = map(str.encode, strings)
    return hashlib.sha256(b"".join(enc_strings)).hexdigest()