import sys
from importlib import import_module


def upd_sqlite_version() -> None:
    import_module("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
