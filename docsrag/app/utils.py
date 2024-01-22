from importlib import import_module
import sys


def upd_sqlite_version():
    import_module("pysqlite3")
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')