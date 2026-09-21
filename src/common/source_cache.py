"""Per-run source cache with one in-flight reader per source identity."""

from concurrent.futures import Future
from threading import Lock


class SourceCache:
    def __init__(self):
        self._lock = Lock()
        self._entries = {}

    def read(self, key, loader):
        with self._lock:
            cached = key in self._entries
            future = self._entries.setdefault(key, Future())
        if not cached:
            try:
                future.set_result(loader())
            except BaseException as exc:
                future.set_exception(exc)
                if not isinstance(exc, Exception):
                    raise
        return future.result().model_copy(deep=True), cached
