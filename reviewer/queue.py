import logging
import threading
from collections import OrderedDict
from typing import Callable

from reviewer import config


class DedupeCache:
    """LRU set of diff fingerprints already reviewed."""

    def __init__(self, maxsize: int = None):
        self._maxsize = maxsize or config.DEDUPE_CACHE_SIZE
        self._entries: OrderedDict[str, bool] = OrderedDict()
        self._lock = threading.Lock()

    def seen(self, fingerprint: str) -> bool:
        with self._lock:
            if fingerprint in self._entries:
                self._entries.move_to_end(fingerprint)
                return True
            return False

    def remember(self, fingerprint: str) -> None:
        with self._lock:
            self._entries[fingerprint] = True
            self._entries.move_to_end(fingerprint)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)


class ReviewQueue:
    """FIFO queue that coalesces repeat submissions for the same key.

    Replaces the old threading.Lock, which could not serialize across
    gunicorn worker processes (defect B3).
    """

    def __init__(self, maxsize: int = None):
        self._maxsize = maxsize or config.QUEUE_MAXSIZE
        self._items: OrderedDict[str, object] = OrderedDict()
        self._cv = threading.Condition()

    def submit(self, key: str, job: object) -> bool:
        with self._cv:
            if key in self._items:
                # Keep queue position, replace payload with the newer one.
                self._items[key] = job
                logging.info("Coalesced queued job %s", key)
                return True
            if len(self._items) >= self._maxsize:
                logging.warning("Review queue full (%s); rejecting %s", self._maxsize, key)
                return False
            self._items[key] = job
            self._cv.notify()
            return True

    def take(self) -> object:
        with self._cv:
            while not self._items:
                self._cv.wait()
            _, job = self._items.popitem(last=False)
            return job

    def size(self) -> int:
        with self._cv:
            return len(self._items)


def start_worker(handler: Callable[[object], None]) -> ReviewQueue:
    """Starts one daemon consumer. Only ever one, to keep Ollama serialized."""
    queue = ReviewQueue()

    def loop() -> None:
        while True:
            job = queue.take()
            try:
                handler(job)
            except Exception as exc:
                logging.error("Worker handler failed for %r: %s", job, exc)

    threading.Thread(target=loop, name="review-worker", daemon=True).start()
    return queue
