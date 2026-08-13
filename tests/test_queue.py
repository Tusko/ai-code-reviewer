import threading
import time

from reviewer.queue import DedupeCache, ReviewQueue, start_worker


def test_take_returns_jobs_in_fifo_order():
    q = ReviewQueue(maxsize=4)
    q.submit("a", (1, 1))
    q.submit("b", (2, 2))
    assert q.take() == (1, 1)
    assert q.take() == (2, 2)


def test_submit_coalesces_same_key_keeping_newest_payload():
    q = ReviewQueue(maxsize=4)
    q.submit("mr-1", (1, "old"))
    q.submit("mr-2", (2, "other"))
    q.submit("mr-1", (1, "new"))
    assert q.size() == 2
    assert q.take() == (1, "new")   # keeps original position
    assert q.take() == (2, "other")


def test_submit_rejects_when_full():
    q = ReviewQueue(maxsize=2)
    assert q.submit("a", (1,)) is True
    assert q.submit("b", (2,)) is True
    assert q.submit("c", (3,)) is False


def test_dedupe_cache_remembers_and_evicts():
    cache = DedupeCache(maxsize=2)
    assert cache.seen("x") is False
    cache.remember("x")
    assert cache.seen("x") is True
    cache.remember("y")
    cache.remember("z")
    assert cache.seen("x") is False   # evicted
    assert cache.seen("z") is True


def test_worker_drains_queue():
    handled = []
    done = threading.Event()

    def handler(job):
        handled.append(job)
        done.set()

    q = start_worker(handler)
    q.submit("a", (7, 8))
    assert done.wait(timeout=5) is True
    assert handled == [(7, 8)]


def test_worker_survives_handler_exception():
    calls = []
    second = threading.Event()

    def handler(job):
        calls.append(job)
        if len(calls) == 1:
            raise RuntimeError("boom")
        second.set()

    q = start_worker(handler)
    q.submit("a", (1,))
    q.submit("b", (2,))
    assert second.wait(timeout=5) is True
    assert calls == [(1,), (2,)]
