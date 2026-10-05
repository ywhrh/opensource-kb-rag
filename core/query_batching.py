"""Bounded query micro-batches: one model invocation at a time per configuration."""
from concurrent.futures import Future, TimeoutError
from queue import Queue, Empty, Full
from threading import Lock, Thread
from time import perf_counter
from uuid import uuid4


class BatchQueueFull(ValueError):
    pass


class QueryBatcher:
    def __init__(self, encode, max_size, wait_ms, capacity):
        self.encode, self.max_size, self.wait_seconds = encode, max_size, wait_ms / 1000
        self.queue = Queue(maxsize=capacity)
        Thread(target=self.work, name='query-embedding-batcher', daemon=True).start()

    def submit(self, text, timeout):
        future = Future()
        item = (future, text, perf_counter(), self.queue.qsize())
        try:
            self.queue.put_nowait(item)
        except Full:
            raise BatchQueueFull('Query embedding queue is full') from None
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    def work(self):
        while True:
            first = self.queue.get()
            items = [first]
            collected_at = perf_counter()
            deadline = collected_at + self.wait_seconds
            while len(items) < self.max_size:
                remaining = deadline - perf_counter()
                try:
                    items.append(self.queue.get(timeout=remaining) if remaining > 0 else self.queue.get_nowait())
                except Empty:
                    break
            # Claim futures before inference; cancelled queued work is not encoded.
            active = [item for item in items if item[0].set_running_or_notify_cancel()]
            if not active:
                continue
            started = perf_counter()
            batch_id = uuid4().hex
            try:
                vectors = self.encode([item[1] for item in active])
                if len(vectors) != len(active):
                    raise ValueError('Batch vector count differs from query count')
                compute_ms = (perf_counter() - started) * 1000
                for index, ((future, _, queued_at, queue_depth), vector) in enumerate(zip(active, vectors)):
                    future.set_result((vector, {
                        'batch_id': batch_id, 'batch_size': len(active), 'batch_index': index,
                        'queue_depth_at_enqueue': queue_depth,
                        'queue_wait_ms': (started - queued_at) * 1000,
                        'collection_wait_ms': (started - collected_at) * 1000,
                        'compute_ms': compute_ms,
                    }))
            except Exception as exc:
                for future, *_ in active:
                    future.set_exception(exc)


_BATCHERS = {}
_BATCHERS_LOCK = Lock()


def batcher(key, encode, settings):
    # Serialize creation so simultaneous first queries cannot create duplicate workers.
    with _BATCHERS_LOCK:
        if key not in _BATCHERS:
            if len(_BATCHERS) >= 8:
                raise ValueError('Restart API before creating more than eight batch configurations')
            _BATCHERS[key] = QueryBatcher(encode, settings.max_size, settings.wait_ms, settings.queue_capacity)
        return _BATCHERS[key]
