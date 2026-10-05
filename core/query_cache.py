"""Bounded process-local LRU of query vectors, with TTL and metadata-only diagnostics."""
from collections import OrderedDict
from threading import Lock
from time import monotonic


class EmbeddingCache:
    def __init__(self):
        self.entries = OrderedDict()
        self.lock = Lock()
        self.vector_bytes = 0

    def get(self, key):
        with self.lock:
            expired = 0
            entry = self.entries.get(key)
            if entry and entry[0] <= monotonic():
                self.entries.pop(key)
                self.vector_bytes -= entry[1].nbytes
                entry = None
                expired = 1
            if entry:
                self.entries.move_to_end(key)
            return (entry[1].copy() if entry else None), {
                'hit': entry is not None, 'expired_entries': expired,
                'entries': len(self.entries), 'vector_bytes': self.vector_bytes,
                'evicted_entries': 0, 'inference_executed': False,
            }

    def put(self, key, vector, settings):
        # Store our own array so callers cannot mutate future cached results.
        stored = vector.copy()
        stored.flags.writeable = False
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous:
                self.vector_bytes -= previous[1].nbytes
            self.entries[key] = (monotonic() + settings.ttl_seconds, stored)
            self.vector_bytes += stored.nbytes
            evicted = 0
            while len(self.entries) > settings.max_entries:
                _, (_, old) = self.entries.popitem(last=False)
                self.vector_bytes -= old.nbytes
                evicted += 1
            return {'entries': len(self.entries), 'vector_bytes': self.vector_bytes,
                    'evicted_entries': evicted, 'inference_executed': True}


# ponytail: one API process; each worker has a separate cache if deployment adds workers.
# Concurrent cold misses can infer the same query; this round does not add single-flight.
QUERY_CACHE = EmbeddingCache()
