"""Runnable checks for cache isolation, LRU/TTL, immutability and signature compatibility."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
from core.query_cache import EmbeddingCache
from core.config_loader import QueryCache, load_config, index_signature
from core.experiment_cache import signature


def main():
    cache = EmbeddingCache()
    settings = QueryCache(enabled=True, max_entries=2, ttl_seconds=10)
    vector = np.asarray([1, 2, 3], dtype=np.float32)
    with patch('core.query_cache.monotonic', return_value=100):
        cache.put(('scope', 'model', 'q1'), vector, settings)
        vector[0] = 99
        first, _ = cache.get(('scope', 'model', 'q1'))
        assert first[0] == 1
        first[0] = 88
        assert cache.get(('scope', 'model', 'q1'))[0][0] == 1
        assert cache.get(('other', 'model', 'q1'))[0] is None
        assert cache.get(('scope', 'different-model', 'q1'))[0] is None
        cache.put('q2', vector, settings)
        cache.get(('scope', 'model', 'q1'))
        assert cache.put('q3', vector, settings)['evicted_entries'] == 1
        assert cache.get('q2')[0] is None
        assert cache.vector_bytes == 2 * vector.nbytes
    with patch('core.query_cache.monotonic', return_value=111):
        assert cache.get('q3')[1]['expired_entries'] == 1
        assert cache.get('q3')[0] is None
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: cache.put(str(i), vector, settings), range(100)))
    assert len(cache.entries) == 2 and cache.vector_bytes == 2 * vector.nbytes
    config = load_config()
    raw = config.model_dump()
    old = config.model_dump()
    old['vector_search'].pop('query_cache')
    assert signature(raw, 'evaluation', 'project', {}) == signature(old, 'evaluation', 'project', {})
    original_index = index_signature(config)
    config.vector_search.query_cache.enabled = True
    assert index_signature(config) == original_index
    assert signature(config.model_dump(), 'evaluation', 'project', {}) != signature(old, 'evaluation', 'project', {})
    print('PASS: isolation, immutable vectors, TTL, LRU, concurrent bounds and legacy signatures')


if __name__ == '__main__':
    main()
