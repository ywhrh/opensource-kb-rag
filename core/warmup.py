"""Warm local retrieval models once per process without hosted API calls."""
from threading import Lock
from core.models import embed
from retrieval.reranker import rerank

_lock = Lock()
_ready = set()


def warmup(config):
    key = (config.vector_search.model_name, config.vector_search.query_device,
           config.retrieval_pipeline.mode, config.rerank.model_name,
           config.rerank.optimization.use_onnx_acceleration)
    with _lock:
        if key in _ready:
            return
        query = 'How does tokenization work?'
        if config.vector_search.model_name != 'text-embedding-3-small':
            local = config.model_copy(deep=True)
            local.vector_search.query_cache.enabled = False
            local.vector_search.query_batching.enabled = False
            embed([query], local, query=True)
        if (config.retrieval_pipeline.mode == 'hybrid_rerank'
                and config.rerank.model_name != 'Cohere Rerank v3'):
            rerank(query, [{'text': 'A tokenizer converts text into tokens.', 'score': 1.0}],
                   config.rerank)
        _ready.add(key)
