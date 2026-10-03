"""BGE cross-encoder with cached ONNX export, or explicitly key-gated Cohere v3."""
from functools import lru_cache
from threading import RLock
import httpx
from core.config_loader import ROOT
from core.models import provider_key

_LOCK = RLock()


@lru_cache(maxsize=2)
def local_reranker(onnx):
    from sentence_transformers import CrossEncoder
    with _LOCK:
        directory = ROOT / 'data' / 'models' / ('bge-reranker-base-onnx' if onnx else 'bge-reranker-base-torch')
        source = str(directory) if (directory / 'config.json').exists() else 'BAAI/bge-reranker-base'
        model = CrossEncoder(source, backend='onnx' if onnx else 'torch', device='cpu', max_length=512)
        if not directory.exists():
            model.save_pretrained(str(directory))
        return model


def rerank(query, candidates, config, credentials=None):
    if not candidates:
        return []
    if config.model_name == 'Cohere Rerank v3':
        response = httpx.post('https://api.cohere.com/v2/rerank', timeout=120,
                              headers={'Authorization': 'Bearer ' + provider_key(credentials, 'cohere')},
                              json={'model': 'rerank-english-v3.0', 'query': query,
                                    'documents': [row['text'] for row in candidates], 'top_n': len(candidates)})
        response.raise_for_status()
        ranks = [(item['index'], float(item['relevance_score'])) for item in response.json()['results']]
    else:
        import torch
        model = local_reranker(config.optimization.use_onnx_acceleration)
        scores = model.predict([(query, row['text']) for row in candidates], batch_size=config.optimization.batch_size,
                               show_progress_bar=False, convert_to_numpy=True, activation_fn=torch.nn.Sigmoid())
        ranks = [(i, float(score)) for i, score in enumerate(scores)]
        ranks.sort(key=lambda item: -item[1])
    return [dict(candidates[i], fusion_score=candidates[i]['score'], rerank_score=score, score=score, score_type='rerank') for i, score in ranks[:config.top_n]]
