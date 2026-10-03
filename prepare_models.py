"""Download the configured local embedding model and prepare the BGE reranker."""
from core.config_loader import load_config
from core.models import embedding_model
from retrieval.reranker import local_reranker

if __name__ == '__main__':
    config = load_config()
    if config.vector_search.model_name.startswith('BAAI/'):
        print('Preparing embedding model: ' + config.vector_search.model_name, flush=True)
        embedding_model(config.vector_search.model_name)
    if config.rerank.model_name == 'bge-reranker-base':
        print('Preparing BGE reranker', flush=True)
        local_reranker(config.rerank.optimization.use_onnx_acceleration)
    print('Configured local models are ready.', flush=True)
