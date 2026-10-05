"""Validated, atomic YAML configuration shared by the API, UI and CLI tools."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config.yaml'
load_dotenv(ROOT / '.env', override=False)
os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True)


class Experiment(Model):
    name: str = Field(default='', max_length=160)
    description: str = Field(default='', max_length=4000)


class Cleaning(Model):
    remove_html_tags: bool = True
    min_chunk_length: int = Field(default=50, ge=1)
    max_chunk_length: int = Field(default=1000, ge=50, le=50000)
    language_filter: list[str] = Field(default_factory=lambda: ['en'])
    deduplication_enabled: bool = True

    @model_validator(mode='after')
    def lengths(self):
        if self.min_chunk_length > self.max_chunk_length:
            raise ValueError('min_chunk_length must not exceed max_chunk_length')
        return self


class FixedChunking(Model):
    chunk_size: int = Field(default=500, ge=50, le=50000)
    chunk_overlap: int = Field(default=50, ge=0)

    @model_validator(mode='after')
    def overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlap must be smaller than chunk_size')
        return self


class SemanticChunking(Model):
    breakpoint_threshold_type: Literal['percentile', 'standard_deviation', 'interquartile'] = 'percentile'
    breakpoint_threshold_amount: float = Field(default=90, ge=0)
    buffer_size: int = Field(default=1, ge=0, le=10)

    @model_validator(mode='after')
    def threshold(self):
        if self.breakpoint_threshold_type == 'percentile' and self.breakpoint_threshold_amount > 100:
            raise ValueError('Percentile must be between 0 and 100')
        return self


class Chunking(Model):
    strategy: Literal['fixed_size', 'semantic', 'recursive'] = 'semantic'
    fixed_size_config: FixedChunking = Field(default_factory=FixedChunking)
    semantic_config: SemanticChunking = Field(default_factory=SemanticChunking)


class Enrichment(Model):
    inject_doc_title: bool = True
    extract_entities: bool = True
    entity_types: list[Literal['API', 'VERSION', 'CODE']] = Field(default_factory=lambda: ['API', 'VERSION', 'CODE'])
    generate_hypothetical_questions: bool = False


class Ingestion(Model):
    cleaning: Cleaning = Field(default_factory=Cleaning)
    chunking: Chunking = Field(default_factory=Chunking)
    metadata_enrichment: Enrichment = Field(default_factory=Enrichment)


class RetrievalMode(Model):
    mode: Literal['vector_only', 'hybrid', 'hybrid_rerank'] = 'hybrid_rerank'


class HNSW(Model):
    efConstruction: int = Field(default=200, ge=1, le=4096)
    ef: int = Field(default=64, ge=1, le=4096)
    M: int = Field(default=16, ge=2, le=128)


class IVF(Model):
    nlist: int = Field(default=128, ge=1, le=65536)
    nprobe: int = Field(default=16, ge=1)

    @model_validator(mode='after')
    def probes(self):
        if self.nprobe > self.nlist:
            raise ValueError('nprobe must not exceed nlist')
        return self


class Quantization(Model):
    type: Literal['none', 'SQ8', 'BINARY'] = 'SQ8'
    enabled: bool = False


class QueryBatching(Model):
    enabled: bool = False
    max_size: int = Field(default=8, ge=1, le=32)
    wait_ms: float = Field(default=10, ge=0, le=100)
    queue_capacity: int = Field(default=128, ge=1, le=1024)
    timeout_seconds: float = Field(default=30, gt=0, le=120)


class QueryCache(Model):
    enabled: bool = False
    max_entries: int = Field(default=1024, ge=1, le=4096)
    ttl_seconds: float = Field(default=300, gt=0, le=86400)


class VectorSearch(Model):
    query_cache: QueryCache = Field(default_factory=QueryCache)
    query_batching: QueryBatching = Field(default_factory=QueryBatching)
    query_device: Literal['cpu', 'mps'] = 'cpu'  # Query inference only; index embeddings stay unchanged.
    enabled: bool = True
    model_name: Literal['BAAI/bge-large-en-v1.5', 'BAAI/bge-m3', 'text-embedding-3-small'] = 'BAAI/bge-large-en-v1.5'
    top_k: int = Field(default=50, ge=1, le=1000)
    index_type: Literal['HNSW', 'IVF_FLAT', 'IVF_SQ8'] = 'HNSW'
    metric_type: Literal['COSINE', 'L2', 'IP'] = 'COSINE'
    hnsw_params: HNSW = Field(default_factory=HNSW)
    ivf_params: IVF = Field(default_factory=IVF)
    quantization: Quantization = Field(default_factory=Quantization)

    @model_validator(mode='after')
    def compatible(self):
        if self.query_cache.enabled and self.model_name == 'text-embedding-3-small':
            raise ValueError('Query embedding cache currently supports local models only')
        if self.query_batching.enabled and self.model_name == 'text-embedding-3-small':
            raise ValueError('Dynamic query batching requires a local embedding model')
        if not self.enabled:
            raise ValueError('All supported retrieval modes require vector search')
        if self.index_type == 'HNSW' and self.hnsw_params.ef < self.top_k:
            raise ValueError('HNSW ef must be at least vector top_k')
        if self.quantization.enabled and self.quantization.type == 'BINARY':
            raise ValueError('BINARY requires binary embeddings and a binary metric; configured embedding models return float vectors')
        if self.quantization.enabled and self.quantization.type == 'SQ8' and self.index_type == 'IVF_FLAT':
            raise ValueError('Choose IVF_SQ8 or HNSW with SQ8 instead of IVF_FLAT with SQ8')
        return self


class BM25(Model):
    enabled: bool = True
    model_name: Literal['BM25Okapi', 'Milvus Built-in Sparse BM25'] = 'BM25Okapi'
    top_k: int = Field(default=50, ge=1, le=1000)
    tokenizer: Literal['standard'] = 'standard'


class Weights(Model):
    vector_weight: float = Field(default=0.7, ge=0, le=1)
    bm25_weight: float = Field(default=0.3, ge=0, le=1)

    @model_validator(mode='after')
    def normalized(self):
        if abs(self.vector_weight + self.bm25_weight - 1) > 1e-6:
            raise ValueError('Fusion weights must sum to 1')
        return self


class Fusion(Model):
    strategy: Literal['weighted_sum', 'RRF'] = 'weighted_sum'
    weights: Weights = Field(default_factory=Weights)
    rrf_k: int = Field(default=60, ge=1)


class Optimization(Model):
    use_onnx_acceleration: bool = True
    batch_size: int = Field(default=16, ge=1, le=256)


class Rerank(Model):
    enabled: bool = True
    model_name: Literal['bge-reranker-base', 'Cohere Rerank v3'] = 'bge-reranker-base'
    top_n: int = Field(default=20, ge=1, le=2000)
    optimization: Optimization = Field(default_factory=Optimization)


class Security(Model):
    enabled: bool = False
    filter_strategy: str = 'milvus_expr_pre_filter'
    test_scenario: str = 'cross_department_access'


class Generation(Model):
    llm_model: Literal['llama3.1:8b', 'gpt-4o-mini'] = 'llama3.1:8b'
    provider: Literal['ollama', 'openai'] = 'ollama'
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=16384)
    context_top_k: int = Field(default=5, ge=1, le=50)
    max_context_chars: int = Field(default=16000, ge=1000, le=100000)
    seed: int = Field(default=42, ge=0)


class EvaluationSLA(Model):
    max_p99_latency_ms: float = Field(default=30, gt=0)
    min_context_recall: float = Field(default=0.9, ge=0, le=1)


class Evaluation(Model):
    dataset_path: str = './data/golden_test_dataset.json'
    judge_model: Literal['llama3.1:8b', 'gpt-4o-mini'] = 'llama3.1:8b'
    judge_max_tokens: int = Field(default=4096, ge=512, le=16384)
    metrics: list[Literal['context_recall', 'context_precision', 'faithfulness', 'answer_correctness']] = Field(default_factory=lambda: ['context_recall', 'context_precision', 'faithfulness', 'answer_correctness'])
    sla_targets: EvaluationSLA = Field(default_factory=EvaluationSLA)


class LoadParameters(Model):
    users: int = Field(default=50, ge=1, le=1000)
    spawn_rate: float = Field(default=5, gt=0, le=1000)
    run_time: str = Field(default='5m', pattern=r'^(?:\d+h)?(?:\d+m)?(?:\d+s)?$')

    @model_validator(mode='after')
    def duration(self):
        import re
        seconds = sum(int(n) * {'h': 3600, 'm': 60, 's': 1}[u] for n, u in re.findall(r'(\d+)([hms])', self.run_time))
        if not 1 <= seconds <= 3600:
            raise ValueError('Load tests must run between 1 second and 1 hour')
        return self


class LoadSLA(Model):
    target_min_qps: float = Field(default=50, gt=0)
    max_p99_latency_ms: float = Field(default=30, gt=0)


class LoadTesting(Model):
    tool: Literal['locust'] = 'locust'
    target_host: str = 'http://localhost:8000'
    endpoint: Literal['/retrieve', '/query'] = '/retrieve'
    parameters: LoadParameters = Field(default_factory=LoadParameters)
    sla_targets: LoadSLA = Field(default_factory=LoadSLA)

    @model_validator(mode='after')
    def local_host(self):
        from urllib.parse import urlparse
        url = urlparse(self.target_host)
        if url.scheme != 'http' or url.hostname not in {'localhost', '127.0.0.1'} or url.username or url.password or url.path not in {'', '/'} or url.query or url.fragment:
            raise ValueError('This local studio load-tests localhost HTTP services only')
        return self


class RAGConfig(Model):
    experiment: Experiment = Field(default_factory=Experiment)
    ingestion_pipeline: Ingestion = Field(default_factory=Ingestion)
    retrieval_pipeline: RetrievalMode = Field(default_factory=RetrievalMode)
    vector_search: VectorSearch = Field(default_factory=VectorSearch)
    bm25_search: BM25 = Field(default_factory=BM25)
    hybrid_fusion: Fusion = Field(default_factory=Fusion)
    rerank: Rerank = Field(default_factory=Rerank)
    security_and_permission: Security = Field(default_factory=Security)
    generation: Generation = Field(default_factory=Generation)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    load_testing: LoadTesting = Field(default_factory=LoadTesting)

    @model_validator(mode='after')
    def effective_switches(self):
        self.bm25_search.enabled = self.retrieval_pipeline.mode != 'vector_only'
        self.rerank.enabled = self.retrieval_pipeline.mode == 'hybrid_rerank'
        self.generation.provider = 'openai' if self.generation.llm_model == 'gpt-4o-mini' else 'ollama'
        if self.security_and_permission.enabled:
            raise ValueError('Permission features are deferred; security_and_permission.enabled must remain false')
        return self


def revision():
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def load_config():
    return RAGConfig.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))


@contextmanager
def config_lock():
    with (ROOT / '.config.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def atomic_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def save_config(config, expected_revision=None):
    config = RAGConfig.model_validate(config)
    with config_lock():
        if expected_revision and revision() != expected_revision:
            raise ValueError('Configuration changed in another session. Reload before saving.')
        atomic_yaml(CONFIG_PATH, config.model_dump())
        return revision()


def index_signature(config):
    vector = config.vector_search
    return {
        'ingestion': config.ingestion_pipeline.model_dump(),
        'embedding': vector.model_name,
        'index': vector.index_type,
        'metric': vector.metric_type,
        'hnsw_build': {'M': vector.hnsw_params.M, 'efConstruction': vector.hnsw_params.efConstruction},
        'ivf_build': {'nlist': vector.ivf_params.nlist},
        'quantization': vector.quantization.model_dump(),
    }


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
