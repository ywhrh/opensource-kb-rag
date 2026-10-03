"""Build immutable Milvus collections; publish only completely ingested snapshots."""
from contextlib import contextmanager
import fcntl
import os
from threading import RLock
from uuid import uuid4
import yaml

from pymilvus import DataType, Function, FunctionType, MilvusClient

from core.config_loader import ROOT, atomic_yaml, digest, index_signature
from core.models import embed
from ingestion.cleaner import Deduplicator, clean_text, permitted_language
from ingestion.enricher import enrich
from ingestion.loader import document_paths, source_directory, source_revision, load_document
from ingestion.splitter import split_document

_LOCK = RLock()
_CLIENT = None
FIELDS = ['chunk_id', 'doc_id', 'section_title', 'text', 'metadata']


def client():
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            _CLIENT = MilvusClient(uri=os.getenv('MILVUS_URI', 'http://127.0.0.1:19530'),
                                   token=os.getenv('MILVUS_TOKEN', ''), timeout=30)
        return _CLIENT


def manifest_path(config, source_id):
    key = digest({'source_id': source_id, 'index_signature': index_signature(config)})
    return ROOT / 'data' / 'collections' / f'{key}.yaml'


def active_manifest(config, source_id):
    path = manifest_path(config, source_id)
    if not path.exists():
        return None
    manifest = yaml.safe_load(path.read_text())
    if not client().has_collection(manifest['collection_name']):
        raise ValueError('The recorded Milvus collection is missing. Reindex this document source.')
    return manifest


@contextmanager
def ingestion_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def create_collection(name, config):
    store = client()
    schema = store.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field('chunk_id', DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field('doc_id', DataType.VARCHAR, max_length=128)
    schema.add_field('section_title', DataType.VARCHAR, max_length=4096)
    schema.add_field('text', DataType.VARCHAR, max_length=65535)
    schema.add_field('search_text', DataType.VARCHAR, max_length=65535, enable_analyzer=True,
                     analyzer_params={'type': 'standard'})
    schema.add_field('metadata', DataType.JSON)
    dimension = 1536 if config.vector_search.model_name == 'text-embedding-3-small' else 1024
    schema.add_field('dense', DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field('sparse', DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name='text_bm25', function_type=FunctionType.BM25,
                                 input_field_names=['search_text'], output_field_names=['sparse']))
    indexes = store.prepare_index_params()
    vector = config.vector_search
    index_type = vector.index_type
    if index_type == 'HNSW':
        params = {'M': vector.hnsw_params.M, 'efConstruction': vector.hnsw_params.efConstruction}
        if vector.quantization.enabled and vector.quantization.type == 'SQ8':
            index_type = 'HNSW_SQ'
            params['sq_type'] = 'SQ8'
    else:
        params = {'nlist': vector.ivf_params.nlist}
    indexes.add_index(field_name='dense', index_type=index_type, metric_type=vector.metric_type, params=params)
    indexes.add_index(field_name='sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25',
                      params={'inverted_index_algo': 'DAAT_MAXSCORE'})
    store.create_collection(collection_name=name, schema=schema, index_params=indexes, consistency_level='Strong')


def index_source(config, source_id='project', credentials=None, progress=None):
    paths = document_paths(source_id)
    if not paths:
        raise ValueError('No documents found. Add Markdown, TXT, PDF, DOCX or HTML files to data/documents, or upload documents.')
    directory = source_directory(source_id)
    source_hash = source_revision(paths, directory)
    path = manifest_path(config, source_id)
    with ingestion_lock(path):
        existing = active_manifest(config, source_id)
        if existing and existing['source_revision'] == source_hash:
            return existing
        name = 'rag_' + digest({'source': source_id, 'revision': source_hash, 'config': index_signature(config)})[:24] + '_' + uuid4().hex[:8]
        create_collection(name, config)
        chunks, skipped, batch = 0, 0, []
        dedup = Deduplicator() if config.ingestion_pipeline.cleaning.deduplication_enabled else None

        def flush():
            nonlocal chunks
            if not batch:
                return
            vectors = embed([item['embedding_text'] for item in batch], config, credentials)
            rows = []
            for item, vector in zip(batch, vectors):
                rows.append({key: value for key, value in item.items() if key != 'embedding_text'} |
                            {'search_text': item['embedding_text'], 'dense': vector.tolist()})
            client().upsert(collection_name=name, data=rows)
            chunks += len(rows)
            batch.clear()

        try:
            for position, file in enumerate(paths, 1):
                document = load_document(file, directory)
                text = clean_text(document.text, config.ingestion_pipeline.cleaning)
                if not permitted_language(text, config.ingestion_pipeline.cleaning.language_filter):
                    skipped += 1
                    continue
                for chunk in split_document(text, config, credentials):
                    if len(chunk['text']) < config.ingestion_pipeline.cleaning.min_chunk_length:
                        skipped += 1
                        continue
                    if dedup and dedup.duplicate(chunk['text']):
                        skipped += 1
                        continue
                    item = enrich(chunk, document, config, credentials)
                    if len(item['text'].encode()) > 65535 or len(item['embedding_text'].encode()) > 65535:
                        raise ValueError(f'{file.name}: a protected code block exceeds Milvus VARCHAR capacity; split the source code block explicitly')
                    batch.append(item)
                    if len(batch) >= 64:
                        flush()
                if progress:
                    progress(f'Processed {position}/{len(paths)} documents; {chunks + len(batch)} chunks')
            flush()
            if chunks == 0:
                raise ValueError('Cleaning and filtering produced no usable chunks. Adjust the ingestion settings.')
            if source_revision(paths, directory) != source_hash:
                raise ValueError('Documents changed while indexing. Retry with a stable document set.')
            client().flush(collection_name=name, timeout=300)
            client().load_collection(collection_name=name, timeout=300)
            manifest = {'collection_name': name, 'source_id': source_id, 'source_revision': source_hash,
                        'index_signature': index_signature(config), 'documents': len(paths), 'chunks': chunks,
                        'skipped_chunks_or_documents': skipped}
            atomic_yaml(path, manifest)
            return manifest
        except Exception:
            # Only this build's unpublished collection is removed; prior snapshots remain intact.
            try:
                client().drop_collection(collection_name=name)
            except Exception:
                pass
            raise


def require_index(config, source_id):
    manifest = active_manifest(config, source_id)
    if manifest is None:
        raise ValueError('No compatible indexed knowledge base. Click Index documents for the current data source and configuration.')
    return manifest


def iter_chunks(collection_name):
    iterator = client().query_iterator(collection_name=collection_name, batch_size=1000, output_fields=FIELDS, filter='')
    try:
        while True:
            rows = iterator.next()
            if not rows:
                return
            yield from rows
    finally:
        iterator.close()
