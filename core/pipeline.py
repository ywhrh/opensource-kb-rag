"""Shared retrieval/generation flow; Live Run never invokes evaluation or load testing."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter, time
from uuid import uuid4

from core.history import new_output, persist
from core.logger import event
from core.profiling import DIAGNOSTICS_VERSION, timed, record_trace
from core.models import SYSTEM_PROMPT, chat_tokens
from ingestion.vector_store_writer import require_index
from retrieval.vector_search import vector_search
from retrieval.bm25_search import bm25_search
from retrieval.hybrid_fusion import fuse
from retrieval.reranker import rerank

SEARCH_POOL_WORKERS = 4
_SEARCH_POOL = ThreadPoolExecutor(max_workers=SEARCH_POOL_WORKERS, thread_name_prefix='retrieval')


def _search_task(submitted, name, stage_timings, operation, *args, **kwargs):
    stage_timings[name + '_queue_wait_ms'] = (perf_counter() - submitted) * 1000
    with timed(stage_timings, name + '_task_ms'):
        return operation(*args, **kwargs)


def retrieve(query, source_id, config, credentials=None, expected_collection=None, diagnostics=None):
    trace = diagnostics if diagnostics is not None else {'request_id': uuid4().hex}
    trace['embedding_device'] = config.vector_search.query_device
    trace.update(version=DIAGNOSTICS_VERSION, timestamp=time(), mode=config.retrieval_pipeline.mode)
    timings = trace.setdefault('timings_ms', {})
    started = perf_counter()
    try:
        with timed(timings, 'index_validation_ms'):
            manifest = require_index(config, source_id)
            if expected_collection and manifest['collection_name'] != expected_collection:
                raise ValueError('The index changed during this experiment. Start a new run using the updated index.')
        retrieval_started = perf_counter()
        if config.retrieval_pipeline.mode == 'vector_only':
            with timed(timings, 'vector_task_ms'):
                candidates = vector_search(query, manifest['collection_name'], config, credentials, timings=timings)
        else:
            dense = _SEARCH_POOL.submit(_search_task, perf_counter(), 'vector', timings,
                vector_search, query, manifest['collection_name'], config, credentials, timings=timings)
            sparse = _SEARCH_POOL.submit(_search_task, perf_counter(), 'bm25', timings,
                bm25_search, query, manifest['collection_name'], config)
            # Await both tasks, including on failure, before logging the shared timing map.
            dense_error = None
            try:
                dense_rows = dense.result()
            except Exception as exc:
                dense_error = exc
            sparse_rows = sparse.result()
            if dense_error:
                raise dense_error
            with timed(timings, 'fusion_ms'):
                candidates = fuse(dense_rows, sparse_rows, config.hybrid_fusion)
        trace['candidate_count_before_rerank'] = len(candidates)
        if config.retrieval_pipeline.mode == 'hybrid_rerank':
            with timed(timings, 'rerank_ms'):
                candidates = rerank(query, candidates, config.rerank, credentials)
        selected, used = [], 0
        with timed(timings, 'evidence_selection_ms'):
            for row in candidates:
                if len(selected) >= config.generation.context_top_k:
                    break
                if used + len(row['text']) > config.generation.max_context_chars:
                    continue
                selected.append(deepcopy(row))
                used += len(row['text'])
        elapsed = (perf_counter() - retrieval_started) * 1000
        trace.update(status='SUCCESS', evidence_count=len(selected), collection=manifest['collection_name'])
        return {'evidence_chunks': selected, 'retrieval_latency_ms': elapsed, 'knowledge_base': manifest,
                'retrieved_candidates': len(candidates), 'diagnostics': trace}
    except Exception as exc:
        import traceback
        from pathlib import Path
        trace.update(status='FAILED', error_type=type(exc).__name__,
                     error_frames=[{'file': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
                                   for frame in traceback.extract_tb(exc.__traceback__)][-12:])
        raise
    finally:
        timings['server_retrieve_ms'] = (perf_counter() - started) * 1000
        if config.vector_search.query_device == 'mps':
            import torch
            trace['device_metrics'] = {'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                                       'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory()}
        record_trace(trace)


def run_events(query, source_id, config, credentials=None, save=True, expected_collection=None):
    config = config.model_copy(deep=True)
    if not config.experiment.name.strip():
        raise ValueError('Experiment name is required')
    if not query.strip():
        raise ValueError('Question is required')
    record = new_output(config, query=query, source_id=source_id, knowledge_base=require_index(config, source_id))
    if save:
        persist(record, initial=True)
    start = perf_counter()
    try:
        yield {'type': 'status', 'message': 'Retrieving evidence'}
        result = retrieve(query, source_id, config, credentials, expected_collection or record["knowledge_base"]["collection_name"])
        record['knowledge_base'] = result['knowledge_base']
        record['results'].update({key: result[key] for key in ['evidence_chunks', 'retrieval_latency_ms', 'retrieved_candidates']})
        yield {'type': 'evidence', 'evidence_chunks': result['evidence_chunks']}
        generation_start = perf_counter()
        if not result['evidence_chunks']:
            answer = 'The available documentation does not contain sufficient evidence to answer this question.'
            yield {'type': 'token', 'text': answer}
        else:
            context = '\n\n'.join(f'[{i}] Document {row["doc_id"]}, section {row["section_title"]}\n{row["text"]}' for i, row in enumerate(result['evidence_chunks'], 1))
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': f'QUESTION:\n{query}\n\nREFERENCE EVIDENCE:\n{context}'}]
            answer = ''
            for token in chat_tokens(messages, config, credentials):
                answer += token
                yield {'type': 'token', 'text': token}
        record['results'].update(status='SUCCESS', answer=answer,
                                  generation_latency_ms=(perf_counter() - generation_start) * 1000,
                                  total_latency_ms=(perf_counter() - start) * 1000)
        event('run_completed', run_id=record['run_id'], latency_ms=record['results']['total_latency_ms'])
    except GeneratorExit:
        record['results'].update(status='CANCELLED', error='Client disconnected before completion')
        if save:
            persist(record)
        raise
    except Exception as exc:
        record['results']['status'] = 'FAILED'
        # Deliberately avoid provider response bodies, which can include private input.
        message = str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__}: pipeline failed; check local service availability and model setup'
        record['results']['error'] = message
        record['results']['total_latency_ms'] = (perf_counter() - start) * 1000
        event('run_failed', run_id=record['run_id'], error_type=type(exc).__name__)
    if save:
        persist(record)
    yield {'type': 'result', 'output': record}


def run_query(query, source_id, config, credentials=None, save=True, expected_collection=None):
    for item in run_events(query, source_id, config, credentials, save, expected_collection):
        if item['type'] == 'result':
            return item['output']
