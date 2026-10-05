"""Local HTTP API for config sync, ingestion, streaming RAG and independent tuning jobs."""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from time import perf_counter
import anyio
from pathlib import Path
import shutil
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import yaml

from api.schemas import ConfigUpdate, RunRequest, EvaluationRequest, IngestRequest, LoadTestRequest, Credentials
from core.config_loader import ROOT, load_config, revision, save_config, config_lock
from core.history import folder
from core.jobs import submit, get_job
from core.load_testing import run_load_test
from core.pipeline import retrieve, run_events, run_query, SEARCH_POOL_WORKERS
from ingestion.loader import SUPPORTED
from ingestion.vector_store_writer import index_source, client, require_index

from core.profiling import source_metadata

# Capture once at import so diagnostics describe this service process, not later disk edits.
BACKEND_SOURCE = source_metadata()
router = APIRouter()


def snapshot(request):
    config = request.config_snapshot or load_config()
    if not config.experiment.name.strip():
        raise HTTPException(422, 'Experiment name is required')
    return config.model_copy(deep=True)


def bad_request(exc):
    return HTTPException(400, str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__}: operation failed; check local service setup')


@router.get('/config')
def get_config():
    with config_lock():
        return {'config': load_config().model_dump(), 'revision': revision()}


@router.put('/config')
def put_config(request: ConfigUpdate):
    try:
        updated = save_config(request.config, request.expected_revision)
        return {'config': request.config.model_dump(), 'revision': updated}
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get('/health')
def health():
    services = {}
    try:
        services['milvus'] = {'ready': True, 'collections': len(client().list_collections(timeout=3))}
    except Exception:
        services['milvus'] = {'ready': False}
    try:
        response = httpx.get(os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434') + '/api/tags', timeout=3)
        response.raise_for_status()
        models = [model['name'] for model in response.json()['models']]
        services['ollama'] = {'ready': True, 'models': models, 'default_model_ready': 'llama3.1:8b' in models}
    except Exception:
        services['ollama'] = {'ready': False, 'default_model_ready': False}
    return {'api': 'ready', 'services': services, 'permissions': 'deferred'}


@router.post('/documents')
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files or len(files) > 100:
        raise HTTPException(422, 'Upload between 1 and 100 documents')
    temporary = ROOT / 'data' / 'uploads' / ('pending_' + uuid4().hex)
    temporary.mkdir(parents=True)
    names, total = set(), 0
    hasher = hashlib.sha256()
    try:
        for file in sorted(files, key=lambda item: item.filename or ''):
            name = Path(file.filename or '').name
            if not name or name in names or Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(422, 'Document names must be unique and use a supported extension')
            names.add(name)
            hasher.update(name.encode())
            size = 0
            with (temporary / name).open('wb') as destination:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    total += len(block)
                    if size > 20 * 1024 * 1024 or total > 200 * 1024 * 1024:
                        raise HTTPException(413, 'Limit: 20 MB per document and 200 MB per upload set')
                    hasher.update(block)
                    destination.write(block)
            if size == 0:
                raise HTTPException(422, 'Documents must not be empty')
        identifier = 'upload_' + hasher.hexdigest()[:32]
        destination = temporary.parent / identifier
        if not destination.exists():
            temporary.rename(destination)
        return {'source_id': identifier, 'documents': sorted(names), 'bytes': total}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        for file in files:
            await file.close()


@router.post('/ingest')
def ingest(request: IngestRequest):
    config = (request.config_snapshot or load_config()).model_copy(deep=True)
    try:
        return submit('ingestion', lambda progress: index_source(config, request.source_id, request.credentials, progress))
    except Exception as exc:
        raise bad_request(exc)


@router.get('/diagnostics')
async def diagnostics_info():
    import platform
    import psutil
    from core.profiling import DIAGNOSTICS_VERSION
    limiter = anyio.to_thread.current_default_thread_limiter()
    return {'diagnostics_version': DIAGNOSTICS_VERSION, 'backend_source': BACKEND_SOURCE, 'api_pid': os.getpid(),
            'logical_cpus': psutil.cpu_count(), 'memory_bytes': psutil.virtual_memory().total,
            'platform': platform.platform(), 'api_worker_tokens': limiter.total_tokens,
            'search_pool_workers': SEARCH_POOL_WORKERS, 'pid_scope': 'local API process',
            'timing_scope': 'worker dispatch wait plus server retrieval stages; HTTP serialization is not a retrieval stage'}


@router.post('/retrieve')
async def retrieval(request: RunRequest, http_request: Request):
    run_id = http_request.headers.get('X-Load-Test-Id', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', run_id):
        run_id = None
    request_id = http_request.headers.get('X-Request-Id', '')
    if not re.fullmatch(r'[a-f0-9]{32}', request_id):
        request_id = uuid4().hex
    trace = {'request_id': request_id, 'load_test_id': run_id, 'timings_ms': {}}
    submitted = perf_counter()
    def execute():
        trace['timings_ms']['api_worker_queue_wait_ms'] = (perf_counter() - submitted) * 1000
        return retrieve(request.query, request.source_id, request.config_snapshot or load_config(),
                        request.credentials, request.index_collection, diagnostics=trace)
    try:
        return await anyio.to_thread.run_sync(execute)
    except Exception as exc:
        error = bad_request(exc)
        return JSONResponse(status_code=error.status_code, content={'detail': error.detail, 'diagnostics': trace})


@router.post('/query')
def query(request: RunRequest):
    config = snapshot(request)
    try:
        return run_query(request.query, request.source_id, config, request.credentials, save=request.persist_history, expected_collection=request.index_collection)
    except Exception as exc:
        raise bad_request(exc)


@router.post('/query/stream')
def stream_query(request: RunRequest):
    config = snapshot(request)
    try:
        require_index(config, request.source_id)
    except Exception as exc:
        raise bad_request(exc)

    def stream():
        for message in run_events(request.query, request.source_id, config, request.credentials, save=request.persist_history, expected_collection=request.index_collection):
            yield json.dumps(message, allow_nan=False) + '\n'
    return StreamingResponse(stream(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store'})


@router.post('/evaluate')
def evaluate(request: EvaluationRequest):
    config = snapshot(request)
    try:
        from evaluate_ragas import run_ablation
        return submit('evaluation', lambda progress: {'run_ids': run_ablation(config, request.source_id, request.credentials, request.modes, progress)})
    except Exception as exc:
        raise bad_request(exc)


@router.post('/evaluations/{identifier}/resume')
def resume_saved_evaluation(identifier: str, credentials: Credentials):
    try:
        from evaluate_ragas import resume_inputs, resume_evaluation
        resume_inputs(identifier)
        return submit('evaluation', lambda progress: resume_evaluation(identifier, credentials, progress))
    except Exception as exc:
        raise bad_request(exc)


@router.post('/load-tests')
def load_tests(request: LoadTestRequest):
    config = snapshot(request)
    try:
        return submit('load_test', lambda progress: run_load_test(config, request.source_id, request.queries, request.credentials, progress))
    except Exception as exc:
        raise bad_request(exc)


@router.get('/jobs/{identifier}')
def job(identifier: str):
    if len(identifier) != 32 or any(char not in '0123456789abcdef' for char in identifier):
        raise HTTPException(422, 'Invalid job ID')
    try:
        return get_job(identifier)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.get('/history')
def history(kind: str = 'evaluation'):
    if kind not in {'live_run', 'evaluation', 'load_test'}:
        raise HTTPException(422, 'Unsupported history kind')
    records, errors = [], []
    for path in sorted(folder(kind).glob('*_output.yaml'), reverse=True):
        try:
            record = yaml.safe_load(path.read_text())
            if record.get('kind') == kind:
                records.append(record)
        except (OSError, yaml.YAMLError, AttributeError):
            errors.append(f'{path.name}: unable to read history record')
    return {'records': records, 'errors': errors}


@router.get('/dataset')
def dataset():
    from evaluation.golden_dataset_loader import load_dataset
    try:
        cases, dataset_hash = load_dataset(load_config().evaluation.dataset_path)
        return {'cases': [case.model_dump() for case in cases], 'sha256': dataset_hash}
    except (ValueError, OSError) as exc:
        return {'cases': [], 'message': str(exc)}
