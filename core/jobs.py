"""Bounded local background jobs for indexing, evaluation and load testing."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import BoundedSemaphore
from uuid import uuid4
import yaml
from core.config_loader import ROOT, atomic_yaml
from core.logger import event

POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix='studio-job')
SLOTS = BoundedSemaphore(2)
DIRECTORY = ROOT / 'data' / 'jobs'


def recover_interrupted_jobs():
    for path in DIRECTORY.glob('*.yaml'):
        job = yaml.safe_load(path.read_text())
        if job.get('status') in {'PENDING', 'RUNNING'}:
            job.update(status='INTERRUPTED', message='The API restarted. Resume saved evaluations from tuning history; restart other jobs.')
            atomic_yaml(path, job)
    from core.history import folder
    for path in folder('evaluation').glob('*_output.yaml'):
        record = yaml.safe_load(path.read_text())
        if record.get('kind') == 'evaluation' and record['results']['status'] in {'PENDING', 'RUNNING'}:
            record['results']['status'] = 'INTERRUPTED'
            atomic_yaml(path, record)


def submit(kind, function):
    if not SLOTS.acquire(blocking=False):
        raise ValueError('Two background jobs are already active. Wait for one to finish.')
    identifier = uuid4().hex
    path = DIRECTORY / f'{identifier}.yaml'
    job = {'job_id': identifier, 'kind': kind, 'status': 'PENDING', 'created_at': datetime.now(timezone.utc).isoformat(), 'message': 'Queued'}
    atomic_yaml(path, job)

    def work():
        def progress(message):
            job['message'] = message
            atomic_yaml(path, job)
        try:
            job['status'] = 'RUNNING'
            progress('Started')
            job['result'] = function(progress)
            job['status'] = 'SUCCESS'
            progress('Completed. Open history for detailed results.')
        except Exception as exc:
            job['status'] = 'FAILED'
            job['message'] = str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__}: job failed; check data and local service setup'
            event('job_failed', job_id=identifier, error_type=type(exc).__name__)
        finally:
            atomic_yaml(path, job)
            SLOTS.release()
    try:
        POOL.submit(work)
    except Exception:
        SLOTS.release()
        raise
    return job.copy()


def get_job(identifier):
    path = DIRECTORY / f'{identifier}.yaml'
    if not path.exists():
        raise ValueError('Job not found')
    return yaml.safe_load(path.read_text())
