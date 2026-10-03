"""Run a dedicated Locust process and persist genuine input/output measurements."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from core.config_loader import ROOT
from core.history import new_output, persist, folder
from ingestion.vector_store_writer import require_index


@contextmanager
def load_lock():
    path = ROOT / 'data' / 'jobs' / 'load_test.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A load test is already running. Wait for it to finish.')
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def run_load_test(config, source_id, queries, credentials=None, progress=None):
    if not config.experiment.name.strip():
        raise ValueError('Experiment name is required')
    if not queries or any(not query.strip() or len(query) > 16000 for query in queries):
        raise ValueError('Provide nonblank load-test queries of at most 16000 characters')
    manifest = require_index(config, source_id)
    with load_lock():
        record = new_output(config, kind='load_test', source_id=source_id, queries=queries,
                            knowledge_base=manifest, latency_scope='retrieval' if config.load_testing.endpoint == '/retrieve' else 'end_to_end')
        persist(record, initial=True)
        directory = folder('load_test')
        prefix = directory / record['run_id']
        metrics_path = prefix.with_name(prefix.name + '_metrics.json')
        env = os.environ.copy()
        env['RAG_LOAD_INPUT'] = str(prefix.with_name(prefix.name + '_input.yaml'))
        env['RAG_LOAD_METRICS'] = str(metrics_path)
        env['RAG_LOAD_CREDENTIALS'] = json.dumps({key: value.get_secret_value() for key, value in credentials if value}) if credentials else '{}'
        parameters = config.load_testing.parameters
        command = [sys.executable, '-m', 'locust', '-f', str(ROOT / 'locustfile.py'), '--headless',
                   '--host', config.load_testing.target_host, '--users', str(parameters.users),
                   '--spawn-rate', str(parameters.spawn_rate), '--run-time', parameters.run_time,
                   '--stop-timeout', '30', '--csv', str(prefix), '--csv-full-history', '--only-summary']
        record['results']['status'] = 'RUNNING'
        persist(record)
        if progress:
            progress(f'Locust running: {parameters.users} users for {parameters.run_time} on {config.load_testing.endpoint}')
        try:
            with prefix.with_name(prefix.name + '_locust.log').open('w') as log:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3900)
            if not metrics_path.exists():
                raise RuntimeError('Locust did not produce a statistics summary')
            metrics = json.loads(metrics_path.read_text())
            if metrics['total_requests'] == 0:
                raise ValueError('No requests were measured. Check the target API and duration.')
            sla = config.load_testing.sla_targets
            latency_passed = metrics['actual_p99_latency_ms'] is not None and metrics['actual_p99_latency_ms'] <= sla.max_p99_latency_ms
            qps_passed = metrics['actual_qps'] >= sla.target_min_qps
            error_free = metrics['failed_requests'] == 0
            record['results'].update(status='SUCCESS', load_test_metrics=metrics, locust_exit_code=process.returncode,
                                     sla_compliance={'latency_passed': latency_passed, 'qps_passed': qps_passed,
                                                     'overall_status': 'PASSED' if latency_passed and qps_passed and error_free else 'FAILED'})
        except Exception as exc:
            record['results'].update(status='FAILED', error=str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__}: load test failed; inspect its Locust log')
        persist(record)
        return {'run_id': record['run_id'], 'status': record['results']['status']}
