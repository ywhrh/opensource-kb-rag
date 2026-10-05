"""Editable Locust retrieval load test: python retrieval_load_test.py.

Tuning workflow: record the user-requested changes and before/after values in
TUNING_ROUND, implement them, select a comparable baseline, then launch only
after the user requests a run. Each run freezes those notes in YAML and places
them at the beginning of its report. Check evidence coverage before comparing.

Only POST /retrieve is exercised. Run with the project virtual environment.
"""
from pathlib import Path

# --------------------------- EDIT CONFIG HERE ---------------------------
# Update these notes for every user-authorized tuning round before launching a run.
TUNING_ROUND = {
    'label': 'query_embedding_cpu_to_mps',
    'baseline_run_id': '20261004T232636_transformers_v4571_baseline_60_incremental_retri_aa973ea8b3',
    'changes': [{'parameter': 'vector_search.query_device', 'before': 'cpu', 'after': 'mps'}],
    'accounting_note': 'Requests cancelled by Locust before receiving a response are recorded separately and excluded from completed-request QPS/errors. The initial MPS trial was retained as PARTIAL.',
    'hypothesis': 'Moving query embedding to Apple GPU may improve throughput and reduce CPU pressure; measured outcome pending.',
}

API_HOST = 'http://127.0.0.1:8000'
ENDPOINT = '/retrieve'
# Set to a saved *_input.yaml to test that exact experiment; None uses config.yaml.
EXPERIMENT_INPUT_PATH = 'data/experiments_history/20261004T013927_transformers_v4571_baseline_60_incremental_vecto_b2974b990a_input.yaml'
SOURCE_ID = 'project'                 # Used when EXPERIMENT_INPUT_PATH is None.
QUESTIONS_PATH = 'data/golden_test_dataset.json'
USERS = 10
QUERY_EMBEDDING_DEVICE = 'mps'
SPAWN_RATE = 2.0                     # New users per second, not request QPS.
WAIT_TIME_SECONDS = 0.1              # Delay after a response, before the next request.
WARMUP_SECONDS = 15                  # Includes user ramp-up; completed warmup requests excluded.
MEASUREMENT_SECONDS = 60
RAMP_UP_TIMEOUT_SECONDS = 60         # Abort if the actual user count never reaches USERS.
MAX_RESOURCE_GAP_SECONDS = 5.0
MAX_DOCKER_GAP_SECONDS = 15.0
REQUEST_TIMEOUT_SECONDS = 30
STOP_TIMEOUT_SECONDS = 35            # Allow in-flight requests to finish after measurement.
QUERY_RANDOM_SEED = 42
TARGET_SUCCESS_QPS = 40.0
MAX_P99_LATENCY_MS = 1000.0          # Editable acceptance target, not a measured result.
RESOURCE_SAMPLE_SECONDS = 1.0
DOCKER_SAMPLE_SECONDS = 5.0
REQUIRE_DOCKER_METRICS = True
METRICS_CHECKPOINT_SECONDS = 5.0
MAX_ERROR_RATE = 0.01               # Fraction: 0.01 means 1%.
# API keys are loaded from .secrets/api_keys.json, never placed here or in snapshots.
# ------------------------------------------------------------------------

import json
import math
from itertools import count
from uuid import uuid4
import os
import random
import subprocess
import sys
import time
import psutil

import yaml
from locust import HttpUser, LoadTestShape, constant, events, task
from requests import get as http_get
from requests.exceptions import Timeout

ROOT = Path(__file__).resolve().parent
INPUT = None
TIMEOUTS = 0
MEASUREMENT_STARTED = None
MEASUREMENT_FINISHED = False
MEASUREMENT_START_RUN_TIME = None
MEASUREMENT_START_USERS = None
MEASUREMENT_ABORT_REASON = None
USER_SEQUENCE = count()
LOCUST_ENVIRONMENT = None
LAST_CHECKPOINT = 0.0
REQUEST_LOG = None
if os.getenv('RAG_LOAD_INPUT'):
    INPUT = yaml.load(Path(os.environ['RAG_LOAD_INPUT']).read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    REQUEST_LOG = Path(os.environ['RAG_LOAD_REQUESTS']).open('w')
CREDENTIALS = json.loads(os.getenv('RAG_LOAD_CREDENTIALS', '{}'))


def required_stages(mode):
    stages = {'index_validation_ms', 'embedding_ms', 'vector_db_search_ms',
              'api_worker_queue_wait_ms', 'server_retrieve_ms'}
    if mode != 'vector_only':
        stages.update({'bm25_task_ms', 'vector_queue_wait_ms', 'bm25_queue_wait_ms', 'fusion_ms'})
    if mode == 'hybrid_rerank':
        stages.add('rerank_ms')
    return stages


class RetrievalUser(HttpUser):
    wait_time = constant(WAIT_TIME_SECONDS)

    def on_start(self):
        self.queries = random.Random(QUERY_RANDOM_SEED + next(USER_SEQUENCE))

    @task
    def retrieve(self):
        query_index = self.queries.randrange(len(INPUT['queries']))
        request_id = uuid4().hex
        start_epoch = time.time()
        payload = {'query': INPUT['queries'][query_index],
                   'source_id': INPUT['source_id'], 'config_snapshot': INPUT['config_snapshot'],
                   'index_collection': INPUT['knowledge_base']['collection_name'],
                   'credentials': CREDENTIALS, 'persist_history': False}
        started = time.perf_counter()
        trace, success, code, error_type = {}, False, None, None
        response_bytes = None
        evidence_ids = []
        try:
            with self.client.post(ENDPOINT, json=payload, name=ENDPOINT,
                                  headers={'X-Load-Test-Id': INPUT['run_id'], 'X-Request-Id': request_id},
                                  catch_response=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response_bytes = len(response.content or b'')
                code = response.status_code
                if code == 0:
                    error_type = type(response.error).__name__
                    return
                try:
                    body = response.json()
                    if not isinstance(body, dict):
                        raise ValueError('Response must be an object')
                    candidate = body.get('diagnostics')
                    if not isinstance(candidate, dict):
                        raise ValueError('Diagnostics must be an object')
                    trace = candidate
                    if not isinstance(trace.get('timings_ms'), dict):
                        trace = {}
                        raise ValueError('Timings must be an object')
                    timings = trace['timings_ms']
                    numeric_timings = all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in timings.values())
                    if not numeric_timings:
                        trace['timings_ms'] = {}
                        raise ValueError('Invalid timing values')
                    if code != 200:
                        error_type = trace.get('error_type') if isinstance(trace.get('error_type'), str) else 'HTTPFailure'
                        response.failure(f'HTTP {code}')
                        return
                    timings = trace.get('timings_ms') or {}
                    mode = INPUT['config_snapshot']['retrieval_pipeline']['mode']
                    required = required_stages(mode)
                    valid = trace.get('version') == 1 and required <= timings.keys()
                    valid = valid and numeric_timings
                    valid = valid and trace.get('embedding_device') == INPUT['config_snapshot']['vector_search']['query_device']
                    valid = valid and isinstance(body.get('evidence_chunks'), list)
                    if not valid:
                        error_type = 'MissingDiagnostics'
                        response.failure('Invalid retrieval response or missing stage diagnostics')
                    else:
                        success = True
                        evidence_ids = [{k: v for k, v in chunk.items() if k in ('chunk_id', 'doc_id', 'score_type') and isinstance(v, str)}
                                        for chunk in body['evidence_chunks'] if isinstance(chunk, dict)]
                except (ValueError, AttributeError, TypeError):
                    error_type = 'InvalidJSON'
                    response.failure('Invalid JSON response')
        finally:
            row = {'timestamp': time.time(), 'phase': 'measurement' if MEASUREMENT_STARTED else 'warmup',
                   'load_test_id': INPUT['run_id'], 'request_id': request_id,
                   'server_request_id': trace.get('request_id') if isinstance(trace.get('request_id'), str) else None,
                   'query_index': query_index, 'started_at_epoch': start_epoch,
                   'started_phase': 'measurement' if MEASUREMENT_STARTED and start_epoch >= MEASUREMENT_STARTED else 'warmup',
                   'active_users': self.environment.runner.user_count,
                   'success': success, 'http_status': code, 'error_type': error_type,
                   'completion_state': 'completed' if code is not None else 'cancelled_before_response',
                   'response_bytes': response_bytes, 'evidence_ids': evidence_ids,
                   'client_elapsed_ms': (time.perf_counter() - started) * 1000,
                   'timings_ms': trace.get('timings_ms', {}),
                   'embedding_device': trace.get('embedding_device'),
                   'device_metrics': trace.get('device_metrics', {}),
                   'candidate_count_before_rerank': trace.get('candidate_count_before_rerank') if type(trace.get('candidate_count_before_rerank')) is int else None,
                   'evidence_count': trace.get('evidence_count') if type(trace.get('evidence_count')) is int else None}
            REQUEST_LOG.write(json.dumps(row, allow_nan=False) + '\n')
            REQUEST_LOG.flush()


class RetrievalLoadShape(LoadTestShape):
    def tick(self):
        global TIMEOUTS, MEASUREMENT_STARTED, MEASUREMENT_FINISHED
        global MEASUREMENT_START_RUN_TIME, MEASUREMENT_START_USERS, MEASUREMENT_ABORT_REASON
        elapsed = self.get_run_time()
        actual_users = self.get_current_user_count()
        if MEASUREMENT_STARTED is None:
            if elapsed >= WARMUP_SECONDS and actual_users == USERS:
                self.runner.stats.reset_all()
                TIMEOUTS = 0
                MEASUREMENT_STARTED = time.time()
                MEASUREMENT_START_RUN_TIME = elapsed
                MEASUREMENT_START_USERS = actual_users
                save_metrics(self.runner.environment, final=False)
            elif elapsed >= WARMUP_SECONDS + RAMP_UP_TIMEOUT_SECONDS:
                MEASUREMENT_ABORT_REASON = 'Target user count was not reached before ramp-up timeout'
                return None
        elif actual_users != USERS:
            MEASUREMENT_ABORT_REASON = 'User count changed during measurement'
            return None
        if MEASUREMENT_START_RUN_TIME is not None and elapsed - MEASUREMENT_START_RUN_TIME >= MEASUREMENT_SECONDS:
            MEASUREMENT_FINISHED = True
            return None
        return USERS, SPAWN_RATE


@events.init.add_listener
def capture_environment(environment, **kwargs):
    global LOCUST_ENVIRONMENT
    LOCUST_ENVIRONMENT = environment
    save_metrics(environment, final=False)


@events.request.add_listener
def count_timeout(exception=None, **kwargs):
    global TIMEOUTS, LAST_CHECKPOINT
    if isinstance(exception, (Timeout, TimeoutError)):
        TIMEOUTS += 1
    if LOCUST_ENVIRONMENT is not None and time.monotonic() - LAST_CHECKPOINT >= METRICS_CHECKPOINT_SECONDS:
        save_metrics(LOCUST_ENVIRONMENT, final=False)
        LAST_CHECKPOINT = time.monotonic()


@events.quitting.add_listener
def save_metrics(environment, final=True, **kwargs):
    if not os.getenv('RAG_LOAD_METRICS'):
        return
    stats = environment.stats.total
    # Includes measured requests completed during graceful shutdown; state this scope explicitly.
    seconds = max(0.0, time.time() - MEASUREMENT_STARTED) if MEASUREMENT_STARTED else 0.0
    success = stats.num_requests - stats.num_failures
    measured = MEASUREMENT_STARTED is not None and stats.num_requests > 0
    metrics = {
        'checkpoint_epoch': time.time(), 'final_snapshot': final,
        'measurement_started': MEASUREMENT_STARTED is not None,
        'measurement_start_epoch': MEASUREMENT_STARTED,
        'measurement_finished': MEASUREMENT_FINISHED,
        'measurement_start_users': MEASUREMENT_START_USERS,
        'actual_warmup_seconds': MEASUREMENT_START_RUN_TIME,
        'measurement_abort_reason': MEASUREMENT_ABORT_REASON,
        'measured_elapsed_seconds': seconds,
        'timing_scope': 'requests completed after warmup reset, including shutdown drain',
        'total_requests': stats.num_requests if measured else 0,
        'successful_requests': success if measured else 0,
        'failed_requests': stats.num_failures if measured else 0,
        'timeout_requests': TIMEOUTS if measured else 0,
        'actual_qps': stats.num_requests / seconds if measured and seconds else 0.0,
        'successful_qps': success / seconds if measured and seconds else 0.0,
        'error_rate': stats.num_failures / stats.num_requests if measured else None,
        **{f'actual_p{percentile}_latency_ms': stats.get_response_time_percentile(percentile / 100) if measured else None
           for percentile in (50, 95, 99)},
        'latency_scope': 'client-observed retrieval HTTP response time, including failed requests',
    }
    if final and REQUEST_LOG is not None:
        REQUEST_LOG.close()
    destination = Path(os.environ['RAG_LOAD_METRICS'])
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    temporary.replace(destination)


def main():
    from core.config_loader import RAGConfig, load_config
    from core.load_diagnostics import docker_resources
    from core.load_testing import load_lock
    from core.secrets import load_keys
    from ingestion.vector_store_writer import require_index

    if not isinstance(TUNING_ROUND.get('label'), str) or not TUNING_ROUND['label'].strip():
        raise ValueError('Provide a tuning round label')
    if not isinstance(TUNING_ROUND.get('changes'), list):
        raise ValueError('Tuning changes must be a list')
    for change in TUNING_ROUND['changes']:
        if not isinstance(change, dict) or not {'parameter', 'before', 'after'} <= change.keys():
            raise ValueError('Each tuning change must record parameter, before and after')
    json.dumps(TUNING_ROUND, allow_nan=False)
    if METRICS_CHECKPOINT_SECONDS <= 0 or RAMP_UP_TIMEOUT_SECONDS <= 0 or MAX_RESOURCE_GAP_SECONDS < RESOURCE_SAMPLE_SECONDS or MAX_DOCKER_GAP_SECONDS < DOCKER_SAMPLE_SECONDS:
        raise ValueError('Invalid ramp-up timeout or sampling coverage limits')
    if USERS < 1 or SPAWN_RATE <= 0 or WAIT_TIME_SECONDS < 0 or WARMUP_SECONDS < 0 or MEASUREMENT_SECONDS <= 0 or REQUEST_TIMEOUT_SECONDS <= 0 or STOP_TIMEOUT_SECONDS < 0:
        raise ValueError('Invalid load-test parameters')
    if WARMUP_SECONDS < USERS / SPAWN_RATE:
        raise ValueError('Warmup must include enough time to spawn all users')
    if STOP_TIMEOUT_SECONDS < REQUEST_TIMEOUT_SECONDS:
        raise ValueError('Stop timeout must cover request timeout to let in-flight requests finish')
    if not 0 <= MAX_ERROR_RATE <= 1 or TARGET_SUCCESS_QPS <= 0 or MAX_P99_LATENCY_MS <= 0:
        raise ValueError('Invalid acceptance targets')
    if ENDPOINT != '/retrieve':
        raise ValueError('This script tests retrieval only: ENDPOINT must be /retrieve')
    try:
        health = http_get(API_HOST.rstrip('/') + '/health', timeout=10)
        health.raise_for_status()
    except Exception as exc:
        raise RuntimeError('FastAPI is unavailable. Start: .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000') from exc
    if not health.json().get('services', {}).get('milvus', {}).get('ready'):
        raise RuntimeError('Milvus is not ready; start the project Docker services before testing')
    if RESOURCE_SAMPLE_SECONDS <= 0 or DOCKER_SAMPLE_SECONDS <= 0:
        raise ValueError('Resource sample intervals must be positive')
    try:
        api_info = http_get(API_HOST.rstrip('/') + '/diagnostics', timeout=5).json()
        if api_info.get('diagnostics_version') != 1:
            raise ValueError('Backend does not expose stage diagnostics')
        from core.profiling import source_metadata
        if (api_info.get('backend_source') or {}).get('source_sha256') != source_metadata()['source_sha256']:
            raise ValueError('Running backend source differs from disk; restart FastAPI')
        api_process = psutil.Process(api_info['api_pid'])
        api_process.memory_info()
    except Exception as exc:
        raise RuntimeError('Stage/resource diagnostics unavailable. Restart the updated FastAPI backend before testing.') from exc
    initial_containers = []
    if REQUIRE_DOCKER_METRICS:
        initial_containers = docker_resources()
        if not any('milvus' in (row.get('image') or '').lower() for row in initial_containers):
            raise RuntimeError('No Milvus container resource statistics found')
    original = None
    if EXPERIMENT_INPUT_PATH:
        path = ROOT / EXPERIMENT_INPUT_PATH
        original = yaml.load(path.read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
        config = RAGConfig.model_validate(original['config_snapshot'])
        source = original['source_id']
    else:
        config, source = load_config(), SOURCE_ID
    config.vector_search.query_device = QUERY_EMBEDDING_DEVICE
    manifest = require_index(config, source)
    if original and any(original['knowledge_base'].get(k) != manifest.get(k) for k in ('collection_name', 'source_revision')):
        raise ValueError('The saved experiment index changed; select a current experiment')
    questions = json.loads((ROOT / QUESTIONS_PATH).read_text())
    queries = [row['question'] for row in questions]
    if not queries or any(not isinstance(q, str) or not q.strip() or len(q) > 16000 for q in queries):
        raise ValueError('Questions must be nonblank strings of at most 16000 characters')
    config.load_testing.endpoint = ENDPOINT
    config.load_testing.target_host = API_HOST
    config.load_testing.parameters.users = USERS
    config.load_testing.parameters.spawn_rate = SPAWN_RATE
    config.load_testing.parameters.run_time = f'{MEASUREMENT_SECONDS}s'
    config.load_testing.sla_targets.target_min_qps = TARGET_SUCCESS_QPS
    config.load_testing.sla_targets.max_p99_latency_ms = MAX_P99_LATENCY_MS
    config = RAGConfig.model_validate(config.model_dump())
    keys = load_keys()
    credentials = {provider + '_api_key': key for provider, key in keys.items()}
    with load_lock():
        return run_locked(config, source, manifest, queries, credentials, keys, original, api_info, initial_containers, questions)


def run_locked(config, source, manifest, queries, credentials, keys, original, api_info, initial_containers, questions):
    from core.config_loader import digest
    from core.load_diagnostics import ResourceSampler, environment_metadata, summarize
    from core.history import new_output, persist, folder
    parameters = {'users': USERS, 'spawn_rate': SPAWN_RATE, 'wait_time_seconds': WAIT_TIME_SECONDS,
                  'warmup_seconds': WARMUP_SECONDS, 'measurement_seconds': MEASUREMENT_SECONDS,
                  'request_timeout_seconds': REQUEST_TIMEOUT_SECONDS, 'stop_timeout_seconds': STOP_TIMEOUT_SECONDS,
                  'random_seed': QUERY_RANDOM_SEED, 'target_success_qps': TARGET_SUCCESS_QPS,
                  'max_p99_latency_ms': MAX_P99_LATENCY_MS, 'max_error_rate': MAX_ERROR_RATE,
                  'resource_sample_seconds': RESOURCE_SAMPLE_SECONDS, 'docker_sample_seconds': DOCKER_SAMPLE_SECONDS,
                  'require_docker_metrics': REQUIRE_DOCKER_METRICS,
                  'metrics_checkpoint_seconds': METRICS_CHECKPOINT_SECONDS,
                  'ramp_up_timeout_seconds': RAMP_UP_TIMEOUT_SECONDS,
                  'max_resource_gap_seconds': MAX_RESOURCE_GAP_SECONDS,
                  'max_docker_gap_seconds': MAX_DOCKER_GAP_SECONDS}
    record = new_output(config, name=config.experiment.name + '_retrieval_load', kind='load_test',
                        source_id=source, queries=queries, question_dataset=questions, knowledge_base=manifest,
                        original_experiment_input=original,
                        load_test_parameters=parameters, original_experiment_id=original['run_id'] if original else None,
                        latency_scope='retrieval', diagnostics_version=1,
                        environment=environment_metadata(api_info), initial_containers=initial_containers,
                        workload_sha256=digest(queries),
                        locust_script_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
    # Put the frozen round description first in both YAML files.
    record = {'tuning_round': json.loads(json.dumps(TUNING_ROUND, allow_nan=False)), **record}
    record['results']['status'] = 'RUNNING'
    persist(record, initial=True)
    prefix = folder('load_test') / record['run_id']
    env = os.environ.copy()
    env['RAG_LOAD_INPUT'] = str(prefix.with_name(prefix.name + '_input.yaml'))
    metrics_path = prefix.with_name(prefix.name + '_metrics.json')
    env['RAG_LOAD_METRICS'] = str(metrics_path)
    env['RAG_LOAD_REQUESTS'] = str(prefix.with_name(prefix.name + '_requests.jsonl'))
    env['RAG_LOAD_CREDENTIALS'] = json.dumps({provider + '_api_key': key for provider, key in keys.items()})
    command = [sys.executable, '-m', 'locust', '-f', str(Path(__file__).resolve()), '--headless',
               '--host', API_HOST, '--stop-timeout', str(STOP_TIMEOUT_SECONDS), '--only-summary',
               '--csv', str(prefix), '--csv-full-history']
    print(f'Retrieval load test: {USERS} users, {MEASUREMENT_SECONDS}s measurement, {WARMUP_SECONDS}s warmup')
    print(f'Results: {prefix}_output.yaml')
    sampler = None
    process = None
    from core.load_diagnostics import save_source_bundle, container_states
    try:
        save_source_bundle(prefix, api_info['backend_source'])
        container_snapshot = {'before': container_states(initial_containers)}
        container_path = prefix.with_name(prefix.name + '_containers.json')
        container_path.write_text(json.dumps(container_snapshot, indent=2))
        # Check the exact frozen pipeline and warm lazy models before starting the timed warmup.
        from requests import post as http_post
        preflight = {'started_at_epoch': time.time(), 'query_index': 0, 'status': 'RUNNING'}
        preflight_path = prefix.with_name(prefix.name + '_preflight.json')
        preflight_path.write_text(json.dumps(preflight))
        preflight_started = time.perf_counter()
        probe = http_post(API_HOST.rstrip('/') + ENDPOINT,
            json={'query': queries[0], 'source_id': source, 'config_snapshot': config.model_dump(),
                  'index_collection': manifest['collection_name'], 'credentials': credentials,
                  'persist_history': False}, headers={'X-Load-Test-Id': record['run_id'], 'X-Request-Id': uuid4().hex}, timeout=REQUEST_TIMEOUT_SECONDS)
        probe.raise_for_status()
        probe_body = probe.json()
        probe_trace = probe_body.get('diagnostics') if isinstance(probe_body, dict) else None
        probe_timings = probe_trace.get('timings_ms') if isinstance(probe_trace, dict) else None
        if (not isinstance(probe_timings, dict) or probe_trace.get('version') != 1
                or not required_stages(config.retrieval_pipeline.mode) <= probe_timings.keys()
                or not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in probe_timings.values())
                or not isinstance(probe_body.get('evidence_chunks'), list)
                or probe_trace.get('embedding_device') != QUERY_EMBEDDING_DEVICE):
            raise RuntimeError('Exact-pipeline preflight did not return all required stage diagnostics')
        preflight.update(status='SUCCESS', finished_at_epoch=time.time(), client_elapsed_ms=(time.perf_counter() - preflight_started) * 1000, diagnostics=probe_trace)
        preflight_path.write_text(json.dumps(preflight, indent=2, allow_nan=False))
        with prefix.with_name(prefix.name + '_locust.log').open('w') as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
                sampler = ResourceSampler(api_info['api_pid'], process.pid,
                    prefix.with_name(prefix.name + '_resources.jsonl'), RESOURCE_SAMPLE_SECONDS, DOCKER_SAMPLE_SECONDS,
                    collect_docker=REQUIRE_DOCKER_METRICS or bool(initial_containers))
                sampler.start()
                try:
                    process.wait(timeout=WARMUP_SECONDS + RAMP_UP_TIMEOUT_SECONDS + MEASUREMENT_SECONDS + STOP_TIMEOUT_SECONDS + 60)
                except (KeyboardInterrupt, subprocess.TimeoutExpired):
                    import signal
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=STOP_TIMEOUT_SECONDS + 5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    sampler.finish()
        if not metrics_path.exists():
            raise RuntimeError('No metrics produced; inspect the Locust log')
        metrics = json.loads(metrics_path.read_text())
        if not metrics['measurement_started'] or not metrics['total_requests']:
            raise RuntimeError('No measured requests completed')
        diagnostic_summary = summarize(prefix, metrics['measurement_start_epoch'], metrics['measurement_start_epoch'] + metrics['measured_elapsed_seconds'])
        stage_count = diagnostic_summary['successful_requests_with_stage_timings']
        resources = diagnostic_summary['resources']
        complete = (stage_count == metrics['successful_requests'] and stage_count > 0
            and diagnostic_summary['invalid_jsonl_lines'] == 0
            and diagnostic_summary['measured_requests'] == metrics['total_requests']
            and metrics.get('final_snapshot') is True
            and resources['processes']['api']['samples'] > 0
            and resources['processes']['locust']['samples'] > 0
            and (not REQUIRE_DOCKER_METRICS or any('milvus' in (v.get('image') or '').lower() for v in resources['containers'].values()))
            and (not REQUIRE_DOCKER_METRICS or resources['docker_failed_samples'] == 0)
            and resources['first_sample_delay_seconds'] is not None
            and resources['first_sample_delay_seconds'] <= MAX_RESOURCE_GAP_SECONDS
            and resources['max_sample_gap_seconds'] <= MAX_RESOURCE_GAP_SECONDS
            and all(v['max_sample_gap_seconds'] is not None and v['max_sample_gap_seconds'] <= MAX_RESOURCE_GAP_SECONDS and v['samples'] >= max(2, int(metrics['measured_elapsed_seconds'] / MAX_RESOURCE_GAP_SECONDS)) for v in resources['processes'].values())
            and (not REQUIRE_DOCKER_METRICS or all(v['max_sample_gap_seconds'] <= MAX_DOCKER_GAP_SECONDS and v['first_sample_delay_seconds'] <= MAX_DOCKER_GAP_SECONDS and v['last_sample_age_seconds'] <= MAX_DOCKER_GAP_SECONDS and v['samples'] >= max(2, int(metrics['measured_elapsed_seconds'] / MAX_DOCKER_GAP_SECONDS)) for v in resources['containers'].values() if 'milvus' in (v.get('image') or '').lower()))
            and resources['sampled_span_seconds'] >= metrics['measured_elapsed_seconds'] - max(5, 2 * RESOURCE_SAMPLE_SECONDS))
        diagnostic_summary['complete'] = complete
        destination = prefix.with_name(prefix.name + '_diagnostics.json')
        destination.write_text(json.dumps(diagnostic_summary, indent=2, allow_nan=False))
        record['results']['diagnostics'] = diagnostic_summary
        latency_passed = metrics['actual_p99_latency_ms'] <= MAX_P99_LATENCY_MS
        qps_passed = metrics['successful_qps'] >= TARGET_SUCCESS_QPS
        errors_passed = metrics['error_rate'] <= MAX_ERROR_RATE
        record['results'].update(status=('SUCCESS' if complete else 'PARTIAL') if metrics['measurement_finished'] else 'INTERRUPTED', load_test_metrics=metrics, locust_exit_code=process.returncode,
            sla_compliance={'latency_passed': latency_passed, 'qps_passed': qps_passed,
                            'error_rate_passed': errors_passed,
                            'overall_status': 'PASSED' if latency_passed and qps_passed and errors_passed else 'FAILED'})
        if not metrics['measurement_finished']:
            record['results']['sla_compliance'] = {'overall_status': 'INCOMPLETE'}
        if not complete:
            record['results']['diagnostics_warning'] = 'Diagnostic coverage is incomplete; inspect request/resource JSONL before drawing bottleneck conclusions.'
        print(json.dumps(metrics, indent=2))
    except KeyboardInterrupt:
        record['results'].update(status='CANCELLED', error='Stopped by user')
    except Exception as exc:
        record['results'].update(status='FAILED', error=f'{type(exc).__name__}: inspect the Locust log')
        print(record['results']['error'], file=sys.stderr)
    finally:
        if 'diagnostics' not in record['results'] and metrics_path.exists():
            try:
                partial_metrics = json.loads(metrics_path.read_text())
                record['results']['load_test_metrics'] = partial_metrics
                partial = summarize(prefix, partial_metrics.get('measurement_start_epoch'), partial_metrics.get('checkpoint_epoch'))
                partial['complete'] = False
                record['results']['diagnostics'] = partial
                prefix.with_name(prefix.name + '_diagnostics.json').write_text(json.dumps(partial, indent=2))
            except (OSError, ValueError, KeyError) as exc:
                record['results']['diagnostics_warning'] = type(exc).__name__ + ': partial diagnostics unavailable'
        from core.load_diagnostics import write_report, artifact_manifest
        if 'container_snapshot' in locals():
            try:
                container_snapshot['after'] = container_states(initial_containers)
                container_path.write_text(json.dumps(container_snapshot, indent=2))
            except OSError as exc:
                record['results']['container_state_warning'] = type(exc).__name__
        if 'preflight' in locals() and preflight['status'] == 'RUNNING':
            preflight.update(status='FAILED', finished_at_epoch=time.time(), error=record['results'].get('error'))
            try:
                preflight_path.write_text(json.dumps(preflight, indent=2, allow_nan=False))
            except OSError as exc:
                record['results']['preflight_save_warning'] = type(exc).__name__
        record['results']['finished_at_epoch'] = time.time()
        if process is not None:
            record['results']['locust_exit_code'] = process.returncode
        try:
            write_report(record, prefix)
        except OSError as exc:
            record['results']['report_warning'] = type(exc).__name__ + ': report could not be saved'
        finally:
            persist(record)
            # The async server logger can trail the final HTTP completion briefly.
            time.sleep(0.25)
            artifacts = artifact_manifest(prefix, record)
            record['results']['artifacts'] = artifacts
            if not artifacts['complete'] and record['results']['status'] == 'SUCCESS':
                record['results']['status'] = 'PARTIAL'
            persist(record)
            try:
                write_report(record, prefix)
            except OSError:
                pass
            # Refresh checksums after the final output/report rewrite.
            artifact_manifest(prefix, record)
    return 0 if record['results']['status'] == 'SUCCESS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
