"""Local process/container resource sampling and per-stage percentile summaries."""
import json
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
import psutil


def percentiles(values):
    values = sorted(values)
    def at(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {'count': len(values), **{f'p{p}_ms': at(p / 100) if values else None for p in (50, 95, 99)}}


def docker_resources():
    result = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{json .}}'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError('Docker resource statistics unavailable')
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError('Docker has no monitored running containers')
    listing = subprocess.run(['docker', 'ps', '--format', '{{json .}}'], capture_output=True, text=True, timeout=5)
    if listing.returncode:
        raise RuntimeError('Docker container identities unavailable')
    images = {row['Names']: row['Image'] for row in (json.loads(line) for line in listing.stdout.splitlines() if line.strip())}
    return [{**{k: row.get(k) for k in ('Name', 'CPUPerc', 'MemUsage', 'MemPerc', 'PIDs', 'BlockIO', 'NetIO')},
             'image': images.get(row['Name'])} for row in rows]


class ResourceSampler:
    def __init__(self, api_pid, load_pid, destination, interval=1.0, docker_interval=5.0, collect_docker=True):
        self.destination = Path(destination)
        self.interval, self.docker_interval = interval, docker_interval
        self.collect_docker = collect_docker
        self.processes = {'api': psutil.Process(api_pid), 'locust': psutil.Process(load_pid),
                          'coordinator': psutil.Process()}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True, name='load-resource-sampler')

    def start(self):
        psutil.cpu_percent()
        for process in self.processes.values():
            process.cpu_percent()
        self.thread.start()

    def sample(self):
        next_docker = 0
        with self.destination.open('w') as output:
            while not self.stop.wait(self.interval):
                row = {'timestamp': time.time(), 'host_cpu_percent': psutil.cpu_percent(),
                       'host_memory_percent': psutil.virtual_memory().percent,
                       'host_swap_used_bytes': psutil.swap_memory().used, 'processes': {}}
                for label, process in self.processes.items():
                    try:
                        memory = process.memory_info()
                        row['processes'][label] = {'cpu_percent': process.cpu_percent(),
                                                  'rss_bytes': memory.rss, 'threads': process.num_threads()}
                    except psutil.Error as exc:
                        row['processes'][label] = {'error_type': type(exc).__name__}
                if self.collect_docker and time.monotonic() >= next_docker:
                    try:
                        row['containers'] = docker_resources()
                    except Exception as exc:
                        row['docker_error_type'] = type(exc).__name__
                    next_docker = time.monotonic() + self.docker_interval
                output.write(json.dumps(row, allow_nan=False) + '\n')
                output.flush()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=20)


def summarize(prefix, measurement_start, measurement_end=None):
    prefix = Path(prefix)
    stages, request_count, errors = {}, 0, {}
    measured_requests = 0
    cancelled_requests = 0
    device_metrics = {}
    batches = {}
    requests_path = prefix.with_name(prefix.name + '_requests.jsonl')
    request_rows, invalid_requests = read_jsonl(requests_path)
    for row in request_rows:
        if row.get('phase') != 'measurement':
            continue
        if row.get('http_status') is None:
            cancelled_requests += 1
            continue
        measured_requests += 1
        if not row.get('success'):
            key = str(row.get('http_status')) + ':' + str(row.get('error_type'))
            errors[key] = errors.get(key, 0) + 1
            continue
        request_count += 1
        batch = row.get('batching') or {}
        if batch.get('batch_id'):
            batches[batch['batch_id']] = batch
        for key, value in row.get('device_metrics', {}).items():
            device_metrics[key] = max(device_metrics.get(key, 0), value)
        for stage, value in row.get('timings_ms', {}).items():
            stages.setdefault(stage, []).append(value)
    resources_path = prefix.with_name(prefix.name + '_resources.jsonl')
    resource_rows, invalid_resources = read_jsonl(resources_path)
    resource_rows = [row for row in resource_rows if measurement_start and row['timestamp'] >= measurement_start
                     and (measurement_end is None or row['timestamp'] <= measurement_end)]
    def gaps(timestamps):
        points = sorted(timestamps)
        if not points:
            return None
        boundaries = [measurement_start, *points]
        if measurement_end is not None:
            boundaries.append(measurement_end)
        return max((right - left for left, right in zip(boundaries, boundaries[1:])), default=0)

    processes = {}
    for label in ('api', 'locust', 'coordinator'):
        values = [r['processes'].get(label, {}) for r in resource_rows]
        processes[label] = {'max_sample_gap_seconds': gaps([r['timestamp'] for r in resource_rows if 'cpu_percent' in r['processes'].get(label, {})]), 'samples': sum('cpu_percent' in v for v in values),
            'peak_cpu_percent': max((v['cpu_percent'] for v in values if 'cpu_percent' in v), default=None),
            'peak_rss_bytes': max((v['rss_bytes'] for v in values if 'rss_bytes' in v), default=None)}
    containers = {}
    for row in resource_rows:
        for container in row.get('containers', []):
            summary = containers.setdefault(container['Name'], {'samples': 0, 'peak_cpu_percent': 0, 'peak_memory_percent': 0, 'image': container.get('image')})
            summary['samples'] += 1
            summary['peak_cpu_percent'] = max(summary['peak_cpu_percent'], float(container['CPUPerc'].strip('%')))
            summary['peak_memory_percent'] = max(summary['peak_memory_percent'], float(container['MemPerc'].strip('%')))
    for name, summary in containers.items():
        timestamps = [r['timestamp'] for r in resource_rows if any(c['Name'] == name for c in r.get('containers', []))]
        summary['max_sample_gap_seconds'] = gaps(timestamps)
        summary['first_sample_delay_seconds'] = min(timestamps) - measurement_start
        summary['last_sample_age_seconds'] = (measurement_end or max(timestamps)) - max(timestamps)
    sizes = [b['batch_size'] for b in batches.values()]
    batch_summary = {'unique_batches': len(batches), 'mean_batch_size': sum(sizes) / len(sizes) if sizes else None,
                     'batch_size_counts': {str(size): sizes.count(size) for size in sorted(set(sizes))},
                     'unique_batch_compute': percentiles([b['compute_ms'] for b in batches.values()]),
                     'scope': 'unique batches containing at least one successful measured request; batches can cross warmup boundaries; queue_wait includes collection and previous-batch execution'}
    return {'batching': batch_summary, 'cancelled_before_response': cancelled_requests, 'device_memory_peaks': device_metrics, 'measured_requests': measured_requests, 'invalid_jsonl_lines': invalid_requests + invalid_resources, 'successful_requests_with_stage_timings': request_count, 'request_failure_reasons': errors,
            'stages': {stage: percentiles(values) for stage, values in stages.items()},
            'resources': {'max_sample_gap_seconds': gaps([r['timestamp'] for r in resource_rows]), 'samples': len(resource_rows),
                'first_sample_delay_seconds': min((r['timestamp'] - measurement_start for r in resource_rows), default=None),
                'sampled_span_seconds': max((r['timestamp'] - measurement_start for r in resource_rows), default=0), 'processes': processes, 'containers': containers,
                'docker_failed_samples': sum('docker_error_type' in r for r in resource_rows),
                'host_peak_cpu_percent': max((r['host_cpu_percent'] for r in resource_rows), default=None),
                'host_peak_memory_percent': max((r['host_memory_percent'] for r in resource_rows), default=None),
                'host_peak_swap_used_bytes': max((r['host_swap_used_bytes'] for r in resource_rows), default=None)},
            'scope': 'successful measured requests; process CPU 100% = one logical core; host CPU is averaged across cores; stages overlap in parallel search and percentiles must not be added',
            'limits': 'sampling shows resource pressure, not proof of causality; client time minus stages is not a measured queue time'}


def environment_metadata(api_info):
    import sys
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for package in ('fastapi', 'uvicorn', 'anyio', 'locust', 'psutil', 'torch', 'transformers', 'sentence-transformers', 'pymilvus', 'rank-bm25', 'numpy', 'PyYAML', 'requests'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {'python_version': sys.version, 'package_versions': packages, 'api': api_info, 'host_platform': platform.platform(), 'logical_cpus': psutil.cpu_count(),
            'physical_cpus': psutil.cpu_count(logical=False), 'memory_bytes': psutil.virtual_memory().total,
            'load_generator_same_host': True, 'docker_cli': shutil.which('docker') is not None,
            'gpu_monitoring': 'MPS allocated/driver memory is captured per retrieval; GPU utilization is not collected'}


def write_report(record, prefix):
    """Write a compact English report without declaring an unproven bottleneck."""
    prefix = Path(prefix)
    results = record['results']
    metrics = results.get('load_test_metrics') or {}
    diagnostics = results.get('diagnostics') or {}
    tuning = record.get('tuning_round') or {}
    lines = ['# Retrieval load diagnostics', '', '## Tuning round — changes in this run', '',
             f"Round: {tuning.get('label', 'Not recorded')}",
             f"Comparison baseline: {tuning.get('baseline_run_id') or 'Not specified'}",
             f"Measurement accounting note: {tuning.get('accounting_note') or 'None'}",
             f"Hypothesis (not a measured conclusion): {tuning.get('hypothesis') or 'Not specified'}", '']
    changes = tuning.get('changes') or []
    if changes:
        lines += ['| Changed parameter / code | Before | After |', '|---|---|---|']
        for change in changes:
            def cell(value):
                return json.dumps(value, ensure_ascii=False, allow_nan=False).replace('|', '&#124;').replace('\n', ' ')
            lines.append(f"| {cell(change['parameter'])} | {cell(change['before'])} | {cell(change['after'])} |")
    else:
        lines += ['No tuning changes recorded for this run.']
    lines += ['', '## Run and measurement scope', '', f"Run: {record['run_id']}",
              f"Status: {results['status']}", '',
              'Preflight warms the exact pipeline before the configured concurrent warmup. Requests completed after the warmup reset count, including shutdown drain. Parallel search stages overlap; percentile values cannot be added. These instrumented runs should be compared against new instrumented baselines, not treated as an optimization of earlier uninstrumented runs.', '',
              '## Client results', '', '| Metric | Value |', '|---|---|']
    for key in ('successful_qps', 'actual_p50_latency_ms', 'actual_p95_latency_ms', 'actual_p99_latency_ms',
                'total_requests', 'failed_requests', 'timeout_requests', 'error_rate', 'measured_elapsed_seconds'):
        lines.append(f'| {key} | {metrics.get(key, "Not measured")} |')
    lines += ['', '## Stage timings — successful measured requests', '',
              '| Stage | Samples | P50 ms | P95 ms | P99 ms |', '|---|---:|---:|---:|---:|']
    for stage, values in diagnostics.get('stages', {}).items():
        formatted = ['Not measured' if values.get(k) is None else f'{values[k]:.3f}' for k in ('p50_ms', 'p95_ms', 'p99_ms')]
        lines.append(f"| {stage} | {values['count']} | {' | '.join(formatted)} |")
    lines += ['', '## Resources', '', '| Process | Samples | Peak CPU % | Peak RSS MiB |', '|---|---:|---:|---:|']
    resources = diagnostics.get('resources', {})
    for label, values in resources.get('processes', {}).items():
        rss = values.get('peak_rss_bytes')
        memory = f'{rss / 1024**2:.1f}' if rss is not None else 'Not measured'
        lines.append(f"| {label} | {values['samples']} | {values.get('peak_cpu_percent')} | {memory} |")
    lines += ['', 'Process CPU 100% means one logical core. Host CPU percentage averages all cores. Locust and the API share the host; resource sampling and instrumentation also add overhead.', '',
              '| Container | Samples | Peak CPU % | Peak memory % |', '|---|---:|---:|---:|']
    for name, values in resources.get('containers', {}).items():
        lines.append(f"| {name} | {values['samples']} | {values['peak_cpu_percent']} | {values['peak_memory_percent']} |")
    lines += ['', '## Coverage and interpretation', '',
              f"Diagnostics complete: {diagnostics.get('complete', False)}",
              f"Successful requests with stage timings: {diagnostics.get('successful_requests_with_stage_timings', 0)} / {metrics.get('successful_requests', 'Not measured')}",
              f"Maximum resource sample gap (seconds): {resources.get('max_sample_gap_seconds', 'Not measured')}",
              f"Docker sampling failures: {resources.get('docker_failed_samples', 'Not measured')}", '',
              'Inspect phase timing growth alongside CPU, memory, queues and the matched configuration. Large embedding time plus API CPU pressure supports an embedding-capacity hypothesis; growing API worker or search-pool wait supports a queue-contention hypothesis. Neither alone establishes causality. Milvus RPC time includes transport and server execution, not pure database compute. MPS memory is captured per request; GPU utilization and HTTP parsing/serialization breakdown are not collected.', '',
              f"Request samples: `{prefix.name}_requests.jsonl`", f"Resource samples: `{prefix.name}_resources.jsonl`",
              f"Diagnostic summary: `{prefix.name}_diagnostics.json`", '']
    lines += ['', '## Dynamic batching', '', json.dumps(diagnostics.get('batching', {}), indent=2), '',
              'Per-request embedding_batch_compute_ms describes a shared batch computation and must not be summed across requests. Queue wait includes time waiting for the worker and collecting a batch; the 10 ms collection window is not an overall queueing bound.', '']
    lines += ['', f"Cancelled before HTTP response (excluded from completed-request QPS/errors): {diagnostics.get('cancelled_before_response', 0)}", '']
    lines += ['', '## Device memory peaks', '', json.dumps(diagnostics.get('device_memory_peaks', {}), indent=2), '']
    lines += ['## Saved artifact coverage', '', json.dumps(results.get('artifacts', {}), indent=2), '',
              'Request query_index maps to the frozen queries/question_dataset in input YAML. Client and server request IDs correlate even for timed-out requests. The server JSONL contains safe error frames without exception messages or credentials. The artifacts JSON is the authoritative final file inventory and checksums.', '']
    if results.get('error'):
        lines += ['Error: ' + results['error'], '']
    prefix.with_name(prefix.name + '_report.md').write_text('\n'.join(lines))


def read_jsonl(path):
    """Recover intact rows after interruption, while recording every damaged line."""
    rows, invalid = [], 0
    if not path.exists():
        return rows, invalid
    with path.open() as source:
        for line in source:
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('JSONL row must be an object')
                rows.append(row)
            except ValueError:
                invalid += 1
    return rows, invalid


def save_source_bundle(prefix, backend_source):
    """Archive the exact fingerprinted sources; exclude environment and secret files."""
    import hashlib
    import zipfile
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(str(prefix) + '_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative, expected in backend_source['source_files_sha256'].items():
            content = (root / relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Backend source changed while archiving; restart and retry explicitly')
            archive.writestr(relative, content)
        archive.write(root / 'retrieval_load_test.py', 'retrieval_load_test.py')


def artifact_manifest(prefix, record):
    """Inventory actual saved files, checksum them, and expose missing analysis evidence."""
    import hashlib
    prefix = Path(prefix)
    required = ('input.yaml', 'output.yaml', 'metrics.json', 'requests.jsonl', 'resources.jsonl',
                'server.jsonl', 'diagnostics.json', 'report.md', 'locust.log', 'source.zip',
                'stats.csv', 'stats_history.csv', 'failures.csv', 'exceptions.csv', 'preflight.json', 'containers.json')
    files = sorted(p for p in prefix.parent.glob(prefix.name + '_*')
                   if p.is_file() and p.name != prefix.name + '_artifacts.json')
    inventory = []
    for path in files:
        checksum = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                checksum.update(block)
        inventory.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': checksum.hexdigest()})
    missing = [suffix for suffix in required if not prefix.with_name(prefix.name + '_' + suffix).exists()]
    empty = [suffix for suffix in ('requests.jsonl', 'resources.jsonl', 'server.jsonl')
             if prefix.with_name(prefix.name + '_' + suffix).exists()
             and prefix.with_name(prefix.name + '_' + suffix).stat().st_size == 0]
    client_rows, bad_client = read_jsonl(prefix.with_name(prefix.name + '_requests.jsonl'))
    server_rows, bad_server = read_jsonl(prefix.with_name(prefix.name + '_server.jsonl'))
    _, bad_resources = read_jsonl(prefix.with_name(prefix.name + '_resources.jsonl'))
    expected = {r['request_id'] for r in client_rows if r.get('success') and r.get('request_id')}
    actual = {r.get('request_id') for r in server_rows}
    unmatched = sorted(expected - actual)
    state_path = prefix.with_name(prefix.name + '_containers.json')
    state_errors = []
    if state_path.exists():
        states = json.loads(state_path.read_text())
        for phase in ('before', 'after'):
            for container in states.get(phase, {}).get('containers', []):
                if container.get('error_type'):
                    state_errors.append({'phase': phase, **container})
        if 'after' not in states:
            state_errors.append({'phase': 'after', 'error_type': 'MissingSnapshot'})
    snapshot = {'container_state_errors': state_errors, 'run_id': record['run_id'], 'checked_at_epoch': time.time(), 'files': inventory,
                'missing_files': missing, 'empty_raw_files': empty,
                'invalid_jsonl_lines': bad_client + bad_server + bad_resources,
                'successful_requests_without_server_trace': unmatched,
                'limitations': 'Client timeouts can have a later server completion; traces arriving after finalization remain in the per-run server file but require refreshed analysis. Abrupt termination may leave an unfinished manifest/output; raw checkpoints remain on disk.'}
    snapshot['complete'] = not (missing or empty or snapshot['invalid_jsonl_lines'] or unmatched or state_errors)
    target = prefix.with_name(prefix.name + '_artifacts.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False))
    temporary.replace(target)
    return {key: snapshot[key] for key in ('complete', 'missing_files', 'empty_raw_files', 'invalid_jsonl_lines', 'successful_requests_without_server_trace', 'container_state_errors')} | {'manifest_file': target.name}


def container_states(containers):
    """Capture lifecycle/resource limits without Docker environment variables or log bodies."""
    states = []
    template = '{"state":{{json .State}},"restart_count":{{.RestartCount}},"memory_limit_bytes":{{.HostConfig.Memory}},"nano_cpus":{{.HostConfig.NanoCpus}},"cpuset_cpus":{{json .HostConfig.CpusetCpus}}}'
    for container in containers:
        try:
            result = subprocess.run(['docker', 'inspect', '--format', template, container['Name']],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode:
                raise RuntimeError('Container state unavailable')
            data = json.loads(result.stdout)
            data['state'] = {key: data['state'].get(key) for key in
                             ('Running', 'Paused', 'Restarting', 'OOMKilled', 'Dead', 'ExitCode', 'StartedAt', 'FinishedAt')}
            states.append({'name': container['Name'], 'image': container.get('image'), **data})
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            states.append({'name': container['Name'], 'error_type': type(exc).__name__})
    return {'timestamp': time.time(), 'containers': states}
