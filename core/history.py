"""Durable input/output pairs with unique IDs and explicit execution states."""
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import re
from uuid import uuid4

from core.config_loader import ROOT, atomic_yaml


def run_id(name):
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_')[:48] or 'experiment'
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + slug + '_' + uuid4().hex[:10]


def new_output(config, name=None, kind='live_run', **fields):
    identifier = run_id(name or config.experiment.name)
    packages = {}
    for package in ['pymilvus', 'sentence-transformers', 'ragas', 'locust']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return {
        'schema_version': 1, 'run_id': identifier, 'kind': kind,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment': {'name': name or config.experiment.name, 'description': config.experiment.description},
        'config_snapshot': config.model_dump(), 'runtime_versions': packages, **fields,
        'results': {
            'status': 'PENDING', 'answer': None, 'evidence_chunks': [],
            'retrieval_latency_ms': None, 'generation_latency_ms': None, 'total_latency_ms': None,
            'algorithm_metrics': {key: None for key in config.evaluation.metrics},
            'load_test_metrics': {'actual_qps': None, 'actual_p99_latency_ms': None, 'total_requests': None, 'failed_requests': None},
            'sla_compliance': {'latency_passed': None, 'qps_passed': None, 'overall_status': None},
        },
    }


def folder(kind):
    return ROOT / 'data' / ('load_test_history' if kind == 'load_test' else 'experiments_history')


def persist(record, initial=False):
    directory = folder(record['kind'])
    identifier = record['run_id']
    if initial:
        input_record = {key: value for key, value in record.items() if key != 'results'}
        atomic_yaml(directory / f'{identifier}_input.yaml', input_record)
    atomic_yaml(directory / f'{identifier}_output.yaml', record)
    return record
