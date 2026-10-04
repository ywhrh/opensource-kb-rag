"""Match immutable paired snapshots by execution inputs, never by display names."""
from copy import deepcopy
from core.config_loader import digest
from core.history import folder
import yaml


def signature(config, kind, source_id, knowledge_base, dataset=None, query=None, queries=None):
    config = deepcopy(config)
    config.pop('experiment', None)
    config.pop('load_testing' if kind != 'load_test' else 'evaluation', None)
    if kind == 'live_run':
        config.pop('evaluation', None)
    else:
        config.get('evaluation', {}).pop('dataset_path', None)
    return digest({'config': config, 'kind': kind, 'source_id': source_id,
                   'index': {k: knowledge_base.get(k) for k in ['collection_name', 'source_revision']},
                   'dataset': (dataset or {}).get('sha256') if kind == 'evaluation' else None,
                   'query': query if kind == 'live_run' else None,
                   'queries': queries if kind == 'load_test' else None})


def record_signature(record):
    return signature(record['config_snapshot'], record['kind'], record['source_id'],
                     record.get('knowledge_base', {}), record.get('dataset'),
                     record.get('query'), record.get('queries'))


def find_runs(wanted, kind):
    hits, warnings = [], []
    for output in sorted(folder(kind).glob('*_output.yaml'), reverse=True):
        input_path = output.with_name(output.name.replace('_output.yaml', '_input.yaml'))
        if not input_path.exists():
            continue
        try:
            saved_input = yaml.safe_load(input_path.read_text())
            saved_output = yaml.safe_load(output.read_text())
            if saved_output['kind'] != kind or saved_input['run_id'] != saved_output['run_id']:
                continue
            if record_signature(saved_input) == wanted == record_signature(saved_output):
                hits.append(saved_output)
        except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError):
            warnings.append(output.name)
    return hits, warnings
