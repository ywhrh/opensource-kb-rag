"""Match immutable paired snapshots by execution inputs, never by display names."""
from copy import deepcopy
from functools import lru_cache
from core.config_loader import digest
from core.history import folder
import yaml


def signature(config, kind, source_id, knowledge_base, dataset=None, query=None, queries=None):
    config = deepcopy(config)
    config.pop('experiment', None)
    # Historical snapshots omitted this field; omitted and explicit CPU are equivalent.
    if config.get('vector_search', {}).get('query_device', 'cpu') == 'cpu':
        config.get('vector_search', {}).pop('query_device', None)
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


@lru_cache(maxsize=128)
def _read_pair(input_path, input_stamp, output_path, output_stamp):
    """Cache parsed pairs by file identity; atomic checkpoints invalidate the cache."""
    loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
    saved_input = yaml.load(input_path.read_text(), Loader=loader)
    saved_output = yaml.load(output_path.read_text(), Loader=loader)
    if saved_input['run_id'] != saved_output['run_id']:
        return None
    wanted = record_signature(saved_input)
    if wanted != record_signature(saved_output):
        return None
    return saved_input, saved_output, wanted


def saved_pairs(kind):
    pairs, warnings = [], []
    for output in sorted(folder(kind).glob('*_output.yaml'), reverse=True):
        input_path = output.with_name(output.name.replace('_output.yaml', '_input.yaml'))
        try:
            if not input_path.exists():
                continue
            def stamp(path):
                stat = path.stat()
                return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_size
            pair = _read_pair(input_path, stamp(input_path), output, stamp(output))
            if pair and pair[0]['kind'] == kind and pair[1]['kind'] == kind:
                pairs.append(deepcopy(pair))
        except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError):
            warnings.append(output.name)
    return pairs, warnings


def find_runs(wanted, kind):
    pairs, warnings = saved_pairs(kind)
    return [output for _, output, recorded in pairs if recorded == wanted], warnings
