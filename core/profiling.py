"""Request-stage timings and asynchronous metadata-only retrieval logs."""
import atexit
from contextlib import contextmanager
from functools import lru_cache
import json
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import SimpleQueue
from time import perf_counter
from core.config_loader import ROOT

DIAGNOSTICS_VERSION = 1


@contextmanager
def timed(timings, stage):
    started = perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            timings[stage] = (perf_counter() - started) * 1000


@lru_cache(maxsize=1)
def trace_logger():
    directory = ROOT / 'data/logs'
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(directory / 'retrieval.jsonl', maxBytes=10_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter('%(message)s'))
    queue = SimpleQueue()
    listener = QueueListener(queue, handler, RunTraceHandler())
    listener.start()
    atexit.register(listener.stop)
    logger = logging.getLogger('retrieval_diagnostics')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = [QueueHandler(queue)]
    return logger


class RunTraceHandler(logging.Handler):
    """Keep load-test traces beside the run so shared-log rotation cannot erase them."""
    def emit(self, record):
        import re
        try:
            trace = json.loads(record.getMessage())
            identifier = trace.get('load_test_id') or ''
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', identifier):
                return
            prefix = ROOT / 'data/load_test_history' / identifier
            if not prefix.with_name(identifier + '_input.yaml').is_file():
                return
            with prefix.with_name(identifier + '_server.jsonl').open('a') as output:
                output.write(record.getMessage() + '\n')
        except Exception:
            self.handleError(record)


def record_trace(trace):
    trace_logger().info(json.dumps(trace, separators=(',', ':'), allow_nan=False))


def source_metadata():
    """Fingerprint backend source only; never read secrets, datasets or raw diffs."""
    import hashlib
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'server.py', root / 'requirements.txt']
    for directory in ('api', 'core', 'retrieval', 'ingestion', 'evaluation'):
        paths.extend((root / directory).rglob('*.py'))
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(paths) if path.is_file()}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError('Git metadata unavailable')
        return result.stdout.strip()
    try:
        commit = git('rev-parse', 'HEAD')
        status = git('status', '--porcelain', '--untracked-files=all', '--', *files.keys())
        changes = status.splitlines()
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        commit, changes = None, None
    return {'source_sha256': fingerprint, 'source_files_sha256': files,
            'git_commit': commit, 'source_working_tree_changes': changes,
            'scope': 'backend Python sources and requirements; hashes and paths only, no diff contents'}
