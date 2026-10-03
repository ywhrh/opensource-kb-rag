"""Log structured event names and timings without prompts, document text or keys."""
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger('rag_studio')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def event(name, **fields):
    logger.info(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'event': name, **fields}))
