"""Dedicated Locust scenario using a frozen input YAML; never evaluate Ragas."""
import json
import os
from pathlib import Path
import random

from locust import HttpUser, task, constant, events
import yaml

INPUT_PATH = Path(os.environ['RAG_LOAD_INPUT'])
INPUT = yaml.safe_load(INPUT_PATH.read_text())
CONFIG = INPUT['config_snapshot']
CREDENTIALS = json.loads(os.getenv('RAG_LOAD_CREDENTIALS', '{}'))


class RAGUser(HttpUser):
    wait_time = constant(0)

    @task
    def request_pipeline(self):
        payload = {'query': random.choice(INPUT['queries']), 'source_id': INPUT['source_id'],
                   'config_snapshot': CONFIG, 'credentials': CREDENTIALS, 'persist_history': False, 'index_collection': INPUT['knowledge_base']['collection_name']}
        endpoint = CONFIG['load_testing']['endpoint']
        with self.client.post(endpoint, json=payload, name=endpoint, catch_response=True, timeout=600) as response:
            if response.status_code != 200:
                response.failure(f'HTTP {response.status_code}')
            else:
                try:
                    data = response.json()
                    if data.get('results', {}).get('status') == 'FAILED':
                        response.failure('Pipeline execution failed')
                except ValueError:
                    response.failure('Invalid JSON response')


@events.quitting.add_listener
def write_summary(environment, **kwargs):
    stats = environment.stats.total
    payload = {
        'actual_qps': stats.total_rps,
        'actual_p99_latency_ms': stats.get_response_time_percentile(0.99) if stats.num_requests else None,
        'total_requests': stats.num_requests,
        'failed_requests': stats.num_failures,
    }
    destination = Path(os.environ['RAG_LOAD_METRICS'])
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(destination)
