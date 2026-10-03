"""Illustrative UI content, never persisted as experimental evidence."""

QUERY = "How do I handle HTTP 429 responses when calling an API?"

SAMPLE_RUNS = [
    {
        "experiment": {
            "name": "Vector-only baseline",
            "description": "Illustrative semantic retrieval baseline.",
        },
        "query": QUERY,
        "config_snapshot": {
            "retrieval_pipeline": {"mode": "vector_only"},
            "vector_search": {"top_k": 50},
            "hybrid_fusion": {"weights": {"vector_weight": 1.0, "bm25_weight": 0.0}},
            "rerank": {"top_n": 20},
        },
        "results": {
            "status": "PREVIEW",
            "answer": "HTTP errors can occur when a server cannot process a request. Check the response status and consult the API documentation.",
            "retrieval_latency_ms": 12.0,
            "algorithm_metrics": {
                "context_recall": 0.62, "context_precision": 0.68,
                "faithfulness": 0.81, "answer_correctness": 0.64,
                "permission_accuracy": None,
            },
            "load_test_metrics": {
                "actual_qps": None, "actual_p99_latency_ms": None,
                "total_requests": None, "failed_requests": None,
            },
            "sla_compliance": {
                "latency_passed": None, "qps_passed": None, "overall_status": None,
            },
            "evidence_chunks": [{
                "doc_id": "http-guide", "chunk_id": "http-guide:012",
                "section_title": "Understanding HTTP responses", "score": 0.83,
                "metadata": {"language": "Python", "topic": "HTTP"},
                "text": "## HTTP response status\n\nA response status describes the outcome of a request. A client should inspect the status before processing the response body.\n\n```python\nresponse = client.get(url)\nresponse.raise_for_status()\n```",
            }],
        },
    },
    {
        "experiment": {
            "name": "Hybrid + rerank",
            "description": "Illustrative exact-status-code retrieval with reranking.",
        },
        "query": QUERY,
        "config_snapshot": {
            "retrieval_pipeline": {"mode": "hybrid_rerank"},
            "vector_search": {"top_k": 50},
            "hybrid_fusion": {"weights": {"vector_weight": 0.7, "bm25_weight": 0.3}},
            "rerank": {"top_n": 20},
        },
        "results": {
            "status": "PREVIEW",
            "answer": "**HTTP 429 means Too Many Requests.** Respect the server's `Retry-After` header when present. Otherwise, use bounded exponential backoff with jitter and a retry limit. [1]\n\nAvoid retrying immediately: this adds load while the server is already rate limiting requests. [2]",
            "retrieval_latency_ms": 24.0,
            "algorithm_metrics": {
                "context_recall": 0.94, "context_precision": 0.91,
                "faithfulness": 0.96, "answer_correctness": 0.93,
                "permission_accuracy": None,
            },
            "load_test_metrics": {
                "actual_qps": None, "actual_p99_latency_ms": None,
                "total_requests": None, "failed_requests": None,
            },
            "sla_compliance": {
                "latency_passed": None, "qps_passed": None, "overall_status": None,
            },
            "evidence_chunks": [
                {
                    "doc_id": "api-client", "chunk_id": "api-client:042",
                    "section_title": "Rate limits and HTTP 429", "score": 0.92,
                    "metadata": {"status": "HTTP 429", "language": "Python", "topic": "Retry-After"},
                    "text": "## Rate limits and HTTP 429\n\nA `429 Too Many Requests` response indicates that a rate limit has been reached. When provided, the `Retry-After` header indicates how long to wait before retrying.\n\n```python\nif response.status_code == 429:\n    retry_after = response.headers.get(\"Retry-After\")\n    schedule_retry(retry_after)\n```\n\nThis snippet illustrates control flow; production code must parse both delay seconds and HTTP dates.",
                },
                {
                    "doc_id": "retry-guide", "chunk_id": "retry-guide:018",
                    "section_title": "Backoff and retry limits", "score": 0.88,
                    "metadata": {"topic": "Exponential backoff", "language": "Python"},
                    "text": "Use bounded exponential backoff with jitter for transient failures. Set a maximum retry count and avoid unbounded retry loops. Only retry operations that can safely be repeated.",
                },
            ],
        },
    },
]
