"""Batch ablation evaluation, separate from Live Run and load testing."""
import argparse
from copy import deepcopy
from time import perf_counter

from api.schemas import Credentials
from core.config_loader import load_config
from core.history import new_output, persist
from core.pipeline import run_query
from evaluation.golden_dataset_loader import load_dataset
from evaluation.ragas_metric_runner import score_cases
from ingestion.vector_store_writer import require_index


def run_ablation(config, source_id='project', credentials=None, modes=None, progress=None):
    cases, dataset_hash = load_dataset(config.evaluation.dataset_path)
    manifest = require_index(config, source_id)
    outputs = []
    for mode in modes or ['vector_only', 'hybrid', 'hybrid_rerank']:
        variant = config.model_copy(deep=True)
        variant.retrieval_pipeline.mode = mode
        variant.bm25_search.enabled = mode != 'vector_only'
        variant.rerank.enabled = mode == 'hybrid_rerank'
        record = new_output(variant, name=f'{config.experiment.name}_{mode}', kind='evaluation',
                            source_id=source_id, dataset={'path': config.evaluation.dataset_path, 'sha256': dataset_hash, 'cases': len(cases), 'snapshot': [case.model_dump() for case in cases]})
        record['knowledge_base'] = manifest
        persist(record, initial=True)
        started = perf_counter()
        try:
            rows, per_case = [], []
            for index, case in enumerate(cases, 1):
                if progress:
                    progress(f'{mode}: answering case {index}/{len(cases)}')
                answer = run_query(case.question, source_id, variant, credentials, save=False, expected_collection=manifest["collection_name"])
                if answer['results']['status'] != 'SUCCESS':
                    raise RuntimeError(f'Case {case.id} failed: {answer["results"].get("error", "pipeline error")}')
                rows.append({'user_input': case.question, 'reference': case.reference_answer,
                             'response': answer['results']['answer'],
                             'retrieved_contexts': [chunk['text'] for chunk in answer['results']['evidence_chunks']]})
                per_case.append({'case_id': case.id, 'category': case.category, 'query': case.question,
                                 'reference_answer': case.reference_answer, 'answer': answer['results']['answer'],
                                 'evidence_chunks': answer['results']['evidence_chunks'],
                                 'retrieval_latency_ms': answer['results']['retrieval_latency_ms']})
            if progress:
                progress(f'{mode}: computing Ragas scores with {config.evaluation.judge_model}')
            metrics, scores, counts = score_cases(rows, variant, credentials)
            for result, score in zip(per_case, scores):
                result['algorithm_metrics'] = score
            complete = all(count == len(cases) for count in counts.values())
            record['results'].update(status='SUCCESS' if complete else 'PARTIAL', algorithm_metrics=metrics,
                                     metric_valid_case_counts=counts, per_case=per_case,
                                     total_latency_ms=(perf_counter() - started) * 1000)
            record['results']['retrieval_latency_ms'] = sum(case['retrieval_latency_ms'] for case in per_case) / len(per_case)
            recall = metrics.get('context_recall')
            record['results']['sla_compliance'] = {'context_recall_passed': recall >= config.evaluation.sla_targets.min_context_recall if recall is not None and complete else None, 'latency_passed': None, 'overall_status': 'NOT_EVALUATED'}
            record['results']['category_metrics'] = {}
            for category in sorted({case['category'] for case in per_case}):
                subset = [case for case in per_case if case['category'] == category]
                record['results']['category_metrics'][category] = {}
                for key in config.evaluation.metrics:
                    values = [case['algorithm_metrics'][key] for case in subset if case['algorithm_metrics'].get(key) is not None]
                    record['results']['category_metrics'][category][key] = sum(values) / len(values) if values else None
            if not complete:
                record['results']['evaluation_warning'] = 'Some judge outputs could not be scored. Missing values are null; averages use only scored cases.'
        except Exception as exc:
            record['results'].update(status='FAILED', error=str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__}: evaluation failed; see service/model setup',
                                     total_latency_ms=(perf_counter() - started) * 1000)
        persist(record)
        outputs.append(record['run_id'])
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', default='project')
    parser.add_argument('--modes', nargs='+', choices=['vector_only', 'hybrid', 'hybrid_rerank'])
    args = parser.parse_args()
    config = load_config()
    config.experiment.name = args.name
    print(run_ablation(config, args.source, Credentials(), args.modes, progress=print))
