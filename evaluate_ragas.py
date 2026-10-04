"""Incremental ablation evaluation with durable case/metric checkpoints and explicit resume."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
import re
from time import perf_counter

import yaml
from api.schemas import Credentials
from core.config_loader import ROOT, RAGConfig, load_config, digest
from core.history import new_output, persist, folder
from core.pipeline import run_query
from evaluation.golden_dataset_loader import load_dataset, GoldenCase
from evaluation.ragas_metric_runner import MetricScorer
from ingestion.vector_store_writer import require_index


@contextmanager
def evaluation_lock():
    """One local evaluation at a time, including resumes of the same checkpoint."""
    path = ROOT / 'data/jobs/evaluation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('An evaluation is already running. Wait for it to finish.')
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def read_evaluation(identifier):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identifier):
        raise ValueError('Invalid evaluation run ID')
    path = folder('evaluation') / f'{identifier}_output.yaml'
    if not path.exists():
        raise ValueError('Evaluation record not found')
    record = yaml.safe_load(path.read_text())
    if record.get('kind') != 'evaluation' or record.get('run_id') != identifier:
        raise ValueError('Not an evaluation record')
    return record


def resume_inputs(identifier):
    """Resume from the recorded dataset/config, never from current UI controls."""
    record = read_evaluation(identifier)
    if record['results']['status'] in {'PENDING', 'RUNNING', 'SUCCESS'}:
        raise ValueError('Only interrupted, cancelled, failed or partial evaluations can be resumed')
    config = RAGConfig.model_validate(record['config_snapshot'])
    saved = record.get('dataset', {}).get('snapshot')
    if not saved:
        raise ValueError('This record has no dataset snapshot; start a new evaluation')
    cases = [GoldenCase.model_validate(case) for case in saved]
    if len({case.id for case in cases}) != len(cases) or digest([case.model_dump() for case in cases]) != record['dataset']['sha256']:
        raise ValueError('The saved dataset snapshot is inconsistent')
    manifest = require_index(config, record['source_id'])
    if manifest['collection_name'] != record['knowledge_base']['collection_name']:
        raise ValueError('The source index changed. Start a new evaluation instead of mixing indexes.')
    return record, config, cases


def valid_score(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def summarize(record, config, total_cases):
    """Partial averages use successful saved scores, with their denominators visible."""
    results = record['results']
    cases = results['per_case']
    keys = config.evaluation.metrics
    counts, averages = {}, {}
    for key in keys:
        values = [case['algorithm_metrics'][key] for case in cases
                  if case['metric_status'][key] == 'SUCCESS' and valid_score(case['algorithm_metrics'].get(key))]
        counts[key] = len(values)
        averages[key] = sum(values) / len(values) if values else None
    attempts = sum(case['metric_status'][key] in {'SUCCESS', 'FAILED'} for case in cases for key in keys)
    results['algorithm_metrics'] = averages
    results['metric_valid_case_counts'] = counts
    results['evaluation_progress'].update(
        total_cases=total_cases, answered_cases=sum(case['answer_status'] == 'SUCCESS' for case in cases),
        fully_scored_cases=sum(all(case['metric_status'][key] == 'SUCCESS' for key in keys) for case in cases),
        total_metrics=total_cases * len(keys), completed_metric_attempts=attempts,
        saved_scores=sum(counts.values()), failed_metrics=attempts - sum(counts.values()))
    latencies = [case['retrieval_latency_ms'] for case in cases if case['answer_status'] == 'SUCCESS']
    results['retrieval_latency_ms'] = sum(latencies) / len(latencies) if latencies else None
    results['category_metrics'] = {}
    for category in sorted({case['category'] for case in cases}):
        subset = [case for case in cases if case['category'] == category]
        results['category_metrics'][category] = {}
        for key in keys:
            values = [case['algorithm_metrics'][key] for case in subset
                      if case['metric_status'][key] == 'SUCCESS' and valid_score(case['algorithm_metrics'].get(key))]
            results['category_metrics'][category][key] = sum(values) / len(values) if values else None
    recall = averages.get('context_recall')
    results['sla_compliance'] = {
        'context_recall_passed': recall >= config.evaluation.sla_targets.min_context_recall
        if recall is not None and counts.get('context_recall') == total_cases else None,
        'latency_passed': None, 'overall_status': 'NOT_EVALUATED'}
    if any(count < total_cases for count in counts.values()):
        results['evaluation_warning'] = 'Incomplete evaluation. Averages use only successful saved scores; valid-case counts are shown.'
    else:
        results.pop('evaluation_warning', None)


def evaluate_record(record, config, cases, credentials, progress=None):
    """Save before and after every answer/metric; successful checkpoints are reused."""
    if not config.evaluation.metrics:
        raise ValueError('Select at least one Ragas metric')
    keys = config.evaluation.metrics
    existing = {case['case_id']: case for case in record['results'].get('per_case', [])}
    per_case = []
    for case in cases:
        saved = existing.get(case.id, {})
        scores = saved.get('algorithm_metrics', {})
        old_status = saved.get('metric_status', {})
        # Old outputs can retain completed answers/scores even without the new status fields.
        statuses = {key: 'SUCCESS' if valid_score(scores.get(key)) and old_status.get(key, 'SUCCESS') == 'SUCCESS'
                    else 'PENDING' for key in keys}
        answered = saved.get('answer_status', 'SUCCESS' if saved.get('answer') is not None else 'PENDING') == 'SUCCESS'
        per_case.append({**saved, 'case_id': case.id, 'category': case.category, 'query': case.question,
                         'reference_answer': case.reference_answer, 'answer_status': 'SUCCESS' if answered else 'PENDING',
                         'answer': saved.get('answer'), 'evidence_chunks': saved.get('evidence_chunks', []),
                         'retrieval_latency_ms': saved.get('retrieval_latency_ms'),
                         'algorithm_metrics': {key: scores.get(key) if statuses[key] == 'SUCCESS' else None for key in keys},
                         'metric_status': statuses, 'metric_errors': {}})
    results = record['results']
    results.update(status='RUNNING', per_case=per_case, checkpoint_version=1,
                   evaluation_progress={'phase': 'starting', 'current_case_id': None, 'current_metric': None})
    results.pop('error', None)
    previous_ms = results.get('total_latency_ms') or 0
    started = perf_counter()
    scorer = None
    mode = config.retrieval_pipeline.mode

    def checkpoint(phase, message, case=None, metric=None):
        results['evaluation_progress'].update(phase=phase, current_case_id=case, current_metric=metric)
        results['total_latency_ms'] = previous_ms + (perf_counter() - started) * 1000
        results['last_checkpoint_at'] = datetime.now(timezone.utc).isoformat()
        summarize(record, config, len(cases))
        persist(record)
        if progress:
            p = results['evaluation_progress']
            progress(f'{mode}: {message}; saved scores {p["saved_scores"]}/{p["total_metrics"]}; failed metrics {p["failed_metrics"]}')

    checkpoint('starting', 'loaded saved checkpoints')
    try:
        for index, case in enumerate(per_case, 1):
            label = f'case {index}/{len(cases)} ({case["case_id"]})'
            if case['answer_status'] != 'SUCCESS':
                case['answer_status'] = 'RUNNING'
                checkpoint('answering', f'{label}: generating answer', case['case_id'])
                answer = run_query(case['query'], record['source_id'], config, credentials, save=False,
                                   expected_collection=record['knowledge_base']['collection_name'])
                if answer['results']['status'] != 'SUCCESS':
                    case.update(answer_status='FAILED', answer_error=answer['results'].get('error', 'Answer generation failed'))
                    checkpoint('answer_failed', f'{label}: answer failed', case['case_id'])
                    raise ValueError(f'{case["case_id"]}: {case["answer_error"]}')
                case.update(answer_status='SUCCESS', answer=answer['results']['answer'],
                            evidence_chunks=answer['results']['evidence_chunks'],
                            retrieval_latency_ms=answer['results']['retrieval_latency_ms'])
                case.pop('answer_error', None)
                checkpoint('answer_saved', f'{label}: answer and evidence saved', case['case_id'])
            row = {'user_input': case['query'], 'reference': case['reference_answer'], 'response': case['answer'],
                   'retrieved_contexts': [chunk['text'] for chunk in case['evidence_chunks']]}
            for metric_index, key in enumerate(keys, 1):
                if case['metric_status'][key] == 'SUCCESS':
                    continue
                case['metric_status'][key] = 'RUNNING'
                checkpoint('scoring', f'{label}: scoring {key} ({metric_index}/{len(keys)})', case['case_id'], key)
                try:
                    if scorer is None:
                        scorer = MetricScorer(config, credentials)
                    case['algorithm_metrics'][key] = scorer.score(row, key)
                    case['metric_status'][key] = 'SUCCESS'
                except Exception as exc:
                    case['algorithm_metrics'][key] = None
                    case['metric_status'][key] = 'FAILED'
                    # Provider exceptions can include private prompts; persist only the class name.
                    case['metric_errors'][key] = f'{type(exc).__name__}: scoring failed; resume to retry this metric'
                checkpoint('metric_saved', f'{label}: {key} {case["metric_status"][key]} saved', case['case_id'], key)
        complete = all(case['answer_status'] == 'SUCCESS' and all(case['metric_status'][key] == 'SUCCESS' for key in keys)
                       for case in per_case)
        results['status'] = 'SUCCESS' if complete else 'PARTIAL'
        checkpoint('complete', 'evaluation finished')
    except Exception as exc:
        results.update(status='FAILED', error=str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__}: evaluation failed')
        checkpoint('failed', 'evaluation stopped; saved checkpoints retained')
    return record['run_id']


def run_ablation(config, source_id='project', credentials=None, modes=None, progress=None):
    with evaluation_lock():
        cases, dataset_hash = load_dataset(config.evaluation.dataset_path)
        manifest = require_index(config, source_id)
        outputs = []
        for mode in modes or ['vector_only', 'hybrid', 'hybrid_rerank']:
            variant = config.model_copy(deep=True)
            variant.retrieval_pipeline.mode = mode
            variant.bm25_search.enabled = mode != 'vector_only'
            variant.rerank.enabled = mode == 'hybrid_rerank'
            record = new_output(variant, name=f'{config.experiment.name}_{mode}', kind='evaluation', source_id=source_id,
                                dataset={'path': config.evaluation.dataset_path, 'sha256': dataset_hash, 'cases': len(cases),
                                         'snapshot': [case.model_dump() for case in cases]}, knowledge_base=manifest)
            persist(record, initial=True)
            outputs.append(evaluate_record(record, variant, cases, credentials, progress))
            if record['results']['status'] == 'FAILED':
                raise ValueError(f'Evaluation stopped at {mode}; saved checkpoints can be resumed')
        return outputs


def resume_evaluation(identifier, credentials=None, progress=None):
    with evaluation_lock():
        record, config, cases = resume_inputs(identifier)
        run_id = evaluate_record(record, config, cases, credentials, progress)
        if record['results']['status'] == 'FAILED':
            raise ValueError('Evaluation stopped; its saved checkpoints can be resumed')
        return {'run_ids': [run_id], 'status': record['results']['status']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name')
    parser.add_argument('--source', default='project')
    parser.add_argument('--modes', nargs='+', choices=['vector_only', 'hybrid', 'hybrid_rerank'])
    parser.add_argument('--resume', metavar='RUN_ID')
    args = parser.parse_args()
    if args.resume:
        print(resume_evaluation(args.resume, Credentials(), progress=print))
    else:
        if not args.name:
            parser.error('--name is required unless --resume is provided')
        config = load_config()
        config.experiment.name = args.name
        print(run_ablation(config, args.source, Credentials(), args.modes, progress=print))
