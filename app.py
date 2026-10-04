"""OpenKB-RAG Studio: cascading columns with disk-backed experiment leaves."""
from copy import deepcopy
import json
import os

import httpx
import streamlit as st
import yaml

from core.config_loader import ROOT, RAGConfig, load_config, revision
from core.experiment_cache import signature, find_runs, record_signature
from core.secrets import load_keys, save_key
from evaluation.golden_dataset_loader import load_dataset

API = os.getenv('RAG_API_URL', 'http://127.0.0.1:8000')
MODES = {'vector_only': 'Vector only', 'hybrid': 'Hybrid', 'hybrid_rerank': 'Hybrid + rerank'}
CATEGORIES = {'evaluation': 'RAGAS · 60-question evaluation', 'live_run': 'Live question & answer', 'load_test': 'Independent load test'}
ENUMS = {
    'vector_search.index_type': ['HNSW', 'IVF_FLAT', 'IVF_SQ8'],
    'vector_search.metric_type': ['COSINE', 'L2', 'IP'],
    'vector_search.quantization.type': ['none', 'SQ8', 'BINARY'],
    'bm25_search.tokenizer': ['standard'], 'hybrid_fusion.strategy': ['weighted_sum', 'RRF'],
    'ingestion_pipeline.chunking.strategy': ['fixed_size', 'semantic', 'recursive'],
    'ingestion_pipeline.chunking.semantic_config.breakpoint_threshold_type': ['percentile', 'standard_deviation', 'interquartile'],
    'load_testing.endpoint': ['/retrieve', '/query'],
}
METRICS = ['context_recall', 'context_precision', 'faithfulness', 'answer_correctness']
st.set_page_config(page_title='OpenKB-RAG Studio', page_icon='◈', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
[data-testid="stHeader"] {background:transparent!important;}
.block-container {max-width:1800px;padding:1rem 2rem;}
.block-container > [data-testid="stVerticalBlock"] {gap:.4rem;}
h1 {font-size:1.9rem!important;font-weight:650!important;letter-spacing:-.07rem;padding:.2rem 0!important;margin:0!important;}
h3 {font-size:1.1rem!important;letter-spacing:-.02rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
[data-testid="stMetric"] {background:#f5f5f7;border-radius:14px;padding:.9rem;margin-bottom:.6rem;}
[data-testid="stExpander"] {border-radius:12px;}
@keyframes revealBranch {from {opacity:0;transform:translateX(16px);} to {opacity:1;transform:translateX(0);}}
[data-testid="stColumn"] {animation:revealBranch .3s ease-out;}
@media (prefers-reduced-motion:reduce) {[data-testid="stColumn"] {animation:none;}}
.eyebrow {font-size:.7rem;letter-spacing:.16em;color:#86868b;}
</style>''', unsafe_allow_html=True)


def api(method, path, **kwargs):
    response = httpx.request(method, API + path, timeout=30, **kwargs)
    if response.is_error:
        raise ValueError(str(response.json().get('detail', 'Request failed')))
    return response.json()


if 'tree_config' not in st.session_state:
    st.session_state.tree_config = load_config().model_dump()
    st.session_state.tree_revision = revision()
    st.session_state.tree_keys = load_keys()
for key in list(st.session_state):
    if key.startswith(('node', 'key_')) or key in {'live_question', 'live_source_mode'}:
        st.session_state[key] = st.session_state[key]
for provider in ['openai', 'cohere']:
    st.session_state.setdefault('key_' + provider, st.session_state.tree_keys.get(provider, ''))
if 'progressive_workspace_version' not in st.session_state:
    st.session_state.progressive_workspace_version = 1
    st.session_state.start_ready = False
    st.session_state.branch_ready = False
    st.session_state.leaf_ready = False


def reset_from(level):
    for key in list(st.session_state):
        if any(key.startswith(f'node{n}_') for n in range(level, 4)):
            del st.session_state[key]
    if level <= 1:
        st.session_state.start_ready = False
    if level <= 2:
        st.session_state.branch_ready = False
    st.session_state.leaf_ready = False
    st.session_state.pop('tree_run', None)
    st.session_state.pop('leaf_record', None)


def root_changed():
    reset_from(1)


def get_value(config, path):
    for part in path.split('.'):
        config = config[part]
    return config


def changed(path, level):
    config = deepcopy(st.session_state.tree_config)
    parts = path.split('.')
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = st.session_state[f'node{level}_{path}']
    config['vector_search']['hnsw_params']['ef'] = max(config['vector_search']['hnsw_params']['ef'], config['vector_search']['top_k'])
    weight = config['hybrid_fusion']['weights']['vector_weight']
    config['hybrid_fusion']['weights']['bm25_weight'] = round(1-weight, 6)
    try:
        validated = RAGConfig.model_validate(config).model_dump()
        result = api('PUT', '/config', json={'config': validated, 'expected_revision': st.session_state.tree_revision})
        st.session_state.tree_config = result['config']
        st.session_state.tree_revision = result['revision']
        reset_from(level + 1)
        # Derived HNSW ef must also update its existing widget.
        if path == 'vector_search.top_k':
            st.session_state.pop('node2_vector_search.hnsw_params.ef', None)
        st.session_state.pop('tree_error', None)
    except (ValueError, httpx.HTTPError) as exc:
        st.session_state.tree_error = str(exc)
        st.session_state[f'node{level}_{path}'] = get_value(st.session_state.tree_config, path)


def control(path, label, level=2, options=None, **kwargs):
    key = f'node{level}_{path}'
    st.session_state.setdefault(key, get_value(st.session_state.tree_config, path))
    args = dict(key=key, on_change=changed, args=(path, level), **kwargs)
    value = st.session_state[key]
    if options:
        return st.selectbox(label, options, **args)
    if isinstance(value, bool):
        return st.toggle(label, **args)
    if isinstance(value, (int, float)):
        return st.number_input(label, **args)
    if isinstance(value, list):
        return st.multiselect(label, list(dict.fromkeys(METRICS + value)) if path == 'evaluation.metrics' else value, **args)
    return st.text_input(label, **args)


def advanced(mapping, prefix):
    for field, value in mapping.items():
        path = prefix + '.' + field
        if path in {'generation.provider', 'vector_search.enabled', 'bm25_search.enabled', 'rerank.enabled', 'hybrid_fusion.weights.bm25_weight'}:
            continue
        if isinstance(value, dict):
            st.markdown('**' + field.replace('_', ' ').title() + '**')
            advanced(value, path)
        else:
            control(path, field.replace('_', ' ').title(), options=ENUMS.get(path))


def credentials():
    return {provider + '_api_key': st.session_state['key_' + provider] for provider in ['openai', 'cohere']}


def save_credential(provider):
    try:
        save_key(provider, st.session_state['key_' + provider])
    except (OSError, ValueError):
        st.session_state.tree_error = 'Unable to save the separate API key file.'


def enqueue(endpoint, payload, wanted=None):
    try:
        job = api('POST', endpoint, json=payload)
        st.session_state.setdefault('tree_jobs', {})[job['job_id']] = {**job, 'signature': wanted}
        st.rerun()
    except (ValueError, httpx.HTTPError) as exc:
        st.error(str(exc))


def render_result(run, prefix):
    results = run['results']
    st.caption(run['experiment']['name'] + ' · ' + results['status'])
    if run['kind'] == 'evaluation':
        progress = results.get('evaluation_progress', {})
        st.caption(f"Saved scores: {progress.get('saved_scores', 0)}/{progress.get('total_metrics', 0)}")
        metrics = results.get('algorithm_metrics') or {}
        keys = list(dict.fromkeys(METRICS + list(metrics)))
        headings = [key.replace('_', ' ').title() for key in keys]
        scores = [f'{metrics[key]:.3f}' if metrics.get(key) is not None else '—' for key in keys]
        st.markdown('| ' + ' | '.join(headings) + ' |\n| ' + ' | '.join(['---'] * len(keys)) + ' |\n| ' + ' | '.join(scores) + ' |')
        with st.expander('评分完整性'):
            total = len(results.get('per_case', []))
            counts = results.get('metric_valid_case_counts') or {}
            for key, heading in zip(keys, headings):
                st.caption(f'{heading}: {counts.get(key, 0)}/{total} valid cases')
    if run['kind'] == 'load_test':
        for key in ['actual_qps', 'total_requests', 'failed_requests']:
            value = (results.get('load_test_metrics') or {}).get(key)
            st.metric(key.replace('_', ' ').title(), value if value is not None else '—')
    p99 = (results.get('load_test_metrics') or {}).get('actual_p99_latency_ms')
    st.caption('Measured P99 latency · ' + (f'{p99:.1f} ms' if p99 is not None else 'Not measured'))
    if results.get('evaluation_warning'):
        st.warning(results['evaluation_warning'])
    if results.get('error'):
        st.error(results['error'])
    case = results
    if run['kind'] == 'evaluation' and results.get('per_case'):
        cases = {c['case_id']: c for c in results['per_case']}
        if st.session_state.get(prefix + '_case') not in cases:
            st.session_state.pop(prefix + '_case', None)
        selected = st.selectbox('Inspect a question', list(cases), key=prefix + '_case')
        case = cases[selected]
        st.write(case['query'])
        with st.expander('Reference answer'):
            st.write(case.get('reference_answer'))
    elif run['kind'] == 'live_run':
        st.write(run.get('query'))
    if case.get('answer'):
        st.markdown('**LLM answer**')
        st.write(case['answer'])
    for i, chunk in enumerate(case.get('evidence_chunks', []), 1):
        with st.expander(f"Evidence {i} · {chunk.get('section_title', 'Source')}"):
            st.caption(chunk.get('metadata', {}).get('source', ''))
            st.write(chunk['text'])
    with st.expander('📂 展开查看/折叠历史实验配置'):
        st.json({k: v for k, v in run.items() if k != 'results'}, expanded=False)
        st.json(results, expanded=False)
        st.download_button('Download output JSON', json.dumps(run, ensure_ascii=False, indent=2), run['run_id'] + '.json', 'application/json', key=prefix + '_download')


@st.fragment(run_every=3)
def job_progress():
    for identifier, saved in list(st.session_state.get('tree_jobs', {}).items()):
        try:
            job = api('GET', '/jobs/' + identifier)
            st.session_state.tree_jobs[identifier] = {**job, 'signature': saved.get('signature')}
            st.caption(job['kind'] + ' · ' + job['status'] + ' · ' + job['message'])
            if saved['status'] in {'PENDING', 'RUNNING'} and job['status'] not in {'PENDING', 'RUNNING'}:
                st.rerun()
        except (ValueError, httpx.HTTPError) as exc:
            st.warning(str(exc))

def live_workspace():
    st.subheader('Live Run Workspace')
    job_progress()
    if st.session_state.get('live_workspace_page', 'input') == 'input':
        with st.container(border=True):
            control('experiment.name', 'Experiment name', level=0)
            control('experiment.description', 'Experiment description (optional)', level=0)
            with st.container(height=430, border=True):
                st.subheader('Input parameters')
                mode = control('retrieval_pipeline.mode', 'Retrieval mode', level=1, options=list(MODES), format_func=MODES.get)
                control('vector_search.model_name', 'Embedding model', level=1, options=['BAAI/bge-m3', 'BAAI/bge-large-en-v1.5', 'text-embedding-3-small'])
                control('generation.llm_model', 'Answer model', level=1, options=['llama3.1:8b', 'gpt-4o-mini'])
                control('vector_search.top_k', 'Vector Top-K', min_value=1, max_value=1000)
                if mode != 'vector_only':
                    control('bm25_search.model_name', 'BM25 engine', level=1, options=['BM25Okapi', 'Milvus Built-in Sparse BM25'])
                    control('bm25_search.top_k', 'BM25 Top-K', min_value=1, max_value=1000)
                if mode == 'hybrid_rerank':
                    control('rerank.model_name', 'Reranker', level=1, options=['bge-reranker-base', 'Cohere Rerank v3'])
                    control('rerank.top_n', 'Rerank Top-N', min_value=1, max_value=2000)
                with st.expander('Advanced input parameters'):
                    for section in ['vector_search', 'hybrid_fusion', 'ingestion_pipeline', 'generation']:
                        mapping = deepcopy(st.session_state.tree_config[section])
                        for field in {'vector_search': ['top_k', 'model_name'], 'generation': ['llm_model']}.get(section, []):
                            mapping.pop(field, None)
                        advanced(mapping, section)
                with st.expander('API credentials'):
                    for provider in ['openai', 'cohere']:
                        st.text_input(provider.title() + ' API key', type='password', key='key_' + provider, on_change=save_credential, args=(provider,))
                    st.caption('Saved separately in Git-ignored .secrets/api_keys.json.')
            source_mode = st.selectbox('Data source', ['Project default', 'Upload documents'], key='live_source_mode')
            live_source = 'project'
            if source_mode == 'Upload documents':
                files = st.file_uploader('Upload documents', type=['pdf', 'md', 'txt', 'html', 'docx'], accept_multiple_files=True, key='live_upload')
                if st.button('Save uploaded documents', disabled=not files):
                    try:
                        response = api('POST', '/documents', files=[('files', (f.name, f.getvalue(), f.type)) for f in files])
                        st.session_state.live_uploaded_source = response['source_id']
                        st.success('Documents saved. Build their index before running.')
                    except (ValueError, httpx.HTTPError) as exc:
                        st.error(str(exc))
                live_source = st.session_state.get('live_uploaded_source')
            else:
                st.caption('Project documents: data/documents')
            if st.button('Build / update document index', disabled=not live_source):
                enqueue('/ingest', {'config_snapshot': st.session_state.tree_config, 'source_id': live_source, 'credentials': credentials()})
            question = st.text_area('Question about the data source', key='live_question', height=100)
            if st.button('Run', type='primary', use_container_width=True, disabled=not live_source or not question.strip() or not st.session_state.tree_config['experiment']['name'].strip()):
                st.session_state.live_pending = {'config_snapshot': deepcopy(st.session_state.tree_config), 'source_id': live_source, 'query': question.strip(), 'credentials': credentials()}
                st.session_state.live_workspace_page = 'result'
                st.session_state.pop('live_output', None)
                st.rerun()
        return
    if st.button('← Back to experiment setup'):
        st.session_state.live_workspace_page = 'input'
        st.rerun()
    pending = st.session_state.pop('live_pending', None)
    if pending:
        st.subheader(pending['config_snapshot']['experiment']['name'])
        with st.container(border=True):
            st.markdown('**Question**')
            st.write(pending['query'])
            st.markdown('**RAG answer**')
            answer_slot = st.empty()
            status_slot = st.empty()
            answer = ''
            try:
                with httpx.stream('POST', API + '/query/stream', json=pending, timeout=httpx.Timeout(600, connect=10)) as response:
                    if response.is_error:
                        response.read()
                        raise ValueError(str(response.json().get('detail', 'Live run failed')))
                    for line in response.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event['type'] == 'token':
                            answer += event['text']
                            answer_slot.markdown(answer)
                        elif event['type'] == 'status':
                            status_slot.caption(event['message'])
                        elif event['type'] == 'result':
                            st.session_state.live_output = event['output']
                status_slot.empty()
            except (ValueError, httpx.HTTPError) as exc:
                st.error(str(exc))
    run = st.session_state.get('live_output')
    if not run:
        return
    if not pending:
        st.subheader(run['experiment']['name'])
        with st.container(border=True):
            st.markdown('**Question**')
            st.write(run['query'])
            st.markdown('**RAG answer**')
            st.write(run['results'].get('answer'))
    if run['results'].get('error'):
        st.error(run['results']['error'])
    with st.container(border=True):
        st.subheader('Output evidence')
        chunks = run['results'].get('evidence_chunks', [])
        if not chunks:
            st.caption('No evidence found.')
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"[{i}] {chunk.get('section_title', 'Source')}", expanded=i == 1):
                st.caption(chunk.get('metadata', {}).get('source', ''))
                st.write(chunk['text'])
    display = deepcopy(run)
    for field in ['algorithm_metrics', 'load_test_metrics', 'sla_compliance']:
        display['results'].pop(field, None)
    input_snapshot = {k: v for k, v in display.items() if k != 'results'}
    for col, title, payload, suffix in zip(st.columns(2), ['Input Config JSON', 'Output JSON'], [input_snapshot, display], ['input', 'output']):
        with col, st.container(border=True):
            st.subheader(title)
            st.json(payload, expanded=False)
            st.download_button('Download ' + title, json.dumps(payload, ensure_ascii=False, indent=2), run['run_id'] + '_' + suffix + '.json', 'application/json', key='live_download_' + suffix)


def reset_tuning_workspace():
    """Clear UI selections so the configuration flow starts again."""
    for key in list(st.session_state):
        if key.startswith('group_') or key in {'grouped_paths', 'grouped_counter', 'grouped_checkout_used'}:
            del st.session_state[key]


st.markdown('<div class="eyebrow">OPEN-SOURCE DOCUMENTATION · EXPERIMENT WORKSPACE</div>', unsafe_allow_html=True)
st.title('OpenKB-RAG Studio')
page = st.radio('Workspace', ['Live Run', 'Tuning & Ablation'], horizontal=True, key='workspace_page', label_visibility='collapsed')
if page == 'Live Run':
    live_workspace()
    st.stop()
refresh_column, reset_column, _ = st.columns([.17, .09, .74], gap='small')
with refresh_column:
    if st.button('Refresh saved results', width='stretch'):
        st.rerun()
with reset_column:
    st.button('Reset', on_click=reset_tuning_workspace, width='stretch')


# History inputs supply node options; no live configuration changes occur here.
history_runs = []
for output_path in sorted((ROOT / 'data/experiments_history').glob('*_output.yaml'), reverse=True):
    input_path = output_path.with_name(output_path.name.replace('_output.yaml', '_input.yaml'))
    if not input_path.exists():
        continue
    try:
        saved = yaml.safe_load(input_path.read_text())
        if saved['kind'] == 'evaluation' and saved['source_id'] == 'project' and len(saved.get('dataset', {}).get('snapshot', [])) == 60:
            history_runs.append(saved)
    except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError):
        continue
if not history_runs:
    st.info('No paired 60-question project experiment configurations found.')
    st.stop()

from grouped_workspace import grouped_workspace
completed_paths = grouped_workspace(history_runs)


def execute_branch(item):
    config = RAGConfig.model_validate(item['config'])
    from ingestion.vector_store_writer import require_index
    current_index = require_index(config, item['source'])
    if current_index['collection_name'] != item['manifest']['collection_name']:
        raise ValueError('This branch index changed. Create a new branch before running.')
    if item['kind'] == 'evaluation':
        _, current_hash = load_dataset(config.evaluation.dataset_path)
        if current_hash != item['dataset']['sha256']:
            raise ValueError('This branch dataset changed. Create a new branch before running.')
    needs_openai = config.vector_search.model_name == 'text-embedding-3-small' or config.generation.llm_model == 'gpt-4o-mini' or (item['kind'] == 'evaluation' and config.evaluation.judge_model == 'gpt-4o-mini')
    if needs_openai and not credentials()['openai_api_key'] and not os.getenv('OPENAI_API_KEY'):
        raise ValueError('Enter an OpenAI key in Start before running.')
    if config.rerank.enabled and config.rerank.model_name == 'Cohere Rerank v3' and not credentials()['cohere_api_key'] and not os.getenv('COHERE_API_KEY'):
        raise ValueError('Enter a Cohere key in Start before running.')
    if not config.experiment.name.strip():
        raise ValueError('Enter an experiment name in Start before running.')
    payload = {'config_snapshot': config.model_dump(), 'source_id': item['source'], 'credentials': credentials()}
    if item['kind'] == 'evaluation':
        enqueue('/evaluate', {**payload, 'modes': [config.retrieval_pipeline.mode]}, item['signature'])
    elif item['kind'] == 'load_test':
        enqueue('/load-tests', {**payload, 'queries': item['queries']}, item['signature'])
    else:
        api('POST', '/query', json={**payload, 'query': item['query'], 'index_collection': item['manifest']['collection_name']})
        st.rerun()


def branch_card(item, prefix):
    hits, corrupt = find_runs(item['signature'], item['kind'])
    if corrupt:
        st.warning(f'{len(corrupt)} unreadable history pairs skipped.')
    if hits:
        ids = [r['run_id'] for r in hits]
        record_key = prefix + '_record'
        if st.session_state.get(record_key) not in ids:
            st.session_state.pop(record_key, None)
            if item.get('preferred_run') in ids:
                st.session_state[record_key] = item['preferred_run']
        chosen = st.selectbox('Saved matching run', ids, key=record_key)
        run = next(r for r in hits if r['run_id'] == chosen)
        st.success('Cache hit · Paired YAML snapshots')
        render_result(run, prefix)
        if run['results']['status'] in {'PARTIAL', 'FAILED', 'INTERRUPTED', 'CANCELLED'} and item['kind'] == 'evaluation':
            if st.button('Resume missing scores', key=prefix + '_resume'):
                try:
                    enqueue('/evaluations/' + run['run_id'] + '/resume', credentials(), item['signature'])
                except (ValueError, httpx.HTTPError) as exc:
                    st.error(str(exc))
    else:
        st.info('No saved result for this branch.')
        active = any(j.get('signature') == item['signature'] and j['status'] in {'PENDING', 'RUNNING'} for j in st.session_state.get('tree_jobs', {}).values())
        if active:
            st.caption('This branch is running. Saved results appear here as checkpoints arrive.')
        if st.button('🚀 Run Experiment', type='primary', key=prefix + '_run', disabled=active):
            try:
                execute_branch(item)
            except (ValueError, OSError, httpx.HTTPError) as exc:
                st.error(str(exc))


if completed_paths:
    st.divider()
    st.subheader('Matching results')
    for offset in range(0, len(completed_paths), 3):
        row = completed_paths[offset:offset+3]
        for item, column in zip(row, st.columns(len(row), gap='large')):
            with column, st.container(border=True):
                st.subheader(item['title'])
                hits, _ = find_runs(item['signature'], 'evaluation')
                st.caption(f'{len(hits)} fully matching historical results')
                if hits:
                    # Show every exact match, rather than silently picking just one run.
                    for run in hits:
                        with st.expander(run['experiment']['name'] + ' · ' + run['results']['status'], expanded=len(hits) == 1):
                            render_result(run, 'path_' + item['id'] + '_' + run['run_id'])
                else:
                    branch_card(item, 'path_' + item['id'])



job_progress()

