"""Five grouped configuration cards with checkout copies and local branch state."""
from copy import deepcopy
from itertools import product
import streamlit as st
from core.config_loader import RAGConfig, digest
from core.experiment_cache import signature, find_runs

TITLES = ['Retrieval', 'Embedding & Index', 'Search & Rerank', 'Answer & Evaluation', 'Results']
MODES = {'vector_only': 'Vector only', 'hybrid': 'Hybrid', 'hybrid_rerank': 'Hybrid + rerank'}


def get(config, path):
    for part in path.split('.'):
        config = config[part]
    return config


def put(config, path, value):
    parts = path.split('.')
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def card_value(config, stage):
    if stage == 0:
        return config['retrieval_pipeline']
    if stage == 1:
        vector = deepcopy(config['vector_search'])
        vector.pop('top_k')
        return vector
    if stage == 2:
        value = {'vector_top_k': config['vector_search']['top_k']}
        if config['retrieval_pipeline']['mode'] != 'vector_only':
            value.update(bm25=config['bm25_search'], fusion=config['hybrid_fusion'])
        if config['retrieval_pipeline']['mode'] == 'hybrid_rerank':
            value['rerank'] = config['rerank']
        return value
    return {'generation': config['generation'], 'evaluation': config['evaluation']}


def fingerprint(item, stage):
    return digest(card_value(item['config'], stage))


def comparison_title(item, paths):
    """Name each result using the settings that differ in the checked-out card."""
    if len(paths) < 2:
        return MODES[item['config']['retrieval_pipeline']['mode']]
    stage = paths[1]['checkout_stage']
    def flatten(value, prefix=''):
        fields = {}
        for key, entry in value.items():
            name = prefix + key
            if isinstance(entry, dict):
                fields.update(flatten(entry, name + '.'))
            else:
                fields[name] = entry
        return fields
    candidates = [flatten(card_value(p['config'], stage)) for p in paths]
    own = flatten(card_value(item['config'], stage))
    labels = {'mode': 'Retrieval', 'model_name': 'Model', 'index_type': 'Index',
              'metric_type': 'Metric', 'top_k': 'Top-K', 'vector_top_k': 'Vector Top-K',
              'ef': 'HNSW ef', 'M': 'HNSW M', 'efConstruction': 'efConstruction',
              'llm_model': 'Answer model', 'judge_model': 'RAGAS judge', 'top_n': 'Rerank Top-N'}
    differences = []
    for key in dict.fromkeys(k for candidate in candidates for k in candidate):
        if all(candidate.get(key) == candidates[0].get(key) for candidate in candidates):
            continue
        name = labels.get(key.rsplit('.', 1)[-1], key.replace('_', ' ').replace('.', ' · '))
        if key == 'model_name' and stage == 1:
            name = 'Embedding'
        elif key == 'bm25.model_name':
            name = 'BM25'
        elif key == 'rerank.model_name':
            name = 'Reranker'
        value = own.get(key, 'Not applicable')
        if key == 'mode':
            value = MODES.get(value, value)
        differences.append(f'{name}: {value}')
    return ' · '.join(differences) or 'Same configuration'


def flatten_fields(value, prefix=''):
    fields = {}
    for key, entry in value.items():
        name = prefix + key
        if isinstance(entry, dict):
            fields.update(flatten_fields(entry, name + '.'))
        else:
            fields[name] = entry
    return fields


def shared_configuration(config, split):
    fields = flatten_fields(config)
    def owned(key):
        if split == 0:
            return key.startswith('retrieval_pipeline.') or key in {'bm25_search.enabled', 'rerank.enabled'}
        if split == 1:
            return key.startswith('vector_search.') and key != 'vector_search.top_k'
        if split == 2:
            return key == 'vector_search.top_k' or key.startswith(('bm25_search.', 'hybrid_fusion.', 'rerank.'))
        return key.startswith(('generation.', 'evaluation.'))
    return digest({k: v for k, v in fields.items() if not owned(k)
                   and not k.startswith(('experiment.', 'load_testing.')) and k != 'evaluation.dataset_path'})


def existing_choices(history, paths, identifier, stage, preceding, field):
    """Offer only values with a saved full configuration for every active path."""
    active = next(i for i, p in enumerate(paths) if p['id'] == identifier)
    root = paths[0]
    records = [r for r in history if r['dataset']['sha256'] == root['dataset']['sha256']
               and all(r['knowledge_base'].get(k) == root['manifest'].get(k)
                       for k in ('collection_name', 'source_revision'))]
    split = paths[1]['checkout_stage'] if len(paths) == 2 else None
    candidates = []
    for i, item in enumerate(paths):
        candidates.append([r for r in records
            if all(card_value(r['config_snapshot'], n) == card_value(item['config'], n) for n in range(stage))
            and (i != active or all(get(r['config_snapshot'], name) == get(item['config'], name) for name in preceding))
            and (split != stage or i == active or card_value(r['config_snapshot'], split) == card_value(item['config'], split))])
    choices = []
    for pair in product(*candidates):
        if split is not None and shared_configuration(pair[0]['config_snapshot'], split) != shared_configuration(pair[1]['config_snapshot'], split):
            continue
        value = get(pair[active]['config_snapshot'], field)
        if not any(value == old[0] for old in choices):
            choices.append((value, pair))
    return choices


def edit_existing(identifier, stage, path, choices):
    value = st.session_state[f'group_{identifier}_{stage}_{path}']
    paths = st.session_state.grouped_paths
    if value is None:
        for item in paths:
            item['ready'][stage:] = [False] * (4-stage)
        return
    pair = next((pair for candidate, pair in choices if candidate == value), None)
    if pair is None:
        return
    for item, record in zip(paths, pair):
        item['config'] = deepcopy(record['config_snapshot'])
        item.pop('focus_stage', None)
        if item['id'] == identifier or item['nodes'][stage] == next(p for p in paths if p['id'] == identifier)['nodes'][stage]:
            item['ready'][stage] = True
        item['ready'][stage+1:] = [False] * (3-stage)
    # Refresh all widget defaults from the selected, coherent history snapshots.
    for key in list(st.session_state):
        if key.startswith('group_'):
            del st.session_state[key]


def select_card(identifier, stage):
    """Return to this configuration and discard the later comparison branches."""
    item = deepcopy(next(p for p in st.session_state.grouped_paths if p['id'] == identifier))
    item.update(id='1', nodes=['1_' + str(n) for n in range(5)],
                source_cards={}, checked_cards={}, focus_stage=stage)
    item['ready'][stage+1:] = [False] * (3-stage)
    for field in ('parent', 'checkout_stage', 'error'):
        item.pop(field, None)
    st.session_state.grouped_paths = [item]
    st.session_state.grouped_counter = 1
    st.session_state.grouped_checkout_used = False
    for key in list(st.session_state):
        if key.startswith('group_'):
            del st.session_state[key]


def edit(identifier, stage, path):
    item = next(p for p in st.session_state.grouped_paths if p['id'] == identifier)
    value = st.session_state[f'group_{identifier}_{stage}_{path}']
    if value is None:
        for branch in st.session_state.grouped_paths:
            branch['ready'][stage+1:] = [False] * (3-stage)
            for key in list(st.session_state):
                if any(key.startswith(f"group_{branch['id']}_{n}_") for n in range(stage+1, 4)):
                    del st.session_state[key]
            if branch.get('nodes', [None]*5)[stage] == item.get('nodes', [None]*5)[stage]:
                branch['ready'][stage:] = [False] * (4-stage)
        return
    config = deepcopy(item['config'])
    put(config, path, value)
    weight = config['hybrid_fusion']['weights']['vector_weight']
    config['hybrid_fusion']['weights']['bm25_weight'] = round(1-weight, 6)
    config['vector_search']['hnsw_params']['ef'] = max(config['vector_search']['top_k'], config['vector_search']['hnsw_params']['ef'])
    try:
        item['config'] = RAGConfig.model_validate(config).model_dump()
        item.pop('focus_stage', None)
        entry = ['retrieval_pipeline.mode', 'vector_search.model_name', 'vector_search.top_k', 'generation.llm_model'][stage]
        item['ready'][stage] = st.session_state.get(f'group_{identifier}_{stage}_{entry}') is not None
        item['ready'][stage+1:] = [False] * (3-stage)
        item.pop('error', None)
        shared = [p for p in st.session_state.grouped_paths if p['id'] != identifier and p.get('nodes', [None]*5)[stage] == item.get('nodes', [None]*5)[stage]]
        for descendant in shared:
            updated = deepcopy(descendant['config'])
            put(updated, path, value)
            updated['vector_search']['hnsw_params']['ef'] = max(updated['vector_search']['top_k'], updated['vector_search']['hnsw_params']['ef'])
            w = updated['hybrid_fusion']['weights']['vector_weight']
            updated['hybrid_fusion']['weights']['bm25_weight'] = round(1-w, 6)
            descendant['config'] = RAGConfig.model_validate(updated).model_dump()
            descendant['ready'][stage] = item['ready'][stage]
            descendant['ready'][stage+1:] = [False] * (3-stage)
            for stale in list(st.session_state):
                if stale.startswith('group_' + descendant['id'] + '_'):
                    del st.session_state[stale]
        # Discard downstream widget values as well as readiness, so stale selections cannot return.
        for key in list(st.session_state):
            if any(key.startswith(f'group_{identifier}_{n}_') for n in range(stage+1, 4)):
                del st.session_state[key]
        for branch in st.session_state.grouped_paths:
            branch['ready'][stage+1:] = [False] * (3-stage)
            for key in list(st.session_state):
                if any(key.startswith(f"group_{branch['id']}_{n}_") for n in range(stage+1, 4)):
                    del st.session_state[key]
        if path == 'vector_search.top_k':
            st.session_state.pop(f'group_{identifier}_1_vector_search.hnsw_params.ef', None)
    except ValueError as exc:
        item['error'] = str(exc)
        st.session_state[f'group_{identifier}_{stage}_{path}'] = get(item['config'], path)


def checkout(identifier, stage):
    if st.session_state.get('grouped_checkout_used') or st.session_state.grouped_counter > 1:
        return
    item = next(p for p in st.session_state.grouped_paths if p['id'] == identifier)
    value = fingerprint(item, stage)
    if not item['ready'][stage] or value in {item['source_cards'].get(stage), item['checked_cards'].get(stage)}:
        return
    item.pop('focus_stage', None)
    st.session_state.grouped_counter += 1
    copy = deepcopy(item)
    copy['id'] = str(st.session_state.grouped_counter)
    copy['parent'] = identifier
    copy['checkout_stage'] = stage
    copy['nodes'] = list(item['nodes'])
    copy['nodes'][stage] = copy['id'] + '_' + str(stage)
    copy['source_cards'] = {n: fingerprint(item, n) for n in range(4)}
    copy['checked_cards'] = {}
    st.session_state.grouped_paths.append(copy)
    st.session_state.grouped_checkout_used = True
    item['checked_cards'][stage] = value


def grouped_workspace(history):
    if not st.session_state.get('grouped_paths'):
        st.session_state.grouped_counter = 1
        st.session_state.grouped_paths = [{'id': '1', 'config': deepcopy(history[0]['config_snapshot']),
                                          'ready': [False]*4, 'source_cards': {}, 'checked_cards': {},
                                          'manifest': deepcopy(history[0]['knowledge_base']),
                                          'dataset': {'sha256': history[0]['dataset']['sha256']}}]
    completed = []
    st.markdown('''<style>
    div[class*="st-key-grouped_path_"] > [data-testid="stHorizontalBlock"] {overflow-x:auto;flex-wrap:nowrap!important;padding-bottom:6px;gap:.7rem;}
    div[class*="st-key-grouped_path_"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {min-width:245px;flex:1 0 245px!important;}
    div[class*="st-key-grouped_path_"] [data-testid="stVerticalBlock"] {gap:.15rem;}
    div[class*="st-key-grouped_path_"] [data-testid="stVerticalBlockBorderWrapper"] {padding:.4rem!important;border-radius:12px;}
    div[class*="st-key-grouped_path_"] [data-testid="stMarkdownContainer"] p {margin:0;font-size:.75rem;line-height:1.15;white-space:nowrap;}
    div[class*="st-key-grouped_path_"] [data-testid="stWidgetLabel"] p {font-size:.82rem;}
    /* Keep two fully expanded configuration cards within a desktop viewport. */
    div[class*="st-key-grouped_path_"] [data-testid="stHorizontalBlock"] {gap:.25rem;}
    div[class*="st-key-configuration_node_"] {padding:6px!important;}
    div[class*="st-key-grouped_path_"] [data-testid="stSelectbox"] [role="group"],
    div[class*="st-key-grouped_path_"] [data-testid="stNumberInputContainer"] {height:24px!important;min-height:24px!important;padding:0!important;}
    div[class*="st-key-grouped_path_"] [data-testid="stSelectbox"] input,
    div[class*="st-key-grouped_path_"] [data-testid="stNumberInput"] input {height:22px!important;min-height:22px!important;padding:0 .35rem!important;font-size:.75rem!important;}
    div[class*="st-key-grouped_path_"] [data-testid="stSelectbox"] button,
    div[class*="st-key-grouped_path_"] [data-testid="stNumberInput"] button {height:22px!important;min-height:22px!important;padding:0!important;width:22px!important;}
    div[class*="st-key-grouped_path_"] [data-testid="stButton"] button {min-height:24px!important;padding:.1rem .3rem!important;}
    div[class*="st-key-grouped_path_"] [data-testid="stCaptionContainer"] p {font-size:.7rem!important;white-space:normal;}
    div[class*="st-key-grouped_path_"] [data-testid="stToggle"] {min-height:28px;}
    </style>''', unsafe_allow_html=True)
    paths = st.session_state.grouped_paths[:2]
    st.session_state.grouped_paths = paths
    # Only the checked-out stage owns a second card, including in existing sessions.
    root = paths[0]
    root['nodes'] = [root['id'] + '_' + str(n) for n in range(5)]
    for item in paths[1:]:
        split = item['checkout_stage']
        item['nodes'] = list(root['nodes'])
        item['nodes'][split] = item['id'] + '_' + str(split)
        for stage in range(4):
            if stage == split:
                continue
            item['ready'][stage] = root['ready'][stage]
            if stage == 0:
                item['config']['retrieval_pipeline'] = deepcopy(root['config']['retrieval_pipeline'])
            elif stage == 1:
                for field, value in root['config']['vector_search'].items():
                    if field != 'top_k':
                        item['config']['vector_search'][field] = deepcopy(value)
            elif stage == 2:
                item['config']['vector_search']['top_k'] = root['config']['vector_search']['top_k']
                for field in ('bm25_search', 'hybrid_fusion', 'rerank'):
                    # Enablement belongs to the retrieval mode, not the shared search card.
                    enabled = item['config'][field].get('enabled')
                    item['config'][field] = deepcopy(root['config'][field])
                    if field in {'bm25_search', 'rerank'}:
                        item['config'][field]['enabled'] = enabled
            else:
                for field in ('generation', 'evaluation'):
                    item['config'][field] = deepcopy(root['config'][field])
    # A shared HNSW setting must accommodate both Top-K choices.
    if len(paths) == 2 and paths[1]['checkout_stage'] != 1:
        ef = max(p['config']['vector_search']['hnsw_params']['ef'] for p in paths)
        ef = max(ef, *(p['config']['vector_search']['top_k'] for p in paths))
        for item in paths:
            item['config']['vector_search']['hnsw_params']['ef'] = ef
    nodes = [{} for _ in range(5)]
    for item in paths:
        end = next((i for i, ready in enumerate(item['ready']) if not ready), 4)
        end = min(end, item.get('focus_stage', 4))
        for stage in range(end + 1):
            nodes[stage].setdefault(item['nodes'][stage], item)
    if len(paths) == 2:
        split = paths[1]['checkout_stage']
        selectors = ', '.join('.st-key-configuration_node_' + p['nodes'][split] for p in paths)
        st.markdown(f'<style>{selectors} {{background:#f0f4f8;border-radius:10px;}}</style>', unsafe_allow_html=True)
    with st.container(key='grouped_path_shared'):
        columns = st.columns(5, gap='small')
        for stage, column in enumerate(columns):
            with column:
                for node, item in nodes[stage].items():
                    identifier = item['id']
                    with st.container(border=True, key='configuration_node_' + node):
                        if stage < 4:
                            st.button(f'{stage+1:02d} · {TITLES[stage]}', key='select_card_' + node,
                                      type='tertiary', on_click=select_card, args=(identifier, stage))
                        else:
                            st.markdown(f'**{stage+1:02d} · {TITLES[stage]}**')
                        config = item['config']
                        preceding = []
                        def widget(path, label, options=None, first=False, **kwargs):
                            key = f'group_{identifier}_{stage}_{path}'
                            if key not in st.session_state:
                                st.session_state[key] = None if first and not item['ready'][stage] else get(config, path)
                            args = dict(key=key, on_change=edit, args=(identifier, stage, path), label_visibility='collapsed')
                            if st.session_state.get('only_existing_tests'):
                                choices = existing_choices(history, paths, identifier, stage, preceding, path)
                                preceding.append(path)
                                options = [value for value, _ in choices]
                                kwargs = {k: v for k, v in kwargs.items() if k == 'format_func'}
                                args.update(on_change=edit_existing, args=(identifier, stage, path, choices))
                                if not options:
                                    st.caption(label + ': No compatible saved test. Reset or change an earlier card.')
                                    return None
                                if st.session_state.get(key) is not None and st.session_state[key] not in options:
                                    st.session_state[key] = options[0]

                            name, value = st.columns([.85, 1.3], gap='small', vertical_alignment='center')
                            with name:
                                st.markdown(label)
                            with value:
                                if options:
                                    return st.selectbox(label, options, index=None, placeholder='Select…', **args, **kwargs)
                                if isinstance(get(config, path), bool):
                                    return st.toggle(label, **args, **kwargs)
                                return st.number_input(label, **args, **kwargs)
                        if stage == 0:
                            widget('retrieval_pipeline.mode', 'Retrieval model', list(MODES), first=True, format_func=MODES.get)
                        elif stage == 1:
                            widget('vector_search.model_name', 'Embedding model', ['BAAI/bge-m3', 'BAAI/bge-large-en-v1.5', 'text-embedding-3-small'], first=True)
                            widget('vector_search.index_type', 'Index type', ['HNSW', 'IVF_FLAT', 'IVF_SQ8'])
                            st.caption('Index settings')
                            widget('vector_search.metric_type', 'Metric', ['COSINE', 'L2', 'IP'])
                            if config['vector_search']['index_type'] == 'HNSW':
                                widget('vector_search.hnsw_params.ef', 'HNSW ef', min_value=max(p['config']['vector_search']['top_k'] for p in paths))
                                widget('vector_search.hnsw_params.M', 'HNSW M', min_value=1)
                                widget('vector_search.hnsw_params.efConstruction', 'efConstruction', min_value=1)
                            else:
                                widget('vector_search.ivf_params.nlist', 'IVF nlist', min_value=1)
                                widget('vector_search.ivf_params.nprobe', 'IVF nprobe', min_value=1)
                            widget('vector_search.quantization.enabled', 'Enable quantization')
                            widget('vector_search.quantization.type', 'Quantization', ['none', 'SQ8', 'BINARY'])
                        elif stage == 2:
                            options = list(dict.fromkeys([20, 50, 100, config['vector_search']['top_k']]))
                            widget('vector_search.top_k', 'Vector Top-K', options, first=True)
                            if any(p['config']['retrieval_pipeline']['mode'] != 'vector_only' for p in paths):
                                widget('bm25_search.model_name', 'BM25 engine', ['BM25Okapi', 'Milvus Built-in Sparse BM25'])
                                widget('bm25_search.top_k', 'BM25 Top-K', min_value=1, max_value=1000)
                                widget('hybrid_fusion.strategy', 'Fusion', ['weighted_sum', 'RRF'])
                                if config['hybrid_fusion']['strategy'] == 'weighted_sum':
                                    widget('hybrid_fusion.weights.vector_weight', 'Vector weight', min_value=0.0, max_value=1.0, step=0.05)
                                else:
                                    widget('hybrid_fusion.rrf_k', 'RRF k', min_value=1)
                            if any(p['config']['retrieval_pipeline']['mode'] == 'hybrid_rerank' for p in paths):
                                widget('rerank.model_name', 'Reranker', ['bge-reranker-base', 'Cohere Rerank v3'])
                                widget('rerank.top_n', 'Rerank Top-N', min_value=1, max_value=2000)
                        elif stage == 3:
                            widget('generation.llm_model', 'Answer model', ['llama3.1:8b', 'gpt-4o-mini'], first=True)
                            widget('generation.context_top_k', 'Context chunks', min_value=1, max_value=50)
                            widget('evaluation.judge_model', 'RAGAS judge', ['llama3.1:8b', 'gpt-4o-mini'])
                            st.caption('Generation & scoring settings')
                            widget('generation.temperature', 'Temperature', min_value=0.0, max_value=2.0, step=0.1)
                            widget('generation.max_tokens', 'Answer max tokens', min_value=1, max_value=16384)
                            widget('generation.max_context_chars', 'Context character limit', min_value=1000, max_value=100000)
                            widget('generation.seed', 'Seed', min_value=0)
                            widget('evaluation.judge_max_tokens', 'Judge max tokens', min_value=512, max_value=16384)
                        else:
                            for result_path in paths:
                                if not all(result_path['ready']) or result_path.get('focus_stage', 4) < 4:
                                    continue
                                wanted = signature(result_path['config'], 'evaluation', 'project', result_path['manifest'], result_path['dataset'])
                                if any(result['signature'] == wanted for result in completed):
                                    continue
                                hits, _ = find_runs(wanted, 'evaluation')
                                label = comparison_title(result_path, paths)
                                st.success(f'{label}: {len(hits)} exact historical matches') if hits else st.info(f'{label}: No exact historical match')
                                completed.append({'title': label, 'id': result_path['id'], 'signature': wanted, 'config': deepcopy(result_path['config']), 'kind': 'evaluation', 'source': 'project', 'manifest': result_path['manifest'], 'dataset': result_path['dataset'], 'query': None, 'queries': []})
                            st.caption('Results are compared below. A missing result can run only when explicitly requested.')
                        if stage < 4 and not st.session_state.get('grouped_checkout_used', False) and st.session_state.grouped_counter == 1:
                            value = fingerprint(item, stage)
                            duplicate = value in {item['source_cards'].get(stage), item['checked_cards'].get(stage)}
                            st.button('Check out', key=f'checkout_{identifier}_{stage}', disabled=not item['ready'][stage] or duplicate,
                                      on_click=checkout, args=(identifier, stage))
                        if item.get('error'):
                            st.error(item['error'])
    return completed

