"""Streamlit workspace backed by the local RAG API."""

from copy import deepcopy
from pathlib import Path
import os
import httpx
import json

import streamlit as st
import yaml

from core.config_loader import RAGConfig, load_config
from core.secrets import load_keys, save_key

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

def api(method, path, **kwargs):
    response = httpx.request(method, API_URL + path, timeout=30, **kwargs)
    if response.is_error:
        detail = response.json().get("detail", "Request failed")
        raise ValueError(str(detail))
    return response.json()

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
MODE_LABELS = {
    "vector_only": "Vector only",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + rerank",
}

st.set_page_config(page_title="RAG Ablation Studio", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}
[data-testid="stSidebar"] {background: #fff; border-right: 1px solid #e8eaf0;}
[data-testid="stSidebar"] .block-container {padding-top: 1.6rem;}
h1 {font-size: 2.35rem !important; letter-spacing: -.075rem; font-weight: 650 !important;}
h2 {font-size: 1.4rem !important; letter-spacing: -.035rem;}
h3 {font-size: 1.08rem !important;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border-radius: 16px;}
[data-testid="stMetricValue"] {font-size: 1.9rem; letter-spacing: -.045rem;}
[data-testid="stTabs"] [role="tablist"] {gap: 2rem; margin-bottom: 1rem;}
[data-testid="stTabs"] [role="tab"] {padding-bottom: .85rem; font-weight: 550;}
.eyebrow {color: #79818e; font-size: .7rem; letter-spacing: .14em; font-weight: 650; margin-bottom: .4rem;}
.muted {color: #79818e; font-size: .9rem;}
.status {display: inline-block; background: #eef4ff; color: #3561a0; font-size: .75rem;
    border: 1px solid #dce8fb; border-radius: 100px; padding: .35rem .75rem;}
.step {color: #79818e; font-size: .7rem; font-weight: 650; letter-spacing: .12em; margin-top: 1.1rem; margin-bottom: .55rem;}
.empty {text-align: center; padding: 2rem 1rem; color: #7e8693;}
.empty strong {display: block; color: #353b46; font-size: 1rem; margin-bottom: .5rem;}
button[kind="primary"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Detach reusable widget values from page cleanup when navigating to results.
for state_key in list(st.session_state):
    if state_key.startswith(("control_", "paid_")) or state_key in {"openai_key", "cohere_key", "query", "data_source"}:
        st.session_state[state_key] = st.session_state[state_key]
st.session_state.setdefault("live_page", "configuration")

if "config_revision" not in st.session_state:
    try:
        initial = api("GET", "/config")
        st.session_state.config = initial["config"]
        st.session_state.config_revision = initial["revision"]
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"Start the local API before opening this workspace: {exc}")
        st.stop()


try:
    saved_keys = load_keys()
except (OSError, ValueError):
    st.error("Unable to read the local API key file. Check .secrets/api_keys.json.")
    st.stop()
for provider in ("openai", "cohere"):
    st.session_state.setdefault(provider + "_key", saved_keys.get(provider, ""))


def save_api_key(provider):
    try:
        save_key(provider, st.session_state[provider + "_key"])
        st.session_state.key_save_error = None
    except (OSError, ValueError):
        st.session_state.key_save_error = "Unable to save the local API key file. The change was not saved."
        st.session_state[provider + "_key"] = saved_keys.get(provider, "")


def credentials():
    return {"openai_api_key": st.session_state.get("openai_key", ""),
            "cohere_api_key": st.session_state.get("cohere_key", "")}


def selected_source():
    return st.session_state.get("upload_source", "project") if st.session_state.get("data_source") == "Upload documents" else "project"


def request_body(name=None, source=None):
    config = deepcopy(st.session_state.config)
    if name is not None:
        config["experiment"]["name"] = name
    return {"config_snapshot": config, "source_id": source or selected_source(), "credentials": credentials()}


def start_job(endpoint, payload):
    try:
        job = api("POST", endpoint, json=payload)
        st.session_state.setdefault("active_jobs", {})[job["job_id"]] = job
    except (httpx.HTTPError, ValueError) as exc:
        st.error(str(exc))


@st.fragment(run_every=2)
def job_progress():
    for identifier in list(st.session_state.get("active_jobs", {})):
        try:
            job = api("GET", "/jobs/" + identifier)
            st.session_state.active_jobs[identifier] = job
            label = f"{job['kind']}: {job['status']} — {job['message']}"
            if job["status"] == "SUCCESS":
                st.success(label)
            elif job["status"] in {"FAILED", "INTERRUPTED"}:
                st.error(label)
            else:
                st.info(label)
        except (httpx.HTTPError, ValueError) as exc:
            st.warning(str(exc))
    if st.session_state.get("active_jobs") and st.button("Dismiss completed jobs"):
        st.session_state.active_jobs = {k: v for k, v in st.session_state.active_jobs.items() if v["status"] in {"PENDING", "RUNNING"}}
        st.rerun()


def nested_get(config, path):
    for part in path.split("."):
        config = config[part]
    return config


def save_control(key, path):
    """Persist a validated control change atomically; credentials use a separate local file."""
    config = deepcopy(st.session_state.config)
    parts = path.split(".")
    section = config
    for part in parts[:-1]:
        section = section[part]
    section[parts[-1]] = st.session_state[key]
    mode = config["retrieval_pipeline"]["mode"]
    config["bm25_search"]["enabled"] = mode != "vector_only"
    config["rerank"]["enabled"] = mode == "hybrid_rerank"
    weight = config["hybrid_fusion"]["weights"]["vector_weight"]
    config["hybrid_fusion"]["weights"]["bm25_weight"] = round(1 - weight, 2)
    config["generation"]["provider"] = (
        "openai" if config["generation"]["llm_model"] == "gpt-4o-mini" else "ollama"
    )
    vector = config["vector_search"]
    if not 0 <= weight <= 1 or not 1 <= vector["top_k"] <= 100:
        st.session_state.save_error = "Invalid retrieval configuration. Changes were not saved."
        return
    # Keep HNSW's search candidate count at least as large as the requested Top-K.
    vector["hnsw_params"]["ef"] = max(vector["hnsw_params"]["ef"], vector["top_k"])
    if path in {"vector_search.top_k", "vector_search.hnsw_params.ef"}:
        st.session_state["control_vector_search.hnsw_params.ef"] = vector["hnsw_params"]["ef"]
    try:
        validated = RAGConfig.model_validate(config).model_dump()
        result = api("PUT", "/config", json={"config": validated, "expected_revision": st.session_state.config_revision})
        st.session_state.config = result["config"]
        st.session_state.config_revision = result["revision"]
        st.session_state.save_error = None
    except (ValueError, httpx.HTTPError) as exc:
        st.session_state.save_error = str(exc)
        st.session_state[key] = nested_get(st.session_state.config, path)


def control(widget, label, path, **kwargs):
    key = f"control_{path}"
    st.session_state.setdefault(key, nested_get(st.session_state.config, path))
    if kwargs.get("disabled"):
        st.session_state[key] = nested_get(st.session_state.config, path)
    return widget(label, key=key, on_change=save_control, args=(key, path), **kwargs)


def model_control(label, path, local_models, paid_model, credential):
    """Expose a disabled paid-model switch until the matching credential is present."""
    paid_key = f"paid_{path}"
    authorized = bool(st.session_state.get(credential, "").strip() or os.getenv({"openai_key": "OPENAI_API_KEY", "cohere_key": "COHERE_API_KEY"}[credential], "").strip())
    if not authorized:
        st.session_state[paid_key] = False
        if nested_get(st.session_state.config, path) == paid_model:
            key = f"control_{path}"
            st.session_state[key] = local_models[0]
            save_control(key, path)
    use_paid = st.checkbox(f"Use {paid_model}", key=paid_key, disabled=not authorized,
                           help="Enter the provider API key below to enable this option.")
    options = local_models + ([paid_model] if use_paid and authorized else [])
    key = f"control_{path}"
    if st.session_state.get(key, nested_get(st.session_state.config, path)) not in options:
        st.session_state[key] = options[0]
        save_control(key, path)
    return control(st.selectbox, label, path, options=options)


def empty(title, message):
    # Only constant developer-authored strings are passed into this HTML helper.
    st.markdown(f'<div class="empty"><strong>{title}</strong>{message}</div>', unsafe_allow_html=True)


def show_evidence(run):
    with st.container(border=True):
        st.subheader("Output Evidence")
        chunks = run["results"].get("evidence_chunks", []) if run else []
        if not chunks:
            empty("No evidence yet", "Inspect source text, metadata and retrieval scores here.")
        for index, chunk in enumerate(chunks, 1):
            score = chunk.get("score")
            score_label = f" · Score {score:.2f}" if isinstance(score, (int, float)) else ""
            with st.expander(f"[{index}] {chunk.get('section_title', 'Untitled section')}{score_label}", expanded=index == 1):
                st.caption(f"Document: {chunk.get('doc_id', '—')} · Chunk: {chunk.get('chunk_id', '—')}")
                metadata = chunk.get("metadata", {})
                if metadata:
                    st.caption("  ·  ".join(f"{key}: {value}" for key, value in metadata.items()))
                st.markdown(chunk.get("text", ""))


# Enumerated controls use explicit choices; numeric fields remain directly editable.
ENUMS = {
    "retrieval_pipeline.mode": list(MODE_LABELS),
    "vector_search.index_type": ["HNSW", "IVF_FLAT", "IVF_SQ8"],
    "vector_search.metric_type": ["COSINE", "L2", "IP"],
    "vector_search.quantization.type": ["none", "SQ8", "BINARY"],
    "bm25_search.tokenizer": ["standard"],
    "hybrid_fusion.strategy": ["weighted_sum", "RRF"],
    "ingestion_pipeline.chunking.strategy": ["fixed_size", "semantic", "recursive"],
    "ingestion_pipeline.chunking.semantic_config.breakpoint_threshold_type": ["percentile", "standard_deviation", "interquartile"],
    "load_testing.tool": ["locust"],
    "load_testing.endpoint": ["/retrieve", "/query"],
}
LIST_OPTIONS = {
    "ingestion_pipeline.cleaning.language_filter": ["en"],
    "ingestion_pipeline.metadata_enrichment.entity_types": ["API", "VERSION", "CODE"],
    "evaluation.metrics": ["context_recall", "context_precision", "faithfulness", "answer_correctness"],
}
MODEL_PATHS = {"vector_search.model_name", "bm25_search.model_name", "rerank.model_name", "generation.llm_model", "generation.provider", "evaluation.judge_model"}


def parameter_fields(mapping, prefix):
    for field, value in mapping.items():
        path = f"{prefix}.{field}"
        if path in MODEL_PATHS:
            continue
        label = field.replace("_", " ").replace("efConstruction", "ef construction").capitalize()
        if isinstance(value, dict):
            st.markdown(f"**{label}**")
            parameter_fields(value, path)
        elif path in ENUMS:
            control(st.selectbox, label, path, options=ENUMS[path])
        elif isinstance(value, bool):
            derived = path in {"bm25_search.enabled", "rerank.enabled", "vector_search.enabled"}
            control(st.toggle, label, path, disabled=derived,
                    help="Determined by the selected retrieval mode." if derived else None)
        elif isinstance(value, (int, float)):
            is_float = isinstance(value, float)
            kwargs = {"min_value": 0.0 if is_float else 0, "step": 0.05 if is_float else 1}
            if field in {"top_k", "top_n", "batch_size", "chunk_size", "max_tokens", "min_chunk_length", "max_chunk_length", "ef", "efConstruction"}:
                kwargs["min_value"] = 1
            if field == "top_k":
                kwargs["max_value"] = 100
            if "weight" in field or field == "min_context_recall":
                kwargs["max_value"] = 1.0
            control(st.number_input, label, path, disabled=path == "hybrid_fusion.weights.bm25_weight", **kwargs)
        elif isinstance(value, list):
            options = list(dict.fromkeys([*LIST_OPTIONS.get(path, []), *value]))
            control(st.multiselect, label, path, options=options)
        else:
            control(st.text_input, label, path)


def show_parameters():
    with st.container(height=420, border=True):
        st.subheader("Input parameters")
        st.caption("Choose options, enter numeric values, and toggle pipeline features.")
        mode = control(st.selectbox, "Retrieval mode", "retrieval_pipeline.mode",
                       options=list(MODE_LABELS), format_func=MODE_LABELS.get)
        columns = st.columns(3)
        with columns[0]:
            control(st.number_input, "Vector Top-K", "vector_search.top_k", min_value=1, max_value=100)
        with columns[1]:
            control(st.number_input, "BM25 Top-K", "bm25_search.top_k", min_value=1, max_value=100, disabled=mode == "vector_only")
        with columns[2]:
            control(st.number_input, "Rerank Top-N", "rerank.top_n", min_value=1, max_value=200, disabled=mode != "hybrid_rerank")
        with st.expander("Models & API credentials"):
            st.text_input("OpenAI API key", type="password", key="openai_key", on_change=save_api_key, args=("openai",))
            st.text_input("Cohere API key", type="password", key="cohere_key", on_change=save_api_key, args=("cohere",))
            st.caption("Keys are saved in .secrets/api_keys.json (Git-ignored) and excluded from configuration and snapshots.")
            if st.session_state.get("key_save_error"):
                st.error(st.session_state.key_save_error)
            model_control("Embedding model", "vector_search.model_name", ["BAAI/bge-large-en-v1.5", "BAAI/bge-m3"], "text-embedding-3-small", "openai_key")
            control(st.selectbox, "Keyword search model", "bm25_search.model_name", options=["BM25Okapi", "Milvus Built-in Sparse BM25"])
            model_control("Reranker", "rerank.model_name", ["bge-reranker-base"], "Cohere Rerank v3", "cohere_key")
            model_control("Answer model", "generation.llm_model", ["llama3.1:8b"], "gpt-4o-mini", "openai_key")
        sections = [
            ("Retrieval, fusion & index settings", ["vector_search", "bm25_search", "hybrid_fusion", "rerank"]),
            ("Ingestion & metadata", ["ingestion_pipeline"]),
            ("Generation", ["generation"]),
        ]
        for title, keys in sections:
            with st.expander(title):
                for section in keys:
                    mapping = deepcopy(st.session_state.config[section])
                    if section in {"vector_search", "bm25_search"}:
                        mapping.pop("top_k", None)
                    if section == "rerank":
                        mapping.pop("top_n", None)
                    st.markdown(f"**{section.replace('_', ' ').title()}**")
                    parameter_fields(mapping, section)
        st.caption("Index and embedding changes require rebuilding a compatible index. Build the index after changing ingestion, embedding or index settings.")


def open_result():
    payload = request_body()
    payload["query"] = st.session_state.get("query", "")
    st.session_state.pending_run = payload
    st.session_state.live_page = "results"
    st.session_state.pop("live_result", None)


def configuration_page():
    with st.container(border=True):
        st.subheader("Live Run Workspace")
        st.caption("Configure an experiment for your documentation knowledge base.")
        control(st.text_input, "Experiment name", "experiment.name", placeholder="e.g. hybrid_rerank_v1")
        control(st.text_area, "Experiment description (optional)", "experiment.description", height=80)
        show_parameters()
        source = st.selectbox("Data source", ["Project default", "Upload documents"], key="data_source")
        if source == "Upload documents":
            files = st.file_uploader("Upload documents", type=["pdf", "md", "txt", "docx", "html"], accept_multiple_files=True)
            if st.button("Save uploaded documents", disabled=not files):
                try:
                    result = api("POST", "/documents", files=[("files", (file.name, file.getvalue(), file.type)) for file in files])
                    st.session_state.upload_source = result["source_id"]
                    st.success("Saved: " + ", ".join(result["documents"]))
                except (ValueError, httpx.HTTPError) as exc:
                    st.error(str(exc))
            st.caption("Source: " + st.session_state.get("upload_source", "Save documents first"))
        else:
            st.caption("Place project documentation in data/documents, then build its index.")
        if st.button("Build / update document index", disabled=source == "Upload documents" and not st.session_state.get("upload_source")):
            start_job("/ingest", request_body())
        st.text_area("Question", key="query", height=95, placeholder="Enter a question about the selected data source.")
        st.button("Run", type="primary", width="stretch", on_click=open_result,
                  disabled=not (st.session_state.config["experiment"]["name"].strip() and st.session_state.get("query", "").strip()) or bool(st.session_state.get("save_error")))
        if st.session_state.get("save_error"):
            st.error(st.session_state.save_error)
        else:
            st.caption("● Parameter changes validated and saved to config.yaml")


def json_pair(run, live=False, prefix="live"):
    output = deepcopy(run)
    if live:
        for key in ["algorithm_metrics", "load_test_metrics", "sla_compliance"]:
            output["results"].pop(key, None)
    input_record = {key: value for key, value in output.items() if key != "results"}
    for column, title, payload, suffix in zip(st.columns(2), ["Input Config JSON", "Output JSON"], [input_record, output], ["input", "output"]):
        with column.container(border=True):
            st.subheader(title)
            text = json.dumps(payload, indent=2, ensure_ascii=False)
            st.code(text, language="json", height=350)
            st.download_button("Download " + title, text, f"{run['run_id']}_{suffix}.json", "application/json", key=f"download_{run['run_id']}_{suffix}_{live}_{prefix}")


def results_page():
    st.button("← Back to configuration", on_click=lambda: st.session_state.update(live_page="configuration"))
    if "pending_run" in st.session_state:
        payload = st.session_state.pop("pending_run")
        st.subheader(payload["config_snapshot"]["experiment"]["name"])
        st.markdown(payload["query"])
        status, answer_box = st.empty(), st.empty()
        answer = ""
        try:
            with httpx.stream("POST", API_URL + "/query/stream", json=payload, timeout=httpx.Timeout(600, connect=10)) as response:
                if response.is_error:
                    raise ValueError(json.loads(response.read()).get("detail", "Run failed"))
                for line in response.iter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if item["type"] == "status":
                        status.info(item["message"])
                    elif item["type"] == "token":
                        answer += item["text"]
                        answer_box.markdown(answer)
                    elif item["type"] == "result":
                        st.session_state.live_result = item["output"]
            if "live_result" not in st.session_state:
                raise ValueError("The response stream ended before the run completed.")
            st.rerun()
        except (httpx.HTTPError, ValueError) as exc:
            st.error(str(exc))
            return
    run = st.session_state.get("live_result")
    if not run:
        st.info("Return to configuration to start a run.")
        return
    with st.container(border=True):
        st.caption("EXPERIMENT NAME")
        st.subheader(run["experiment"]["name"])
        st.caption(run["experiment"].get("description", ""))
    if run["results"].get("error"):
        st.error(run["results"]["error"])
    with st.container(border=True):
        st.subheader("Question")
        st.markdown(run["query"])
        st.divider()
        st.subheader("RAG Answer")
        st.markdown(run["results"].get("answer") or "No answer was generated.")
    show_evidence(run)
    json_pair(run, live=True)


def saved_history(kind):
    try:
        response = api("GET", "/history", params={"kind": kind})
        for message in response["errors"]:
            st.warning(message)
        return response["records"]
    except (ValueError, httpx.HTTPError) as exc:
        st.warning(str(exc))
        return []


def compare_history(kind):
    runs = saved_history(kind)
    if not runs:
        st.info("No saved runs yet. Completed runs will appear here.")
        return
    selected = []
    for i, column in enumerate(st.columns(2)):
        with column:
            chosen = st.selectbox("Saved run " + "AB"[i], range(len(runs)), index=min(i, len(runs)-1),
                                  key=f"select_{kind}_{i}", format_func=lambda n: runs[n]["experiment"]["name"] + " · " + runs[n]["run_id"])
            selected.append(runs[chosen])
    section = "algorithm_metrics" if kind == "evaluation" else "load_test_metrics"
    fields = list(dict.fromkeys(key for run in selected for key in (run["results"].get(section) or {})))
    st.dataframe([{"Metric": field, "A": (selected[0]["results"].get(section) or {}).get(field),
                   "B": (selected[1]["results"].get(section) or {}).get(field)} for field in fields], hide_index=True, width="stretch")
    for i, run in enumerate(selected):
        with st.expander("Run " + "AB"[i] + " details", expanded=i == 0):
            st.write("Status: " + run["results"]["status"])
            if run["results"].get("error"):
                st.error(run["results"]["error"])
            if kind == "evaluation":
                cases = run["results"].get("per_case", [])
                if cases:
                    case = st.selectbox("Test case", cases, format_func=lambda row: row.get("query", str(row.get("case_id", ""))), key=f"case_{i}_{run['run_id']}")
                    st.json(case)
            else:
                st.json(run["results"].get("sla_compliance", {}))
            json_pair(run, prefix=str(i))


st.markdown('<div class="eyebrow">OPEN-SOURCE TECH & API DOCUMENTATION</div>', unsafe_allow_html=True)
st.title("RAG Ablation Studio")
job_progress()
live_tab, compare_tab = st.tabs(["Live Run Workspace", "Tuning & Ablation Matrix"], key="workspace_tab", on_change="rerun")
with live_tab:
    if st.session_state.live_page == "results":
        results_page()
    else:
        configuration_page()
with compare_tab:
    sources = ["project"] + ([st.session_state.upload_source] if st.session_state.get("upload_source") else [])
    tuning_source = st.selectbox("Testing data source", sources, key="testing_source")
    if st.button("Build / update testing index"):
        start_job("/ingest", request_body(source=tuning_source))
    evaluation_tab, load_tab = st.tabs(["RAGAS Evaluation", "Load Test"])
    with evaluation_tab:
        st.subheader("Tuning & Ablation Matrix")
        st.caption("Evaluate your golden test set against vector, hybrid and reranked retrieval. Reference answers are required.")
        name = st.text_input("Evaluation experiment name", key="evaluation_name")
        with st.expander("Evaluation settings"):
            model_control("Evaluation judge", "evaluation.judge_model", ["llama3.1:8b"], "gpt-4o-mini", "openai_key")
            parameter_fields(st.session_state.config["evaluation"], "evaluation")
        modes = st.multiselect("Retrieval modes", list(MODE_LABELS), default=list(MODE_LABELS), format_func=MODE_LABELS.get)
        st.caption("Evaluations use the testing data source and run independently.")
        if st.button("Run evaluation", type="primary", disabled=not name.strip() or not modes):
            start_job("/evaluate", {**request_body(name, tuning_source), "modes": modes})
        st.button("Refresh evaluation history")
        compare_history("evaluation")
    with load_tab:
        st.subheader("Independent Load Test")
        name = st.text_input("Load-test experiment name", key="load_name")
        with st.expander("Load-test input configuration", expanded=True):
            parameter_fields(st.session_state.config["load_testing"], "load_testing")
        queries = st.text_area("Load-test queries (one per line)", key="load_queries", placeholder="Provide representative questions for the load test.")
        if st.button("Run load test", type="primary", disabled=not name.strip() or not queries.strip()):
            start_job("/load-tests", {**request_body(name, tuning_source), "queries": [q.strip() for q in queries.splitlines() if q.strip()]})
        st.button("Refresh load-test history")
        compare_history("load_test")
