# RAG Ablation Study & Tuning Studio

Local Streamlit + FastAPI studio for technical and API documentation. Milvus provides dense and native BM25 indexing; Ollama supplies the default answer and evaluation model.

## Start

Python 3.12, Docker Desktop and Ollama are required. In this directory:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # only for a new setup; preserve existing settings
ollama pull llama3.1:8b
docker compose up -d
.venv/bin/python prepare_models.py
```

Run each service in its own terminal:

```sh
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

```sh
.venv/bin/python -m streamlit run app.py --server.port 8501
```

Open http://localhost:8501. API documentation: http://127.0.0.1:8000/docs. Services bind locally. This is a local studio; multi-user authentication and document permissions are deferred. Do not expose it as an authenticated enterprise service.

OpenKB-RAG Studio has separate Live Run and Tuning & Ablation pages with no sidebar. Tuning fixes RAGAS evaluation, 60 golden questions and project source. The progressive horizontal path has five grouped cards: Retrieval, Embedding & Index, Search & Rerank, Answer & Evaluation, Results. The first selection in each group opens the next group; changing a group resets downstream groups. Search controls depend on the retrieval mode. Draft branch configuration stays in session state and does not modify config.yaml or run evaluation automatically. Exact paired YAML lookup happens only at Results and includes configuration, dataset hash and index identity. All matching historical runs can be compared, including clearly marked partial results. No match requires an explicit Run action.

## Documents and Live Run

Live Run retains its original input layout and streamed answer/evidence/input-output JSON result page, without RAGAS scores. Each of the four tuning configuration cards has Check out: the original path remains and an editable duplicate is created at that group, inheriting the earlier groups as shared nodes. Only the checked-out configuration card is duplicated. Earlier and later configuration cards stay shared; their controls apply to both comparison paths. The Results column also has one shared card summarizing both paths; the detailed results compare below. Identical execution signatures appear once in both the Results summary and Matching results. Across all five columns, only the checked-out column can contain two cards. Editing either copy resets the shared continuation. Editing a shared ancestor updates all descendant paths and resets their later groups. Check out is allowed once for the current comparison: after creating one editable copy, every Check out button is hidden. Clicking a configuration card title keeps that path through the selected card, clears all later cards and other branches, and restores the one-checkout allowance. Editing or checking out the selected card continues the flow. Downstream groups follow the shared progressive selections. Completed paths compare results side by side. Branches are kept in the Streamlit session; experiment records are durable on disk. P99 appears only when independently measured.

Place original UTF-8 Markdown/text/HTML, text-based PDFs or DOCX documents in `data/documents`, or upload a separate document set through the UI. Upload limits are 20 MB per file, 200 MB per set, 100 files. Scanned PDFs need external OCR.

Click **Build / update document index**, wait for its job to finish, then enter an experiment name and question and click **Run**. The independent result page streams the answer and shows original evidence plus input/output JSON. Live Run never invokes RAGAS or Locust. Returning to configuration preserves inputs.

The pipeline cleans prose, filters configured languages, optionally deduplicates chunks using MinHash, preserves fenced code, chunks by fixed size, recursive boundaries or semantic distance, enriches metadata and embeds the resulting search text. Evidence retains the original cleaned chunk, excluding generated hypothetical questions and metadata prefixes. Oversized code remains atomic; chunks exceeding Milvus's UTF-8 field capacity produce a clear error rather than silent truncation. Context packing skips chunks that exceed the remaining context budget.

Vector-only, hybrid and hybrid-with-reranking share the same index. Weighted normalized fusion or RRF merges dense and BM25 results. BM25Okapi retains technical identifiers; Milvus native sparse BM25 is available for larger corpora. BGE reranking uses an ONNX backend by default, with a 512-token per-pair limit. Local embeddings run on CPU; the ONNX runtime selects available execution providers.

## Configuration and snapshots

Every control change is validated by Pydantic and saved atomically to `config.yaml` using revision checks. Invalid changes are rejected. A conflicting session must reload before saving. API keys entered in the UI are saved separately in `.secrets/api_keys.json`, with file permissions `0600` and directory permissions `0700`. The entire `.secrets/` directory is Git-ignored, including temporary files; keys are excluded from config, JSON downloads and YAML run snapshots. Clearing a key field removes its saved key. Environment keys remain an optional fallback. Paid OpenAI/Cohere models require their keys; enabling them sends relevant documents/questions to those providers.

Ingestion, embedding or index-build changes require rebuilding a compatible index. Rebuilds publish a new collection only after successful ingestion; existing collections remain available. Previous collections are retained and can be removed manually in Milvus when no longer needed. Source files are checked during ingestion; edits require clicking build/update again. Quantized HNSW uses `HNSW_SQ` with SQ8; IVF_SQ8 is supported. Binary quantization is rejected for the configured float embeddings.

Each run saves immutable input configuration and an output record with explicit status, evidence and real measured latency. JSON is the display/download format; persistence uses YAML. Locations:

- `data/experiments_history/*_input.yaml` and `*_output.yaml`: Live Run and evaluation records, filtered separately in the UI.
- `data/load_test_history/`: independent load-test snapshots, output YAML, Locust CSVs/logs.
- `data/jobs/`: durable background job status; active jobs become interrupted after an API restart.
- `data/collections/`: index manifests; `data/models/`: downloaded local model cache.

No demonstration corpus, fabricated metrics or synthetic experiment history is created.

## RAGAS tuning

Choose the testing source independently from Live Run. Provide your golden dataset at `data/golden_test_dataset.json` (or another path within `data/`) as a JSON list. Each row must have a unique string `id`, a `category` of `semantic`, `keyword` or `distractor`, a nonblank `question`, and a nonblank `reference_answer`. `reference` and `ground_truth` are accepted aliases for the answer.

Select retrieval modes and run evaluation. The batch answers every case and calculates context recall, context precision, faithfulness and answer correctness with RAGAS 0.3.9. The default judge is Ollama `llama3.1:8b`; a configured GPT judge requires an OpenAI key. Snapshot records include the dataset hash and exact cases, model versions and index manifest. Missing judge scores stay null; partial evaluations report valid-case counts rather than fabricated zeros. Comparison reads saved outputs and never recomputes scores.

CLI alternative:

```sh
.venv/bin/python evaluate_ragas.py --name baseline --modes vector_only hybrid hybrid_rerank
```

Local judge scoring can take substantial time and is model-dependent. Gold references are necessary for this evaluation; Live Run requires none.

Evaluation now saves each answer immediately, then saves each individual RAGAS metric as it finishes, using atomic output-file replacement. Job progress names the current case and metric; history includes saved-answer, saved-score and fully-scored-case counts. Partial averages include their valid-case denominators, and failed metrics remain null with explicit errors. An individual metric may require multiple model calls before its checkpoint can be saved.

Use **Resume an unfinished evaluation** in the RAGAS tab, or `.venv/bin/python evaluate_ragas.py --resume RUN_ID`. Resume skips saved answers and successful metrics, retries failed or interrupted work, and uses the original config/dataset snapshot. A changed index prevents resume to avoid mixing results. API restarts mark unfinished evaluations interrupted without discarding checkpoints. Only one evaluation runs at a time; recovery never starts scoring automatically.

## Independent load testing

Provide a name, representative questions, user count, spawn rate and duration. `/retrieve` measures retrieval; `/query` measures complete retrieval plus generation. Load tests do not evaluate answer quality or write a Live Run record for each request. The output includes actual aggregate QPS, Locust P99 request latency, request/failure counts and SLA comparisons. Warm/cold model state and concurrent indexing/evaluation affect results; keep workloads comparable when comparing runs. Only one load-test process runs at a time.

Targets are limited to the local API. BM25Okapi loads the corpus into memory and scans it for each query; use Milvus native BM25 for large corpora. This implementation does not claim million-document or 30 ms latency guarantees; those require actual workload measurements and deployment tuning.

## Implementation status

The local services and real UI/API wiring are implemented. End-to-end retrieval, golden-set evaluation and load measurements await your documents and references. No automated tests were requested or run.

History-constrained selection is the default. The top workspace selector contains Live Run, Tuning & Ablation, and Create new test. Tuning always restricts selection to saved experiments; Create new test provides free configuration and explicit execution. Each configuration page retains its own selections when switching. There is no mode-switch button below the selector. All parameter controls offer saved values that can complete a matching input/output pair for the fixed dataset and index, with shared settings compatible across both comparison paths. This mode never offers a Run trigger for missing results; normal mode retains free configuration.

History YAML parsing uses the safe C loader when available and a bounded cache keyed by file modification time, change time, inode and size. Switching pages and adjusting cards reuse unchanged parsed pairs; new files and atomic evaluation checkpoints are discovered through fresh file metadata checks.

Entering Tuning renders its first-card loading view immediately while a read-only worker loads history. A small UI fragment checks completion without blocking and enables the normal cards when ready. Refresh reloads history in the background; loading does not start evaluations or modify YAML.

When Retrieval is checked out in Only existing tests mode, BM25 and reranker enablement follow each retrieval mode. These derived flags are excluded from the shared-settings comparison and are preserved per saved configuration, so valid Vector-only/Hybrid/Rerank comparisons remain selectable.

Run a configurable retrieval-only Locust test with `.venv/bin/python retrieval_load_test.py`. All workload settings are at the top of the script: host, saved experiment input (or current config), question file, users, spawn rate, wait time, warmup, measurement duration, request timeout and acceptance thresholds. It sends POST /retrieve with frozen configuration and index identity. API keys come from the ignored secret file. Outputs go to data/load_test_history as paired input/output YAML, metrics JSON, Locust CSVs and a process log. The JSON measurements exclude requests completed before the warmup reset and include graceful-shutdown completions; in-flight requests crossing the warmup boundary count when they finish. CSV history may also contain warmup samples. QPS and latency scopes are explicit; relevance is not judged during this test. Use separate runs for each concurrency level; check the current USERS setting before starting.

The retrieval load script checks backend diagnostics, Milvus, the frozen index, and Docker resource monitoring before launching Locust. Each request records worker/executor queue waits and retrieval stages; process and container resource samples are saved alongside request JSONL and an aggregate diagnostics JSON/report. Missing diagnostic coverage marks the run PARTIAL. See [load-test readiness and measurement scopes](reports/retrieval_load_test_readiness.md). These diagnostics require restarting FastAPI after updating the backend.

The standalone script holds the shared load-test lock during preflight and execution, waits for the actual target user count before measurement, and enforces ramp-up and resource-coverage limits configured at its top. Input snapshots include the backend startup source fingerprint, commit and changed source paths. A backend fingerprint mismatch requires a restart before the run can proceed.

Each standalone run retains its original experiment input, effective configuration, complete question dataset, per-request question indexes and client/server correlation IDs. Dedicated server JSONL, periodic metrics checkpoints, preflight and container lifecycle snapshots, source ZIP and a final artifact/checksum manifest support later analysis. Missing/corrupt files, mismatched request counts or nonfinal metrics mark the run incomplete. See the readiness document for file names and interruption limits.

Before each tuning round, update `TUNING_ROUND` at the top of the standalone load script with the requested changes, before/after values, comparison baseline run ID and hypothesis. The report begins with these notes, and both YAML snapshots freeze them as their first field. Run only after explicit user execution instructions.

Query embedding device is explicit in `vector_search.query_device` (CPU by default, MPS optional). Index building continues to use CPU and keeps the same index signature. MPS experiments fail if unavailable or CPU fallback is enabled. Legacy CPU snapshots remain cache-compatible; MPS signatures are distinct. GPU memory is recorded per request, while GPU utilization is outside current monitoring.
