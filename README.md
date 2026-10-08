# OpenKB RAG

A RAG system and experiment studio built for developer documentation. Ask questions across technical docs, inspect the original evidence, and compare retrieval strategies instead of guessing which configuration works best.

**Stack:** Python · FastAPI · Streamlit · Milvus · BGE-M3 · BM25 · ONNX · RAGAS · Locust

## 1. Built for technical documentation

- **Meaning and exact terms.** Combine dense semantic search with BM25 to retrieve both conceptual explanations and exact API names, version numbers, and error codes. Compare vector-only, hybrid, and hybrid + rerank modes.
- **Code-aware chunking.** Preserve fenced code blocks and Markdown heading boundaries. Choose semantic, recursive, or fixed-size chunking; semantic chunking uses embedding distances to identify changes in meaning.
- **Context-rich search, original evidence.** Add document titles and section headings to searchable text while keeping evidence separate from enrichment. Answers stream alongside their original source passages.
- **Technical metadata.** Record source identifiers, content hashes, programming-language labels, and regex-extracted API/version/code annotations. BM25 matches technical terms in searchable text; entity annotations are not retrieval filters.
- **A second relevance pass.** Merge results with weighted fusion or reciprocal rank fusion (RRF), then optionally rerank with a BGE cross-encoder using ONNX or Cohere Rerank.
- **Measured quality and performance.** Compare configurations on a 60-question reference set spanning semantic, keyword, and distractor questions using four RAGAS metrics. Run independent Locust benchmarks with per-stage diagnostics; optionally enable Apple MPS query embedding, bounded dynamic batching, and LRU/TTL query caching.
- **Reproducible experiments.** Branch configurations in the tuning UI, retain config/dataset/index snapshots, and resume interrupted evaluations from saved answers and metric checkpoints.

## 2. Live Run — start locally and ask questions

### Install the project

You need **Python 3.12** and **Docker with Docker Compose**. Install Ollama only if you choose the local-model option below. The first setup downloads embedding and reranking models.

```bash
git clone https://github.com/ywhrh/opensource-kb-rag.git
cd opensource-kb-rag

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # First setup only; preserve an existing .env.

docker compose up -d
.venv/bin/python prepare_models.py
```

Choose **one** answer-model option:

### Option A — Local answers with Ollama

Start Ollama using its desktop app or `ollama serve`, then download the answer model:

```bash
ollama pull llama3.1:8b
```

The default configuration already uses `llama3.1:8b`. In `.env`, `OLLAMA_URL=http://127.0.0.1:11434` points to your local Ollama server. BGE embeddings and BGE reranking also run locally, so this configuration needs no model API key.

### Option B — OpenAI answers with your own API key

Set your key in the Git-ignored `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
```

After opening the UI, select **Live Run → Answer model → gpt-4o-mini**. Alternatively, enter your key under **API credentials → OpenAI API key**; UI keys are stored separately in the Git-ignored `.secrets/api_keys.json`, not in experiment snapshots.

BGE embeddings and reranking remain local unless you select a hosted alternative. Optional choices are `text-embedding-3-small` with your OpenAI key, and **Cohere Rerank v3** with a Cohere key. Hosted calls incur provider charges and send relevant input to that provider. This repository's available answer models are Ollama `llama3.1:8b` and OpenAI `gpt-4o-mini`.

### Start the backend and UI

Run these commands from the project directory in **two separate terminals**:

```bash
# Terminal 1 — backend
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2 — UI
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501**. API docs are at http://127.0.0.1:8000/docs.

### Ask your first question

1. Open **Live Run**.
2. Add your documents to `data/documents/`, or choose **Upload documents** in the UI and save your files. The local document corpus is not included in Git.
3. Click **Build / update document index** and wait for indexing to finish.
4. Enter your question and click **Run** to see the streamed answer and original evidence.

Supported inputs: Markdown, TXT, HTML, DOCX, and text-based PDF. Scanned PDFs require external OCR. Local studio uploads allow 20 MB per file, 200 MB per set, and 100 files; portfolio demo limits are separate.

**Local use:** services bind to loopback. This studio does not implement multi-user authentication or document access control; a public deployment needs a separate access layer.


## 3. Tuning — evaluate quality and benchmark performance

### Included golden dataset

The repository provides **60 question–reference answer pairs** based on [Hugging Face Transformers v4.57.1 documentation](https://github.com/huggingface/transformers/tree/v4.57.1/docs/source/en): **20 semantic, 20 keyword, and 20 distractor questions**.

- [Golden questions and answers](data/golden_test_dataset.json): the runtime evaluation dataset.
- [Source annotations](data/golden_test_dataset_sources.json): supporting document references and excerpts.
- [Readable review edition](data/golden_test_dataset_review.md): questions, reference answers, and supporting passages. References are source-grounded drafts pending independent human review.

The source corpus is not bundled. Add the matching Transformers documents to `data/documents/` and build the index before evaluating this dataset.

### Compare retrieval quality

**Tuning & Ablation** lets you branch configurations and compare saved experiments. **Create new test** lets you choose settings and explicitly run a new evaluation. Compare vector-only, hybrid, and hybrid + rerank with four RAGAS metrics: **context recall, context precision, faithfulness, and answer correctness**.

You can also run the evaluation script:

```bash
.venv/bin/python evaluate_ragas.py --name my-ablation --modes vector_only hybrid hybrid_rerank
```

Evaluation uses Ollama as its default judge. To use your OpenAI key instead, set `evaluation.judge_model: gpt-4o-mini` in `config.yaml`. Answers and individual metric scores are checkpointed; resume an unfinished run without regenerating completed work:

```bash
.venv/bin/python evaluate_ragas.py --resume RUN_ID
```

### Run pressure / load tests

The repository includes **`retrieval_load_test.py`**, an instrumented Locust script for retrieval throughput, P95/P99 latency, failures, per-stage timings, and process/container resource usage. It measures `/retrieve`, excluding answer generation and RAGAS scoring.

Before running, edit the settings at the top of the script:

- Set `EXPERIMENT_INPUT_PATH = None` to use your current `config.yaml`, or point it to an existing experiment's input snapshot.
- Choose `QUESTIONS_PATH`, `USERS`, `SPAWN_RATE`, `WARMUP_SECONDS`, and `MEASUREMENT_SECONDS` for your workload.
- Set `QUERY_EMBEDDING_DEVICE` to `cpu` unless you intend to test Apple MPS. Configure `QUERY_BATCHING`, `QUERY_CACHE`, and `CACHE_SCENARIO` explicitly; the checked-in settings describe a previous MPS/warm-cache experiment.
- Update `TUNING_ROUND` with your changes and comparison hypothesis. Keep the backend running and build an index compatible with the selected configuration.

```bash
.venv/bin/python retrieval_load_test.py
```

The UI also supports independent load tests targeting retrieval or complete retrieval + generation. Quality evaluation and load testing run separately.

### Test your own documents and configuration

1. Put your documents in `data/documents/`.
2. Adjust `config.yaml` or use **Create new test** to change chunking, embedding, retrieval, fusion, reranking, and generation settings. Rebuild the index when ingestion or index-related settings change.
3. For quality evaluation, save your own JSON question/reference-answer array under `data/`, then set `evaluation.dataset_path` to that file. Each item needs a unique `id`, a `category` (`semantic`, `keyword`, or `distractor`), a `question`, and a `reference_answer`. The CLI can evaluate your dataset; the tuning comparison UI is designed around the fixed 60-question set.
4. For load tests, set `QUESTIONS_PATH` to your question dataset and configure the workload in `retrieval_load_test.py`.

### Where results are saved

| Directory | Contents |
|---|---|
| `data/experiments_history/` | Live-run and evaluation input/output YAML pairs, including configuration snapshots, dataset/index identity, answers, scores, and evaluation checkpoints. |
| `data/load_test_history/` | Load-test input/output YAML pairs, metrics, Locust CSVs, logs, and instrumented request/resource artifacts. |
| `reports/` | Selected readable summaries and exported comparison reports. Raw run results are saved in the history directories above. |

Each run receives a unique ID. Runtime histories are Git-ignored; the tuning UI reads saved results without automatically rerunning tests.
