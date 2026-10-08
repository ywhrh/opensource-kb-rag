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

## 2. Run locally

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

For experiments, **Tuning & Ablation** compares saved results; **Create new test** lets you configure and explicitly run new experiments. RAGAS evaluation is separate from live answers and uses Ollama by default. To evaluate without Ollama, set `evaluation.judge_model` to `gpt-4o-mini` in `config.yaml` and provide your OpenAI key. The included 60-question reference set targets Hugging Face Transformers v4.57.1; use a matching corpus or supply a reference set for your own documents.

**Local use:** services bind to loopback. This studio does not implement multi-user authentication or document access control; a public deployment needs a separate access layer.
