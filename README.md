# RAG Knowledge Base API (Flask + Groq + Qwen)

A Retrieval-Augmented Generation (RAG) API built with Flask. It ingests PDF
documents, indexes them with a hybrid dense (Chroma) + sparse (BM25)
retriever, and answers questions grounded in that content using either the
**Groq API** (hosted) or **Qwen** (running locally via `transformers`).

This copy ships with a sample document already in `data/`:
**`LLM_Complete_Guide.pdf`** — a reference guide covering transformer
architecture, training, RAG, fine-tuning, evaluation, agents, and more. Use
it to try the API immediately without needing your own PDF.

---

## 1. Project structure

```
project/
├── app.py                  # Flask application factory / entrypoint
├── config.py                # All settings: models, chunking, retrieval, paths
├── requirements.txt
├── .env                      # Environment variables (API keys, model choice…)
├── data/
│   └── LLM_Complete_Guide.pdf   # Sample document, ready to ingest
├── api/
│   ├── routes.py             # Flask Blueprint: /chat, /upload-pdf, /health…
│   └── schemas.py            # Pydantic request/response validation
├── models/
│   ├── registry.py           # Tracks + switches the active LLM at runtime
│   ├── generator.py          # RAGGenerator — dispatches to Groq or Qwen
│   └── qwen_loader.py        # Local Qwen model loading (transformers)
├── rag/
│   ├── ingestion.py           # PDF → per-page text (PyMuPDF)
│   ├── chunking.py            # Page text → overlapping chunks
│   ├── embeddings.py          # SentenceTransformer embeddings
│   └── retrieval.py           # Hybrid (vector + BM25) retriever
└── vectorDB/
    ├── chroma_store.py         # Persistent Chroma vector store
    └── BM25_DB.py               # In-memory BM25 sparse index
```

---

## 2. Setup

### 2.1 Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `torch` + `transformers` (needed for local Qwen inference) are
> large downloads. If you only plan to use Groq, you can remove
> `transformers`, `torch`, and `accelerate` from `requirements.txt` to speed
> up installation — the app still runs fine as long as you never switch to
> the `qwen3-8b` model.

### 2.2 Configure `.env`

Open `.env` and set at least your Groq key:

```env
GROQ_API_KEY=your_actual_groq_key_here
```

Get a free key at https://console.groq.com. Everything else has sensible
defaults — see the full list below.

| Variable              | Default                        | Purpose                                   |
|-----------------------|---------------------------------|--------------------------------------------|
| `GROQ_API_KEY`         | —                                | Required to use any `groq`-provider model |
| `QWEN_MODEL`            | `Qwen/Qwen3-8B`                  | HF repo id used for local inference        |
| `QWEN_DEVICE`            | `auto`                           | `auto` \| `cpu` \| `cuda`                  |
| `QWEN_MAX_NEW_TOKENS`     | `1024`                           | Max tokens generated per Qwen reply        |
| `DEFAULT_MODEL`           | `llama-3.3-70b-versatile`        | Active model at startup                    |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50`               | Text chunking parameters                   |
| `TOP_K`                    | `5`                              | Chunks retrieved per query                 |
| `RERANK_TOP_K`               | `3`                              | Chunks kept after reranking (if enabled)   |
| `ENABLE_RERANKING`             | `False`                         | Toggle cross-encoder reranking             |
| `HYBRID_ALPHA`                  | `0.5`                            | Vector vs BM25 weight (0=BM25, 1=vector)  |
| `PORT`                            | `5000`                           | Flask server port                          |

### 2.3 Run the server

```bash
python app.py
```

The API will be live at `http://localhost:5000`.

---

## 3. Quick start with the included PDF

The sample file `data/LLM_Complete_Guide.pdf` is already sitting in the
`data/` folder, but it still needs to be **ingested** (chunked, embedded,
and indexed) before you can query it — just upload it once:

```bash
curl -X POST http://localhost:5000/upload-pdf \
  -F "file=@data/LLM_Complete_Guide.pdf"
```

Response:
```json
{
  "filename": "LLM_Complete_Guide.pdf",
  "pages_processed": 26,
  "chunks_created": 145,
  "message": "Successfully ingested LLM_Complete_Guide.pdf"
}
```

Now ask it a question:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and why does it matter?"}'
```

Response (shape):
```json
{
  "question": "What is RAG and why does it matter?",
  "answer": "...",
  "sources": [
    {"source": "LLM_Complete_Guide.pdf", "page_number": 11, "score": 0.91, "excerpt": "..."}
  ],
  "model": "llama-3.3-70b-versatile",
  "retrieved_count": 5,
  "usage": {"prompt_tokens": 812, "completion_tokens": 143, "total_tokens": 955},
  "latency_s": 0.87
}
```

Some other questions worth trying against this specific document:
- "What's the difference between fine-tuning and RAG?"
- "Explain self-attention like I'm new to transformers."
- "What are the risks of prompt injection in a RAG system?"
- "Summarize the chunking strategies mentioned in the guide."

---

## 4. API Reference

| Method | Endpoint         | Description                                      |
|--------|------------------|---------------------------------------------------|
| GET    | `/health`         | Service status, active model, doc count, GPU info |
| GET    | `/models`          | List available models + which one is active/loaded|
| POST   | `/switch-model`      | Change the active LLM at runtime                  |
| POST   | `/upload-pdf`          | Ingest a PDF into the knowledge base              |
| POST   | `/chat`                 | Ask a question, get a grounded answer + sources   |
| GET    | `/sources`                | List all ingested PDF filenames                   |

### POST `/chat`

```json
{
  "question": "Your question here",
  "model": "llama-3.1-8b-instant",   // optional override
  "top_k": 5,                          // optional
  "rerank_top_k": 3,                    // optional
  "use_reranking": false                 // optional
}
```

### POST `/switch-model`

```json
{ "model": "qwen3-8b" }
```

Switching to `qwen3-8b` triggers a **one-time local download and load** of
the model on first use (can take a while depending on your connection and
hardware). Subsequent requests reuse the already-loaded model.

### POST `/upload-pdf`

Multipart form upload, field name `file`, PDF only.

---

## 5. Available models

| Name                          | Provider | Notes                                   |
|-------------------------------|----------|------------------------------------------|
| `llama-3.3-70b-versatile`      | groq     | Default. Strong general-purpose model.   |
| `llama-3.1-8b-instant`          | groq     | Faster, lighter, still solid quality.     |
| `qwen3-8b`                        | qwen     | Runs **locally**, no API key required — but needs enough RAM/VRAM and is slower without a GPU. |

Switch models anytime with `POST /switch-model`, or override per-request
with the `model` field in `/chat`.

---

## 6. Troubleshooting

- **`GROQ_API_KEY is not set`** → edit `.env`, replace `your_groq_api_key`
  with your real key, restart the server.
- **`Knowledge base is empty`** on `/chat`** → you haven't called
  `/upload-pdf` yet in this session (the vector store persists on disk, but
  the in-memory BM25 index does not survive a restart — re-upload after
  restarting the server).
- **Qwen is very slow** → it's likely running on CPU. Check `gpu_info` in
  `/health`. Consider a smaller model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) via
  `QWEN_MODEL` in `.env` if you don't have a GPU.
