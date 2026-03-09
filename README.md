# Multimodal RAG Pipeline

RAG system for mixed-content PDFs — handles text, tables, and images together rather than treating documents as plain text.

---

## Overview

Most RAG pipelines chunk raw text and call it a day. This one uses Unstructured.io to split PDFs into typed elements (text blocks, HTML tables, base64 images), then generates GPT-4o summaries of chunks that contain visual content before embedding them. The idea is that embedding `<td>0.512</td>` directly is meaningless — a natural language description of what the table *says* retrieves much better.

At query time, the original tables and images are passed alongside the retrieved text to GPT-4o for grounded answer generation.

---

## Pipeline

```
PDF
 │
 ├─ Partition (Unstructured, hi_res + OCR)
 │    extracts text / tables / images as separate typed elements
 │
 ├─ Chunk by title
 │    respects document structure instead of fixed token windows
 │    max 3000 chars, merges fragments < 500 chars
 │
 ├─ AI summarisation (GPT-4o vision)
 │    text-only chunks → embed raw text
 │    mixed chunks → generate searchable summary first, then embed
 │
 ├─ ChromaDB (cosine, text-embedding-3-small)
 │    summary used for retrieval
 │    original content (tables as HTML, images as base64) stored in metadata
 │
 └─ Query → top-k retrieval → GPT-4o generates answer from full context
```

---

## Setup

```bash
# System deps (Linux)
apt-get install poppler-utils tesseract-ocr libmagic-dev

# Python
pip install -Uq "unstructured[all-docs]" langchain_chroma langchain langchain-community langchain-openai python_dotenv

# API key
echo "OPENAI_API_KEY=sk-..." > .env
```

## Usage

```python
# Ingest
db = run_complete_ingestion_pipeline("./docs/paper.pdf")

# Query
chunks = db.as_retriever(search_kwargs={"k": 3}).invoke("What does Table 2 show?")
print(generate_final_answer(chunks, query))
```

Tested on the Attention Is All You Need paper — correctly retrieves Table 3 and reads off attention head dimensions from the HTML.

---

## Stack

- **Parsing** — Unstructured.io (`hi_res`, Tesseract OCR)
- **Chunking** — title-based (Unstructured)
- **Embeddings** — `text-embedding-3-small`
- **Vector store** — ChromaDB
- **LLM** — GPT-4o

---

## Limitations

- Depends on OpenAI APIs throughout — not fully local
- No reranking step before generation
- Large PDFs are slow due to per-chunk GPT-4o calls

---

Saket Vispute, IIT Bombay
