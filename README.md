# IntelliQA: Document-Grounded RAG System

<p align="center">
  <a href="https://theanalyticmind.com/projects/intelliqa/">
    <img
      src="https://img.shields.io/badge/🚀%20Live%20Demo-IntelliQA-38BDF8?style=for-the-badge&labelColor=0F172A"
      alt="Live Demo: IntelliQA"
    />
  </a>
  <img src="https://img.shields.io/badge/v3.3-22C55E?style=for-the-badge&labelColor=111827" alt="Version 3.3" />
</p>

IntelliQA is a production-oriented Retrieval Augmented Generation backend for grounded question answering over private documents. Core logic ships as a Python wheel (`rag_pipeline`), so the same package powers a notebook demo, a live portfolio site, and an API service without rewrites.

Sessions are isolated so there is no cross-user leakage, uploads are capped and deduplicated, and a scheduled cron job manages storage. The model answers only from retrieved chunks at `temperature=0` and refuses when context is insufficient. Retrieval quality is not asserted, it is **measured** with a RAGAS evaluation harness that drives every tuning decision below.

> **The driving question:** how do you make LLM answers reliable, multi-tenant, and operationally sustainable on private documents?

## Problem Statement

<p align="center">
  <img src="./docs/RAG-Comparison.png" alt="Why standard RAG fails in production: hallucination, prototype fragility, vendor lock-in"
  style="max-width: 900px; width: 100%; height: auto; border-radius: 16px;" />
</p>

## Tech Stack

<div align="center">

| Category | Technology |
| :--- | :--- |
| **LLM & Inference** | ![OpenAI GPT-OSS 120B](https://img.shields.io/badge/OpenAI_GPT--OSS_120B-412991?style=flat&logo=openai&logoColor=white) ![Groq LPU](https://img.shields.io/badge/Groq_LPU-F55036?style=flat&logo=cpu&logoColor=white) |
| **Embeddings** | ![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black) ![BAAI/bge-large-en-v1.5](https://img.shields.io/badge/BAAI_bge--large--en--v1.5-0052CC?style=flat&logo=target&logoColor=white) |
| **Reranking** | ![Cross-Encoder](https://img.shields.io/badge/Cross--Encoder_ms--marco--MiniLM-6E56CF?style=flat&logo=huggingface&logoColor=white) |
| **RAG Framework** | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3A?style=flat&logo=langchain&logoColor=white) |
| **Vector Store** | ![PGVector](https://img.shields.io/badge/PGVector-4169E1?style=flat&logo=postgresql&logoColor=white) ![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?style=flat&logo=supabase&logoColor=white) |
| **Document Parsing** | ![Apache Tika](https://img.shields.io/badge/Apache_Tika-E65100?style=flat&logo=apache&logoColor=white) |
| **Evaluation** | ![Ragas](https://img.shields.io/badge/Ragas-FF6F00?style=flat&logo=googleanalytics&logoColor=white) ![Gemini](https://img.shields.io/badge/Gemini_Flash--Lite_(judge)-8E75FF?style=flat&logo=googlegemini&logoColor=white) ![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat&logo=ollama&logoColor=white) |
| **Deployment** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat&logo=amazonec2&logoColor=white) |
| **Packaging** | ![wheel](https://img.shields.io/badge/setup.py_%2B_wheel-3776AB?style=flat&logo=python&logoColor=white) |

</div>

## Key Features

<p align="center">
  <img src="./docs/RAG-key-features.png"
       alt="Six production-ready features"
       style="max-width: 900px; width: 100%; height: auto; border-radius: 16px;" />
</p>

## Architecture

At its core, IntelliQA wraps four stages into one installable pipeline: a parser converts documents to text, an embedder converts text to vectors, a vector store holds them for similarity search, and an LLM generates answers grounded in the retrieved chunks. The production system adds operational layers around that core.

**Ingestion** &nbsp;·&nbsp; Apache Tika (long-lived server) parses multi-format input. Text is normalized, deduplicated by content hash, and chunked before vectors touch the store.

**Storage** &nbsp;·&nbsp; PGVector on Supabase persists 1024-dimensional vectors from `bge-large-en-v1.5`. Tenant isolation is enforced by a `session_id` in each chunk's metadata, filtered at retrieval time.

**Retrieval** &nbsp;·&nbsp; Dense similarity search retrieves a wide candidate set, then a cross-encoder reranker reorders and trims it before generation.

**Generation** &nbsp;·&nbsp; GPT-OSS 120B (served on Groq LPU) at `temperature=0`, prompted to answer only from retrieved context and to refuse when the context is insufficient.

**Operations** &nbsp;·&nbsp; Session lifecycle, per-session upload quotas, and a daily cron job for cleanup.

<p align="center">
  <img src="./docs/RAG_Pipeline.png"
       alt="IntelliQA architecture: full system with session isolation and lifecycle management"
       style="height: auto; width: 100%; border-radius: 16px;"/>
</p>

## Evaluation

RAG quality is measured with a [RAGAS](https://github.com/explodinggradients/ragas) harness built on four metrics, so every component change (chunk size, retrieval `k`, embeddings, reranking) can be compared against a fixed baseline rather than guessed at.

| Metric | What it measures |
|:---|:---|
| **Faithfulness** | Does the answer follow from the retrieved chunks? *(generation grounding)* |
| **Answer Relevancy** | Does the response actually address the question? *(generation quality)* |
| **Context Precision** | Of the chunks retrieved, what fraction are relevant? *(retrieval ranking)* |
| **Context Recall** | Of the chunks needed to answer, how many were retrieved? *(retrieval coverage)* |

**Harness.** ~50 synthetic question-answer pairs generated with RAGAS over a deliberately diverse corpus (financial filings, tax publications, legal agreements, medical and structural engineering papers) to mirror the arbitrary documents a real user might upload. Each question is answered by the live pipeline; retrieved chunks, generated answer, and reference are scored by an LLM judge (Gemini Flash-Lite), with results checkpointed to JSONL so long runs survive rate limits.

### The finding that mattered most: the test set was the bug

Early scores looked poor, with answer relevancy near `0.31` and recall stuck no matter how `k` was raised. The instinct was to keep tuning retrieval. The data said otherwise.

The auto-generated questions were **vague by construction**. Generated one chunk at a time, they never named the entity they were about, asking things like *"what is the unrecognized tax benefit?"* against a corpus holding many near-identical filings. No retriever can resolve that, and RAGAS answer relevancy penalizes vague questions mechanically, since it reverse-generates questions from the answer and compares them to the original.

The fix was to rebuild the test set: generate per document so the source entity is known, then rewrite each question to name it. **Answer relevancy roughly doubled with zero changes to the RAG itself**, proving the low scores were a measurement artifact, not a retrieval failure.

The lesson generalizes: an evaluation set is a first-class artifact, and it needs validating before you trust a single number it produces.

### Retrieval tuning against the rebuilt test set

With a test set that could be trusted, `k` was swept one variable at a time.

| Config | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|:---|:---:|:---:|:---:|:---:|
| `k=4` (baseline) | **0.364** | 0.495 | 0.729 | 0.703 |
| `k=10` | 0.352 | 0.629 | 0.716 | 0.788 |
| `k=15` | 0.348 | 0.642 | 0.685 | **0.819** |
| `k=20` | 0.348 | **0.707** | **0.736** | 0.800 |
| `k=20` + rerank top-5 | **0.412** | 0.539 | 0.725 | 0.775 |

**What this shows:**

- **Recall climbs with `k` while precision stays flat (~0.35).** Retrieval finds the answer-bearing chunks when given room, but cannot rank them to the top. That flat precision across a 5x change in `k` is the textbook signal for a reranker.
- **The cross-encoder reranker lifts precision (`0.348` → `0.412`) but costs recall (`0.707` → `0.539`).** Trimming 20 chunks to a top-5 discards content that multi-fact answers need. A reranker reorders what retrieval found, it cannot recover what retrieval missed, so `top_n` is a precision-versus-context knob, not a recall lever.
- **Tuning `top_n` is the current work**, finding the point where precision stays above baseline while recall returns toward `0.70`.

> Every row above is one variable changed against a frozen test set. That discipline, not any single score, is the point of the harness.

## Installation & Usage

### Option 1: Install the prebuilt wheel

Use this if you want IntelliQA as a ready-to-use RAG backend. This is the path the live site uses.

```bash
git clone https://github.com/abhijitdeshpande83/IntelliQA-RAG-Powered-Document-Intelligence.git
cd IntelliQA-RAG-Powered-Document-Intelligence
pip install dist/rag_pipeline-3.2-py3-none-any.whl
```

Set credentials in a `.env` file at the project root:

```bash
GROQ_API_KEY="your-groq-key"
DATABASE_URL="your-postgres-connection-string"
```

Then import and use:

```python
from rag_pipeline.query_engine import load_data, vectorstore, ask_question, is_supported_file

chunks = load_data("report.pdf", session_id="abc123", file_name="report.pdf")
store = vectorstore(documents=chunks)
answer = ask_question("What was the reported revenue?", store, session_id="abc123")
```

The package exposes:

- `rag_pipeline.config` — model names and tuning defaults (`k`, `top_n`, chunk size)
- `rag_pipeline.utils` — parsing, chunking, deduplication
- `rag_pipeline.vector_store` — PGVector setup and indexing
- `rag_pipeline.query_engine` — retrieval, reranking, prompt assembly, generation

See `IntelliQA.ipynb` for end-to-end examples.

### Option 2: Install from source

Use this to read, modify, or extend the core RAG logic.

```bash
git clone https://github.com/abhijitdeshpande83/IntelliQA-RAG-Powered-Document-Intelligence.git
cd IntelliQA-RAG-Powered-Document-Intelligence

python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

jupyter notebook IntelliQA.ipynb
```

## Challenges & Lessons Learned

**Apache Tika JVM warm-up.** Spawning a fresh JVM per request caused unacceptable cold-start latency. Running a long-lived Tika server and proxying to it cut parse time from seconds to milliseconds. Tika is fast when warm, slow when treated like a CLI tool.

**Disk pressure on shared EC2.** The instance hosts both the portfolio site and IntelliQA, so vector storage and Tika temp files would grow unbounded. This drove the cron-based cleanup design. Shared infrastructure requires explicit lifecycle management.

**Embedding upgrade, measured not assumed.** The pipeline started on `all-MiniLM-L6-v2` (384 dims) for CPU efficiency. Migrating to `bge-large-en-v1.5` (1024 dims) was justified by measured retrieval gains on the eval harness, not by benchmark reputation.

**Clean extraction is a retrieval concern, not cosmetics.** Raw Tika output carried tabs, non-breaking spaces, and PDF hyphenation splits (`vesi-\ncles`). Split words fail to match at retrieval time. Normalizing text before chunking is a small change with a direct effect on embedding quality, and it was only visible because the eval harness surfaced retrieval failures.

**Over-refusal was a prompt problem, not a noise problem.** The default RetrievalQA prompt was too conservative, refusing whenever the answer was not stated word-for-word. Cleaning the text did not reduce refusals; rewriting the prompt to permit partial and rephrased context did.

## Status

|  |  |
| --- | --- |
| **Shipped** | Core pipeline, session isolation, upload quotas, deduplication, scheduled cleanup, PGVector migration, `bge-large` embeddings, GPT-OSS 120B generation, AWS EC2 deployment, `rag_pipeline-3.2` wheel, RAGAS evaluation harness with a validated test set, cross-encoder reranking |
| **In progress** | Reranker `top_n` tuning to recover recall while holding the precision gain |
| **Next** | Wider candidate pools feeding the reranker; parent-document retrieval to resolve the recall-versus-context tradeoff structurally; hybrid retrieval (BM25 + dense) for exact-term misses; inline citations linking answers to source chunks |