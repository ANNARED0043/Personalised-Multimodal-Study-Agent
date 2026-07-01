# Personalised Multimodal Study Agent

This repository contains a personalised multimodal agent system for INFS4205/7205 Assignment 3. The system builds a course-specific knowledge base from lecture PDFs and notes, indexes both slide text and rendered slide images, retrieves grounded evidence, and uses a LangGraph workflow to answer questions, find slides, generate revision plans, support branch follow-up questions, and run quantitative ablation evaluation.

## Project Structure

```text
app.py                         Flask backend, SQLite schema, indexing, retrieval, LangGraph agent, evaluation API
static/index.html              Web application shell
static/styles.css              ChatGPT-style responsive interface
static/app.js                  Frontend state, streaming chat, sessions, evidence and branch panels
requirements.txt               Detailed dependency list
scripts/                       Report/figure generation utilities
data/evaluation/               Benchmark cases, evaluation outputs, generated figures
data/courses/INFS4205/uploads/ Optional preloaded lecture PDFs for reproducibility
data/courses/INFS4205/page_images/ Optional rendered slide images
```

Runtime database files are stored under `data/`. The current indexed database used in the evaluation is:

```text
data/study_agent_runtime_v2.sqlite
```

## Installation Instructions

1. Create and activate a Python environment. Python 3.10+ is recommended.

```powershell
conda create -n infs4205 python=3.11 -y
conda activate infs4205
```

If the environment already exists:

```powershell
conda activate infs4205
```

2. Install Python dependencies from the project root.

```powershell
pip install -r requirements.txt
```

3. Install Ollama for local LLM/VLM inference, then pull the recommended models.

```powershell
ollama pull qwen2.5:7b
ollama pull llava:latest
```

The app still runs if Ollama is unavailable, but the strongest generated answers use:

```text
text:qwen2.5:7b / vision:llava:latest
```

## Dependency File

Detailed Python dependencies are documented in `requirements.txt`. External model dependencies are installed through Ollama:

```powershell
ollama pull qwen2.5:7b
ollama pull llava:latest
```

## Run Instructions

Run the app from the project root:

```powershell
python app.py
```

Open the web interface:

```text
http://127.0.0.1:5000
```

The terminal should show a Flask development server similar to:

```text
Running on http://127.0.0.1:5000
```

This is expected for local coursework demonstration. It is not a production WSGI deployment.

## Optional Configuration

Set a course folder:

```powershell
$env:STUDY_AGENT_COURSE="INFS4205"
```

Set a custom data root:

```powershell
$env:STUDY_AGENT_DATA_DIR="E:\UQ\26S1\INFS4205\AASS\A3\data"
```

Set a custom database path:

```powershell
$env:STUDY_AGENT_DB_PATH="E:\UQ\26S1\INFS4205\AASS\A3\data\study_agent_runtime_v2.sqlite"
```

Enable automatic vision summaries with LLaVA during ingestion. This is slower but can produce richer visual captions:

```powershell
$env:STUDY_AGENT_AUTO_VISION_SUMMARY="1"
python app.py
```

## Main Features

- Personalised course knowledge base with 9 indexed lecture documents and 1017 evidence records in the current evaluation database.
- Multimodal indexing over `slide_text` and rendered `slide_image` evidence.
- Hybrid retrieval using text matching, vector similarity, visual signatures, metadata boosts, and rank fusion.
- LangGraph agent workflow with query routing, retrieval planning, grounding checks, memory/state updates, and persistent sessions.
- Streaming chat responses with evidence cards and slide-image references.
- Branch follow-up panel for local clarification questions without rewriting the main revision plan.
- Conversation history with create, reopen, continue, and delete support.
- Quantitative benchmark covering factual retrieval, cross-modal retrieval, analytical/multi-hop synthesis, and conversational personalised follow-up.
- Ablation comparisons including plain LLM, text-only, caption-only, no visual, no router, no rerank, no memory, and final agent.

## Evaluation Instructions

Run the benchmark from the UI:

```text
Open http://127.0.0.1:5000 -> Eval -> Run benchmark
```

Or run it via API while the Flask app is running:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/evaluate
```

Evaluation files are written to:

```text
data/evaluation/benchmark_cases.json
data/evaluation/evaluation_results.json
data/evaluation/evaluation_summary.csv
data/evaluation/failure_cases.md
```

The current summary table is:

```text
plain_llm:  recall=0.000, page_recall=0.000, MRR=0.000
text_only:  recall=0.627, page_recall=0.430, MRR=1.000
caption_only: recall=0.723, page_recall=0.603, MRR=1.000
no_rerank:  recall=0.610, page_recall=0.371, MRR=0.917
final_agent: recall=0.748, page_recall=0.553, MRR=1.000
```

## Reproducibility Notes

If the submitted package includes `data/study_agent_runtime_v2.sqlite`, the indexed knowledge base can be used immediately. If the database is omitted, upload the lecture PDFs through the Library page and click indexing/reindexing to rebuild the multimodal evidence records.

The system is intentionally local-first: it does not require a commercial API key. Ollama improves language and vision generation, while deterministic grounded fallbacks keep the retrieval and evaluation pipeline reproducible.

## Academic Integrity / Originality Note

The project implements an original course-study agent with a personalised lecture knowledge base, separate text/image evidence records, hybrid retrieval, LangGraph orchestration, branch-aware interaction, and structured ablation evaluation. The design, report, and evaluation are tailored to INFS4205/7205 A3 rather than copying the teaching demo.
