# paper-qa-local

Local literature Q&A backed by [paper-qa](https://github.com/Future-House/paper-qa) and an OpenAI-compatible LLM endpoint (LM Studio, Ollama, vLLM, ...). Drop PDFs in, build an index, ask questions and get cited answers — in the terminal or in a browser. Everything runs on your machine; no data leaves it.

## Prerequisites

- Python 3.11+ (uses `tomllib` from stdlib)
- An OpenAI-compatible LLM server running locally. Default config targets [LM Studio](https://lmstudio.ai/) at `http://localhost:1234`.
- Two models loaded on that server:
  - chat: `qwen2.5-14b-instruct-1m`
  - embedding: `text-embedding-qwen3-embedding-4b`

  (Both names are configurable in `config.toml`.)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows bash; use Activate.ps1 for PowerShell
pip install -r requirements.txt
```

## Usage

### 1. Add PDFs

Drop them into `data/test/` (or change `[paths].paper_dir` in `config.toml`).

Or copy in a single PDF and update the index in one shot:

```bash
python src/add.py path/to/local.pdf
python src/add.py path/to/local.pdf --no-index   # copy only, skip index update
python src/add.py path/to/local.pdf --overwrite  # replace if same filename exists
```

### 2. Build the index

```bash
python src/build_index.py
```

Index files land in `data/index/`. Re-running picks up new PDFs without re-processing existing ones.

### 3. Ask questions

**Web UI (recommended)** — double-click `start.bat`, or:

```bash
python src/web.py
```

Opens a chat interface at `http://127.0.0.1:7860` with the cited source snippets in a side panel.

**Terminal REPL:**

```bash
python src/ask.py
```

**One-shot CLI** (good for scripting / piping):

```bash
python src/ask.py "What is the corrected transport-velocity formulation in SPH?"
```

**One-shot with Markdown report** (question + answer + references + raw context snippets):

```bash
python src/ask.py "What is SPH?" --out report.md
```

**Batch mode** (one question per line, `#` for comments, blank lines skipped):

```bash
python src/ask.py --batch questions.txt --out answers.md
```

All entry points run a health check first — if LM Studio is down or a required model is not loaded, you get a clear error before paper-qa starts up.

### 4. Status

```bash
python src/status.py           # paths, paper count, index size, LLM health
python src/status.py --json    # machine-readable
```

## Configuration

All tunable parameters live in [`config.toml`](config.toml):

| Section | Knobs |
|---------|-------|
| `[paths]` | `paper_dir`, `index_dir` |
| `[llm]` | `api_base`, `api_key`, `chat_model`, `embedding_model`, `embedding_alias` |
| `[index]` | `recurse_subdirectories` |
| `[answer]` | `evidence_k`, `answer_max_sources`, `max_concurrent_requests` |
| `[parsing]` | `multimodal`, `use_doc_details`, `disable_doc_valid_check` |

To switch from the 5-PDF test set to the full library, change `paper_dir` to `…/data` (and probably set `recurse_subdirectories = true` if you have nested folders). The index is keyed off `paper_dir`, so a different value triggers a fresh index automatically.

## Logs

Every run writes a timestamped log alongside terminal output:

```
log/build_2026-04-27_14-12-32.log
log/ask_2026-04-27_14-15-08.log
```

The `log/` folder is gitignored.

## Project layout

| Path | Purpose |
|------|---------|
| `config.toml` | All tunable parameters |
| `requirements.txt` | Python dependencies |
| `start.bat` | One-click launcher for the web UI on Windows |
| `src/paperqa_config.py` | Loads config, runs health check, builds `Settings`, wires per-run logging |
| `src/build_index.py` | Builds the search index |
| `src/add.py` | Copy a local PDF into the paper directory and update the index |
| `src/ask.py` | REPL, one-shot, or batch Q&A with optional Markdown report |
| `src/status.py` | Status report: paths, paper / index stats, LLM health |
| `src/web.py` | Gradio web UI with sources side panel |
| `data/test/` | PDFs to index *(gitignored)* |
| `data/index/` | paper-qa index files *(gitignored)* |
| `log/` | Per-run logs *(gitignored)* |

## Troubleshooting

- **`Cannot reach LLM endpoint …`** — LM Studio (or your server) is not running, or `api_base` in `config.toml` is wrong.
- **`Endpoint … missing model(s)`** — the model name in `config.toml` does not match any model loaded on the server. Either load it or rename in config.
- **AES-encrypted PDF errors** — already handled by the `cryptography` dependency. If you still see them, reinstall: `pip install -r requirements.txt`.
- **Stale results after moving papers** — delete `data/index/` and rebuild.
