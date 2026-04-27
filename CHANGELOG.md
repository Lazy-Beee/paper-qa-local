# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; versions
follow [SemVer](https://semver.org/).

## [1.0.0] — 2026-04-27

First tagged release. Stable enough for daily use against a local LM Studio
(or other OpenAI-compatible) backend.

### Added
- **Cross-encoder reranker.** Optional Qwen3-style reranker (default
  `qwen3-reranker-0.6b`) is inserted between embedding retrieval and the
  LLM summary stage. Pulls `evidence_k * oversample` candidates, scores
  each (query, chunk) pair as yes/no, keeps the top `evidence_k`. Configured
  via `[reranker]` in `config.toml`.
- **Config template + auto-seed.** `config.example.toml` is the tracked
  template; `config.toml` is gitignored. First run copies the example to
  `config.toml` automatically.
- **Web UI.** Gradio chat at `http://127.0.0.1:7860` with:
  - Inline paper-directory editor + index rebuild button.
  - Sources panel grouped by source document (chunks from the same paper
    collapse into one card with a click-to-copy citation block).
  - Conversation history (loadable past Q&As with their original sources).
  - Response time + token totals shown below each answer.
  - LaTeX rendering in answers and source summaries.
- **CLI surface.**
  - `add.py` — copy a local PDF in and update the index.
  - `ask.py` — REPL, one-shot, or batch (`--batch questions.txt`); optional
    Markdown report (`--out report.md`).
  - `build_index.py` — build/rebuild the paper-qa index.
  - `status.py` — paths, paper / index counts, LLM health (`--json` for
    machine-readable output).
- **Per-run logging.** Every entry point tees stdout/stderr to
  `log/<command>_<timestamp>.log` with ANSI colors stripped from the file.
- **Health check.** Verifies the LLM endpoint is reachable and required
  models (chat, embedding, reranker) are loaded before paper-qa starts up.
- `start.bat` one-click launcher for Windows.

### Notes
- Targets `paper-qa>=5,<6`, `gradio>=6.0.0`, Python 3.11+.
- Logprobs are not currently returned by LM Studio for the Qwen3 reranker,
  so reranker scoring is binary (yes/no); embedding rank is the tiebreaker.
