"""Gradio web UI for paper-qa.

Layout:
- Top accordion: library status (paper directory, indexed/pending/failed
  counts, Update Index button) and conversation history (load past Q&A
  from log/conversations.jsonl).
- Main row: chat (left) + sources panel showing retrieved context
  chunks for the latest answer (right).

Each answered question appends a JSONL record to log/conversations.jsonl
so it can be recalled later from the History dropdown.
"""
import asyncio
import datetime
import html as _html
import json
import re
import sys
import time
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr
from paperqa import ask

from paperqa_config import (
    CONFIG_PATH,
    LOG_DIR,
    _resolve_path,
    health_check,
    load_config,
    make_settings,
    setup_run_log,
)

log_path = setup_run_log("web")
print(f"Logging to {log_path}")

health_check()
settings = make_settings(rebuild_index=False)

HISTORY_PATH = LOG_DIR / "conversations.jsonl"
EMPTY_CONTEXTS_MSG = "*Contexts will appear here after the first answer.*"
HISTORY_LABEL_MAX = 70

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "$", "right": "$", "display": False},
]

PAGE_CSS = """
.gradio-container,
.gradio-container *:not(pre):not(code):not(.pqa-citation) {
    font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
}
.pqa-citation {
    user-select: all;
    cursor: copy;
    padding: 8px 10px;
    background: rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(0, 0, 0, 0.18);
    border-radius: 4px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: Arial, "Helvetica Neue", sans-serif;
    font-size: 0.92em;
    margin: 4px 0 8px 0;
    line-height: 1.4;
}
.pqa-citation:hover { background: rgba(0, 0, 0, 0.08); }
.pqa-citation-label {
    font-weight: 600;
    margin-top: 6px;
    font-size: 0.95em;
}
.pqa-citation-hint {
    font-weight: 400;
    font-size: 0.82em;
    opacity: 0.7;
    margin-left: 6px;
}
.pqa-stats {
    font-size: 0.82em;
    opacity: 0.65;
    font-style: italic;
    margin: 0 4px 4px 4px;
    padding: 0;
    line-height: 1.2;
    min-height: 0;
}
.pqa-stats p { margin: 0; }
.pqa-contexts details { margin-bottom: 14px; }
.pqa-contexts details:last-of-type { margin-bottom: 0; }
"""


# --- Library counts -------------------------------------------------------


def _count_pdfs(paper_dir: Path) -> int:
    return len(list(paper_dir.glob("*.pdf"))) if paper_dir.exists() else 0


async def _probe_paperqa_index() -> dict:
    from paperqa.agents.search import get_directory_index

    try:
        idx = await get_directory_index(settings=settings, build=False)
        files = await idx.index_files
        successful = sum(1 for v in files.values() if v != "failed")
        failed = sum(1 for v in files.values() if v == "failed")
        return {"indexed": successful, "failed": failed, "tracked": len(files)}
    except Exception as e:
        return {"indexed": 0, "failed": 0, "tracked": 0, "error": str(e)}


def _render_library() -> str:
    cfg = load_config()
    paper_dir = Path(_resolve_path(cfg["paths"]["paper_dir"]))
    total_pdfs = _count_pdfs(paper_dir)
    counts = asyncio.run(_probe_paperqa_index())
    indexed = counts["indexed"]
    failed = counts["failed"]
    tracked = counts["tracked"]
    pending = max(total_pdfs - tracked, 0)
    lines = [
        f"**Paper directory:** `{paper_dir}`",
        "",
        f"- PDFs on disk: **{total_pdfs}**",
        f"- Indexed: **{indexed}**",
        f"- Failed to index: **{failed}**",
        f"- Pending (not yet seen by index): **{pending}**",
    ]
    if "error" in counts:
        lines.append(f"\n*Index probe note: {counts['error']}*")
    return "\n".join(lines)


# --- Paper directory editing ---------------------------------------------


def _initial_paper_dir() -> str:
    return load_config()["paths"]["paper_dir"]


def _write_paper_dir_to_config(new_path: str) -> None:
    """Replace the paper_dir line in config.toml, preserving comments and line endings."""
    import re

    safe_path = new_path.replace("\\", "/")
    text = CONFIG_PATH.read_bytes().decode("utf-8")
    new_text, n = re.subn(
        r'^(\s*paper_dir\s*=\s*)"[^"]*"',
        lambda m: m.group(1) + f'"{safe_path}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        raise ValueError("paper_dir line not found in [paths] section of config.toml")
    CONFIG_PATH.write_bytes(new_text.encode("utf-8"))


def change_paper_dir_handler(new_path: str):
    global settings
    new_path = (new_path or "").strip()
    if not new_path:
        return "*Path cannot be empty.*", _render_library()

    resolved = Path(_resolve_path(new_path))
    if not resolved.exists():
        return f"*Path does not exist: `{resolved}`*", _render_library()
    if not resolved.is_dir():
        return f"*Not a directory: `{resolved}`*", _render_library()

    try:
        _write_paper_dir_to_config(new_path)
    except Exception as e:
        return f"*Failed to write config.toml: {e}*", _render_library()

    settings = make_settings(rebuild_index=False)
    return (
        f"*Switched to `{resolved}`. Click **Update index** to (re)build the index for this folder.*",
        _render_library(),
    )


# --- Index build ----------------------------------------------------------


async def _build_index_async() -> None:
    from paperqa.agents.search import get_directory_index

    settings.agent.rebuild_index = False
    await get_directory_index(settings=settings, build=True)


def update_index_handler():
    yield "*Building index — this can take a while...*", gr.update()
    try:
        asyncio.run(_build_index_async())
        yield "*Index update complete.*", _render_library()
    except Exception as e:
        yield f"*Index update failed: {e}*", _render_library()


# --- Conversation history -------------------------------------------------


def _append_history(
    question: str,
    answer: str,
    contexts: list[dict] | None = None,
    stats: str = "",
) -> None:
    entry = {
        "id": str(uuid.uuid4()),
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "contexts": contexts or [],
        "stats": stats,
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


_INLINE_STATS_RE = re.compile(r"\n\n---\n_(.+?)_\s*$", re.DOTALL)


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    entries = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Backward-compat: older entries embedded the stats line as a footer
        # inside `answer`. Split it out into the new `stats` field on read so
        # both flows render the same way.
        if not e.get("stats"):
            ans = e.get("answer", "")
            m = _INLINE_STATS_RE.search(ans)
            if m:
                e["stats"] = m.group(1).strip()
                e["answer"] = ans[: m.start()]
        entries.append(e)
    return entries


def _history_choices() -> list[tuple[str, str]]:
    entries = _load_history()
    entries.reverse()
    out = []
    for e in entries:
        ts = e.get("ts", "")[:16]
        q = (e.get("question") or "").replace("\n", " ")
        if len(q) > HISTORY_LABEL_MAX:
            q = q[:HISTORY_LABEL_MAX] + "…"
        out.append((f"[{ts}] {q}", e.get("id", "")))
    return out


def refresh_history_handler():
    return gr.update(choices=_history_choices(), value=None)


def load_history_handler(entry_id):
    if not entry_id:
        return [], EMPTY_CONTEXTS_MSG, ""
    for e in _load_history():
        if e.get("id") == entry_id:
            history = [
                {"role": "user", "content": e.get("question", "")},
                {"role": "assistant", "content": e.get("answer", "")},
            ]
            saved_contexts = e.get("contexts") or []
            stats = e.get("stats") or ""
            if saved_contexts:
                return history, _render_contexts(saved_contexts), stats
            return (
                history,
                "*Loaded from history — no contexts were saved with this entry "
                "(it predates context persistence).*",
                stats,
            )
    return [], "*Entry not found.*", ""


# --- Contexts rendering ---------------------------------------------------


def _ctx_to_dict(ctx) -> dict:
    """Pluck the fields _render_contexts shows from a paper-qa Context."""
    text_obj = getattr(ctx, "text", None)
    name = getattr(text_obj, "name", None) or getattr(ctx, "id", "")
    doc = getattr(text_obj, "doc", None) if text_obj else None
    citation = (getattr(doc, "citation", "") or "").strip() if doc else ""
    dockey = getattr(doc, "dockey", None) if doc else None
    return {
        "name": str(name),
        "score": getattr(ctx, "score", None),
        "summary": (getattr(ctx, "context", "") or "").strip(),
        "chunk": (getattr(text_obj, "text", "") or "").strip(),
        "citation": citation,
        # Dockey is a stable per-document id; falling back to citation keeps
        # historical entries (saved before this field existed) deduping by
        # the only stable identifier they have.
        "dockey": str(dockey) if dockey else citation,
    }


def _doc_label(group: list[dict]) -> str:
    """Short label for a doc card: prefer the trimmed chunk name prefix."""
    name = group[0].get("name") or ""
    # Chunk names look like "Ihmsen2014 chunk 5" or "Ihmsen2014 pages 1-2".
    # Strip the per-chunk suffix so the card header names the doc, not a chunk.
    for sep in (" chunk ", " pages "):
        if sep in name:
            return name.split(sep, 1)[0]
    return name or "(unnamed)"


def _render_contexts(contexts) -> str:
    """Render contexts grouped by source document (dedupes same-doc chunks)."""
    if not contexts:
        return "*No contexts retrieved.*"
    items = [c if isinstance(c, dict) else _ctx_to_dict(c) for c in contexts]

    # Group by dockey; preserve first-seen order so the highest-scored doc
    # (paperqa returns sorted) appears first.
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for item in items:
        key = item.get("dockey") or item.get("citation") or item.get("name") or ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    n_chunks = len(items)
    n_docs = len(order)
    lines = [f"### Sources ({n_docs} doc{'s' if n_docs != 1 else ''} · {n_chunks} chunk{'s' if n_chunks != 1 else ''})\n"]
    for i, key in enumerate(order, 1):
        group = groups[key]
        label = _doc_label(group)
        scores = [g.get("score") for g in group if g.get("score") is not None]
        best = max(scores) if scores else None
        best_str = f" — best score {best}" if best is not None else ""
        chunk_count = len(group)
        chunk_str = f" · {chunk_count} chunks" if chunk_count > 1 else ""
        citation = (group[0].get("citation") or "").strip()

        lines.append(
            f"<details><summary><b>{i}. {_html.escape(label)}</b>{best_str}{chunk_str}</summary>\n"
        )
        if citation:
            lines.append(
                '<div class="pqa-citation-label">Citation'
                '<span class="pqa-citation-hint">(click to select, Ctrl+C to copy)</span>'
                "</div>\n"
                f'<pre class="pqa-citation">{_html.escape(citation)}</pre>\n'
            )
        for j, item in enumerate(group, 1):
            chunk_name = item.get("name") or ""
            score = item.get("score")
            score_str = f" — score {score}" if score is not None else ""
            summary = (item.get("summary") or "").strip()
            chunk = (item.get("chunk") or "").strip()
            heading = (
                f"\n**Chunk {j}** (`{_html.escape(chunk_name)}`{score_str})\n"
                if chunk_count > 1
                else ""
            )
            if heading:
                lines.append(heading)
            if summary:
                lines.append(f"\n**Summary:** {summary}\n")
            if chunk:
                preview = chunk if len(chunk) <= 1500 else chunk[:1500] + "…"
                lines.append(f"\n```text\n{preview}\n```\n")
        lines.append("</details>\n")
    return "\n".join(lines)


# --- Stats footer ---------------------------------------------------------


def _format_stats(elapsed_s: float, session) -> str:
    """One-line summary appended below an answer: time + token totals."""
    parts = [f"{elapsed_s:.1f}s"]
    tokens = getattr(session, "token_counts", None) or {}
    total_p = total_c = 0
    for vals in tokens.values():
        if isinstance(vals, (list, tuple)) and len(vals) >= 2:
            try:
                total_p += int(vals[0])
                total_c += int(vals[1])
            except (TypeError, ValueError):
                continue
    if total_p or total_c:
        parts.append(
            f"prompt {total_p:,} + completion {total_c:,} = {total_p + total_c:,} tokens"
        )
    cost = getattr(session, "cost", None)
    try:
        if cost and float(cost) > 0:
            parts.append(f"${float(cost):.4f}")
    except (TypeError, ValueError):
        pass
    return " · ".join(parts)


# --- Chat handler ---------------------------------------------------------


def respond(message: str, history: list):
    if not message.strip():
        yield history, "*Enter a question first.*", ""
        return

    history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Searching the library..."},
    ]
    yield history, "*Working...*", ""

    start = time.perf_counter()
    try:
        response = ask(message, settings=settings)
    except Exception as e:
        elapsed = time.perf_counter() - start
        history[-1] = {"role": "assistant", "content": f"**Error:** {e}"}
        yield history, "*Failed.*", f"{elapsed:.1f}s (failed)"
        return
    elapsed = time.perf_counter() - start

    session = response.session
    answer_text = session.answer or session.formatted_answer or "(empty answer)"
    stats = _format_stats(elapsed, session)
    history[-1] = {"role": "assistant", "content": answer_text}
    contexts = getattr(session, "contexts", None) or []
    ctx_dicts = [_ctx_to_dict(c) for c in contexts]
    try:
        _append_history(message, answer_text, ctx_dicts, stats)
    except Exception as e:
        print(f"Failed to append history: {e}")
    yield history, _render_contexts(ctx_dicts), stats


# --- UI -------------------------------------------------------------------


with gr.Blocks(title="Paper QA") as demo:
    gr.Markdown("# Paper QA")
    gr.Markdown(
        "Ask questions against the indexed paper library. "
        "Answers include cited references; the right panel shows the "
        "source snippets."
    )

    with gr.Accordion("Library & history", open=True):
        with gr.Row():
            with gr.Column():
                paper_dir_input = gr.Textbox(
                    value=_initial_paper_dir(),
                    label="Paper directory (relative to project root, or absolute)",
                    interactive=True,
                )
                with gr.Row():
                    apply_dir_btn = gr.Button("Apply path")
                    refresh_lib_btn = gr.Button("Refresh counts")
                    update_idx_btn = gr.Button("Update index", variant="primary")
                dir_status = gr.Markdown("")
                library_md = gr.Markdown(_render_library())
                build_status = gr.Markdown("")
            with gr.Column():
                history_dd = gr.Dropdown(
                    choices=_history_choices(),
                    label="Past conversations",
                    value=None,
                    interactive=True,
                )
                with gr.Row():
                    refresh_hist_btn = gr.Button("Refresh history")
                    load_hist_btn = gr.Button("Load selected", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                min_height=200,
                max_height=600,
                label="Conversation",
                latex_delimiters=LATEX_DELIMS,
            )
            stats_md = gr.Markdown("", elem_classes=["pqa-stats"])
            msg = gr.Textbox(
                placeholder="Ask a question and press Enter...",
                show_label=False,
                autofocus=True,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=1):
            contexts_panel = gr.Markdown(
                EMPTY_CONTEXTS_MSG,
                latex_delimiters=LATEX_DELIMS,
                elem_classes=["pqa-contexts"],
            )

    gr.Examples(
        examples=[
            "What is SPH?",
            "What is the EDAC SPH scheme?",
            "How is SPH used in additive manufacturing?",
        ],
        inputs=msg,
    )

    def _submit(message, history):
        yield from respond(message, history)

    def _clear():
        return [], "", EMPTY_CONTEXTS_MSG, ""

    msg.submit(
        _submit, [msg, chatbot], [chatbot, contexts_panel, stats_md]
    ).then(lambda: "", None, msg).then(refresh_history_handler, None, history_dd)
    submit_btn.click(
        _submit, [msg, chatbot], [chatbot, contexts_panel, stats_md]
    ).then(lambda: "", None, msg).then(refresh_history_handler, None, history_dd)
    clear_btn.click(_clear, None, [chatbot, msg, contexts_panel, stats_md])

    apply_dir_btn.click(
        change_paper_dir_handler, paper_dir_input, [dir_status, library_md]
    )
    refresh_lib_btn.click(_render_library, None, library_md)
    update_idx_btn.click(update_index_handler, None, [build_status, library_md])
    refresh_hist_btn.click(refresh_history_handler, None, history_dd)
    load_hist_btn.click(
        load_history_handler, history_dd, [chatbot, contexts_panel, stats_md]
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(font=["Arial", "Helvetica Neue", "sans-serif"]),
        css=PAGE_CSS,
    )
