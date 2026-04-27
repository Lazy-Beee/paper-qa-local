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
import json
import sys
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr
from paperqa import ask

from paperqa_config import (
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


def _append_history(question: str, answer: str) -> None:
    entry = {
        "id": str(uuid.uuid4()),
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    entries = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
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
        return [], EMPTY_CONTEXTS_MSG
    for e in _load_history():
        if e.get("id") == entry_id:
            history = [
                {"role": "user", "content": e.get("question", "")},
                {"role": "assistant", "content": e.get("answer", "")},
            ]
            return history, "*Loaded from history — contexts are not stored.*"
    return [], "*Entry not found.*"


# --- Contexts rendering ---------------------------------------------------


def _render_contexts(contexts) -> str:
    if not contexts:
        return "*No contexts retrieved.*"
    lines = [f"### Contexts ({len(contexts)})\n"]
    for i, ctx in enumerate(contexts, 1):
        text_obj = getattr(ctx, "text", None)
        name = getattr(text_obj, "name", None) or getattr(ctx, "id", f"ctx-{i}")
        score = getattr(ctx, "score", None)
        score_str = f" — score {score}" if score is not None else ""
        summary = (getattr(ctx, "context", "") or "").strip()
        chunk = (getattr(text_obj, "text", "") or "").strip()
        lines.append(f"<details><summary><b>{i}. {name}</b>{score_str}</summary>\n")
        if summary:
            lines.append(f"\n**Summary:** {summary}\n")
        if chunk:
            preview = chunk if len(chunk) <= 1500 else chunk[:1500] + "…"
            lines.append(f"\n```text\n{preview}\n```\n")
        lines.append("</details>\n")
    return "\n".join(lines)


# --- Chat handler ---------------------------------------------------------


def respond(message: str, history: list):
    if not message.strip():
        yield history, "*Enter a question first.*"
        return

    history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Searching the library..."},
    ]
    yield history, "*Working...*"

    try:
        response = ask(message, settings=settings)
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"**Error:** {e}"}
        yield history, "*Failed.*"
        return

    session = response.session
    answer_text = session.answer or session.formatted_answer or "(empty answer)"
    history[-1] = {"role": "assistant", "content": answer_text}
    try:
        _append_history(message, answer_text)
    except Exception as e:
        print(f"Failed to append history: {e}")
    yield history, _render_contexts(getattr(session, "contexts", None) or [])


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
                library_md = gr.Markdown(_render_library())
                build_status = gr.Markdown("")
                with gr.Row():
                    refresh_lib_btn = gr.Button("Refresh counts")
                    update_idx_btn = gr.Button("Update index", variant="primary")
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
            chatbot = gr.Chatbot(height=600, label="Conversation")
            msg = gr.Textbox(
                placeholder="Ask a question and press Enter...",
                show_label=False,
                autofocus=True,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=1):
            contexts_panel = gr.Markdown(EMPTY_CONTEXTS_MSG)

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
        return [], "", EMPTY_CONTEXTS_MSG

    msg.submit(_submit, [msg, chatbot], [chatbot, contexts_panel]).then(
        lambda: "", None, msg
    ).then(refresh_history_handler, None, history_dd)
    submit_btn.click(_submit, [msg, chatbot], [chatbot, contexts_panel]).then(
        lambda: "", None, msg
    ).then(refresh_history_handler, None, history_dd)
    clear_btn.click(_clear, None, [chatbot, msg, contexts_panel])

    refresh_lib_btn.click(_render_library, None, library_md)
    update_idx_btn.click(update_index_handler, None, [build_status, library_md])
    refresh_hist_btn.click(refresh_history_handler, None, history_dd)
    load_hist_btn.click(load_history_handler, history_dd, [chatbot, contexts_panel])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
