"""Gradio web UI for paper-qa with progressive status and a contexts panel.

Layout: left column = chat history + input; right column = contexts panel
showing the chunks paper-qa retrieved for the latest answer (citation key,
score, summary, and the original snippet).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr
from paperqa import ask

from paperqa_config import health_check, make_settings, setup_run_log

log_path = setup_run_log("web")
print(f"Logging to {log_path}")

health_check()
settings = make_settings(rebuild_index=False)


def _render_contexts(contexts) -> str:
    """Render paper-qa Context list as a Markdown panel."""
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


def respond(message: str, history: list):
    """Generator yielding (history, contexts_md) at each progress step."""
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
    yield history, _render_contexts(getattr(session, "contexts", None) or [])


with gr.Blocks(title="Paper QA") as demo:
    gr.Markdown(
        "# Paper QA\n"
        "Ask questions against the indexed paper library. "
        "Answers include cited references; the right panel shows the source snippets."
    )
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
            contexts_panel = gr.Markdown(
                "*Contexts will appear here after the first answer.*",
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
        return [], "", "*Contexts will appear here after the first answer.*"

    msg.submit(_submit, [msg, chatbot], [chatbot, contexts_panel]).then(
        lambda: "", None, msg
    )
    submit_btn.click(_submit, [msg, chatbot], [chatbot, contexts_panel]).then(
        lambda: "", None, msg
    )
    clear_btn.click(_clear, None, [chatbot, msg, contexts_panel])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
