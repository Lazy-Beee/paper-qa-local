"""Gradio web UI for paper-qa."""
import re
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


_QUESTION_PREFIX_RE = re.compile(r"^Question: .*?\n\n", re.DOTALL)


def respond(message: str, history) -> str:
    if not message.strip():
        return "Please enter a question."
    answer = ask(message, settings=settings)
    text = answer.session.formatted_answer
    return _QUESTION_PREFIX_RE.sub("", text, count=1)


demo = gr.ChatInterface(
    fn=respond,
    title="Paper QA",
    description="Ask questions against the indexed paper library. Answers include cited references.",
    examples=[
        "What is SPH?",
        "What is the EDAC SPH scheme?",
        "How is SPH used in additive manufacturing?",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
