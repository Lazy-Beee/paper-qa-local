"""Ask questions against the indexed paper library.

Three modes:

  python src/ask.py                                # interactive REPL
  python src/ask.py "What is X?"                   # one-shot
  python src/ask.py "What is X?" --out report.md   # one-shot + markdown report
  python src/ask.py --batch questions.txt --out answers.md  # batch run

Batch input format: one question per line. Lines starting with ``#`` and
blank lines are skipped.
"""
import argparse
import datetime
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from paperqa import ask

from paperqa_config import health_check, load_config, make_settings, setup_run_log


def _session_to_markdown(session, header_level: int = 2) -> str:
    """Render a single PQASession as Markdown."""
    h = "#" * header_level
    parts = [
        f"{h} Question\n\n{session.question.strip()}\n",
        f"{h} Answer\n\n{(session.answer or '').strip()}\n",
    ]
    if getattr(session, "answer_reasoning", None):
        parts.append(f"{h} Reasoning\n\n{session.answer_reasoning.strip()}\n")
    if getattr(session, "references", None):
        parts.append(f"{h} References\n\n{session.references.strip()}\n")
    contexts = getattr(session, "contexts", None) or []
    if contexts:
        ctx_parts = [f"{h} Contexts ({len(contexts)})\n"]
        for i, ctx in enumerate(contexts, 1):
            text_obj = getattr(ctx, "text", None)
            name = getattr(text_obj, "name", None) or getattr(ctx, "id", f"ctx-{i}")
            score = getattr(ctx, "score", None)
            score_str = f" — score {score}" if score is not None else ""
            summary = (getattr(ctx, "context", "") or "").strip()
            chunk = (getattr(text_obj, "text", "") or "").strip()
            ctx_parts.append(f"{h}# {i}. `{name}`{score_str}\n")
            if summary:
                ctx_parts.append(f"**Summary:** {summary}\n")
            if chunk:
                ctx_parts.append(f"```\n{chunk}\n```\n")
        parts.append("\n".join(ctx_parts))
    cost = getattr(session, "cost", None)
    tokens = getattr(session, "token_counts", None)
    if cost or tokens:
        parts.append(f"{h} Stats\n")
        if cost is not None:
            parts.append(f"- cost: {cost}")
        if tokens:
            parts.append(f"- token_counts: {dict(tokens) if hasattr(tokens, 'items') else tokens}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _build_report(sessions: list, settings_summary: dict) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = [
        "# paper-qa-local report",
        "",
        f"*Generated {timestamp}*",
        "",
        "## Settings",
        "",
    ]
    for k, v in settings_summary.items():
        head.append(f"- **{k}**: {v}")
    head.append("")
    body = []
    for i, session in enumerate(sessions, 1):
        body.append(f"---\n\n## {i}. {session.question.strip()}\n")
        body.append(_session_to_markdown(session, header_level=3))
    return "\n".join(head + body)


def _settings_summary(settings) -> dict:
    cfg = load_config()
    return {
        "chat_model": cfg["llm"]["chat_model"],
        "embedding_model": cfg["llm"]["embedding_model"],
        "api_base": cfg["llm"]["api_base"],
        "evidence_k": cfg["answer"]["evidence_k"],
        "answer_max_sources": cfg["answer"]["answer_max_sources"],
    }


def answer_one(question: str, settings):
    print(f"\nQuestion: {question}\n")
    print("Retrieving and generating answer...\n")
    response = ask(question, settings=settings)
    session = response.session
    print("=" * 70)
    print("[Answer]")
    print(session.formatted_answer)
    print("=" * 70)
    return session


def repl(settings) -> None:
    print("=" * 70)
    print("Paper QA - enter a question (empty line to quit)")
    print("=" * 70)
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            print("Bye!")
            break
        try:
            answer_one(question, settings)
        except Exception as e:
            print(f"Error: {e}")


def _read_batch(path: Path) -> list[str]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        questions.append(stripped)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions against the indexed paper library."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Single question. Omit for REPL (or use --batch).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write a Markdown report to this path (one-shot or --batch only).",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="Path to a file with one question per line (# and blank lines skipped).",
    )
    args = parser.parse_args()

    if args.batch and args.question:
        parser.error("Pass either a positional question or --batch, not both.")

    log_path = setup_run_log("ask")
    print(f"Logging to {log_path}")

    health_check()
    settings = make_settings(rebuild_index=False)

    if args.batch:
        if not args.batch.exists():
            parser.error(f"Batch file not found: {args.batch}")
        questions = _read_batch(args.batch)
        if not questions:
            parser.error(f"No questions found in {args.batch}")
        print(f"Running {len(questions)} questions from {args.batch}")
        sessions = []
        for i, q in enumerate(questions, 1):
            print(f"\n--- [{i}/{len(questions)}] ---")
            try:
                sessions.append(answer_one(q, settings))
            except Exception as e:
                print(f"Error on question {i}: {e}")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(_build_report(sessions, _settings_summary(settings)), encoding="utf-8")
            print(f"\nReport written: {args.out}")
        return

    if args.question:
        session = answer_one(args.question, settings)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(_build_report([session], _settings_summary(settings)), encoding="utf-8")
            print(f"\nReport written: {args.out}")
        return

    if args.out:
        print("Note: --out is ignored in REPL mode.")
    repl(settings)


if __name__ == "__main__":
    main()
