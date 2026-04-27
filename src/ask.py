import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from paperqa import ask

from paperqa_config import health_check, make_settings, setup_run_log


def answer_one(question: str, settings) -> None:
    print(f"\nQuestion: {question}\n")
    print("Retrieving and generating answer...\n")
    try:
        answer = ask(question, settings=settings)
        print("=" * 70)
        print("[Answer]")
        print(answer.session.formatted_answer)
        print("=" * 70)
    except Exception as e:
        print(f"Error: {e}")


def repl(settings) -> None:
    print("=" * 70)
    print("Paper QA - enter a question (empty line to quit)")
    print("=" * 70)
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            print("Bye!")
            break
        answer_one(question, settings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions against the indexed paper library."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Single question to answer. Omit to enter the interactive REPL.",
    )
    args = parser.parse_args()

    log_path = setup_run_log("ask")
    print(f"Logging to {log_path}")

    health_check()
    settings = make_settings(rebuild_index=False)

    if args.question:
        answer_one(args.question, settings)
    else:
        repl(settings)


if __name__ == "__main__":
    main()
