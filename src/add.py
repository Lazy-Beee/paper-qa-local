"""Add a local PDF to the library and trigger an incremental index update.

Usage:
  python src/add.py path/to/file.pdf
  python src/add.py path/to/file.pdf --no-index
  python src/add.py path/to/file.pdf --overwrite
"""
import argparse
import asyncio
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from paperqa_config import (
    _resolve_path,
    health_check,
    load_config,
    make_settings,
    setup_run_log,
)


def _looks_like_pdf(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(4) == b"%PDF"


def add_paper(src: Path, paper_dir: Path, overwrite: bool = False) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"No such file: {src}")
    if not src.is_file():
        raise ValueError(f"Not a file: {src}")
    if not _looks_like_pdf(src):
        raise ValueError(f"{src} does not look like a PDF (missing %PDF header).")

    paper_dir.mkdir(parents=True, exist_ok=True)
    dst = paper_dir / src.name
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"{dst} already exists. Pass --overwrite to replace it."
        )
    shutil.copy2(src, dst)
    print(f"Saved: {dst}  ({dst.stat().st_size:,} bytes)")
    return dst


async def run_index(settings) -> None:
    """Trigger an incremental index update for newly added papers."""
    from paperqa.agents.search import get_directory_index

    print("\nUpdating index (incremental)...")
    try:
        await get_directory_index(settings=settings, build=True)
        print("Index update complete.")
    except Exception as e:
        print(
            f"Index update raised: {e}. "
            "PDF is saved; rerun build_index.py later to retry."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a local PDF into the paper directory and update the index.",
    )
    parser.add_argument("path", type=Path, help="Path to a local PDF file.")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Copy the PDF only; skip the incremental index update.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination file if it already exists.",
    )
    args = parser.parse_args()

    log_path = setup_run_log("add")
    print(f"Logging to {log_path}")

    cfg = load_config()
    paper_dir = Path(_resolve_path(cfg["paths"]["paper_dir"]))

    add_paper(args.path, paper_dir=paper_dir, overwrite=args.overwrite)

    if args.no_index:
        print("\nSkipping index update (--no-index). Run build_index.py later.")
        return

    health_check()
    settings = make_settings(rebuild_index=False)
    asyncio.run(run_index(settings))


if __name__ == "__main__":
    main()
