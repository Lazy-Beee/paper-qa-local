"""Print a status report: paths, paper / index stats, LLM health.

Designed to work even when the LLM endpoint is down — endpoint and
index probes are best-effort and never abort the report.

Usage:
  python src/status.py
  python src/status.py --json    # emit machine-readable JSON
"""
import argparse
import asyncio
import datetime
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from paperqa_config import LOG_DIR, _resolve_path, load_config


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n} B"


def _fmt_mtime(path: Path) -> str:
    if not path.exists():
        return "(missing)"
    ts = max(p.stat().st_mtime for p in path.rglob("*")) if path.is_dir() else path.stat().st_mtime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def collect_paper_stats(paper_dir: Path) -> dict:
    pdfs = sorted(paper_dir.glob("*.pdf")) if paper_dir.exists() else []
    metas = sorted(paper_dir.glob("*.meta.json")) if paper_dir.exists() else []
    return {
        "path": str(paper_dir),
        "exists": paper_dir.exists(),
        "pdf_count": len(pdfs),
        "meta_sidecar_count": len(metas),
        "total_bytes": _dir_size(paper_dir),
        "last_modified": _fmt_mtime(paper_dir) if paper_dir.exists() else None,
    }


def collect_index_stats(index_dir: Path) -> dict:
    return {
        "path": str(index_dir),
        "exists": index_dir.exists(),
        "total_bytes": _dir_size(index_dir),
        "last_modified": _fmt_mtime(index_dir) if index_dir.exists() else None,
    }


async def _probe_paperqa_index() -> dict:
    """Best-effort: count successfully indexed documents."""
    from paperqa.agents.search import get_directory_index

    from paperqa_config import make_settings

    settings = make_settings(rebuild_index=False)
    idx = await get_directory_index(settings=settings, build=False)
    files = await idx.index_files
    indexed = sum(1 for v in files.values() if v != "failed")
    failed = sum(1 for v in files.values() if v == "failed")
    return {"indexed": indexed, "failed": failed, "tracked": len(files)}


def collect_paperqa_index_probe() -> dict:
    try:
        return {"ok": True, **asyncio.run(_probe_paperqa_index())}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def collect_llm_health(api_base: str, chat_model: str, embedding_model: str) -> dict:
    url = f"{api_base.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        return {"ok": False, "endpoint": url, "error": str(getattr(e, "reason", e))}

    available = sorted({m["id"] for m in data.get("data", [])})
    needed = {"chat": chat_model, "embedding": embedding_model}
    missing = [k for k, v in needed.items() if v not in available]
    return {
        "ok": not missing,
        "endpoint": url,
        "available_models": available,
        "needed_models": needed,
        "missing": missing,
    }


def collect_log_stats() -> dict:
    if not LOG_DIR.exists():
        return {"path": str(LOG_DIR), "exists": False, "file_count": 0, "total_bytes": 0}
    files = list(LOG_DIR.glob("*.log"))
    return {
        "path": str(LOG_DIR),
        "exists": True,
        "file_count": len(files),
        "total_bytes": _dir_size(LOG_DIR),
        "last_modified": _fmt_mtime(LOG_DIR),
    }


def gather() -> dict:
    cfg = load_config()
    paper_dir = Path(_resolve_path(cfg["paths"]["paper_dir"]))
    index_dir = Path(_resolve_path(cfg["paths"]["index_dir"]))
    llm = cfg["llm"]
    return {
        "paths": {"paper_dir": str(paper_dir), "index_dir": str(index_dir)},
        "papers": collect_paper_stats(paper_dir),
        "index_disk": collect_index_stats(index_dir),
        "index_paperqa": collect_paperqa_index_probe(),
        "llm": collect_llm_health(llm["api_base"], llm["chat_model"], llm["embedding_model"]),
        "logs": collect_log_stats(),
    }


def render(report: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("paper-qa-local status")
    lines.append("=" * 70)

    p = report["papers"]
    lines.append("\n[Papers]")
    lines.append(f"  dir          : {p['path']}")
    if not p["exists"]:
        lines.append("  state        : MISSING")
    else:
        lines.append(f"  PDF files    : {p['pdf_count']}")
        lines.append(f"  with metadata: {p['meta_sidecar_count']}")
        lines.append(f"  total size   : {_fmt_bytes(p['total_bytes'])}")
        lines.append(f"  last modified: {p['last_modified']}")

    d = report["index_disk"]
    lines.append("\n[Index (disk)]")
    lines.append(f"  dir          : {d['path']}")
    if not d["exists"]:
        lines.append("  state        : NOT BUILT")
    else:
        lines.append(f"  total size   : {_fmt_bytes(d['total_bytes'])}")
        lines.append(f"  last modified: {d['last_modified']}")

    pq = report["index_paperqa"]
    lines.append("\n[Index (paper-qa)]")
    if pq["ok"]:
        lines.append(f"  indexed docs : {pq['indexed']}")
        lines.append(f"  failed docs  : {pq['failed']}")
        lines.append(f"  tracked total: {pq['tracked']}")
    else:
        lines.append(f"  probe failed : {pq['error']}")

    h = report["llm"]
    lines.append("\n[LLM endpoint]")
    lines.append(f"  endpoint     : {h['endpoint']}")
    if not h["ok"] and "available_models" not in h:
        lines.append(f"  state        : UNREACHABLE ({h.get('error')})")
    else:
        needed = h["needed_models"]
        lines.append(f"  chat model   : {needed['chat']}  "
                     f"{'OK' if needed['chat'] in h['available_models'] else 'NOT LOADED'}")
        lines.append(f"  embedding    : {needed['embedding']}  "
                     f"{'OK' if needed['embedding'] in h['available_models'] else 'NOT LOADED'}")
        lines.append(f"  loaded models: {', '.join(h['available_models']) or '(none)'}")
        if h["missing"]:
            lines.append(f"  MISSING      : {', '.join(h['missing'])}")

    lg = report["logs"]
    lines.append("\n[Logs]")
    if not lg["exists"]:
        lines.append("  state        : empty")
    else:
        lines.append(f"  dir          : {lg['path']}")
        lines.append(f"  files        : {lg['file_count']}")
        lines.append(f"  total size   : {_fmt_bytes(lg['total_bytes'])}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show project status (paths, papers, index, LLM health).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a formatted report.")
    args = parser.parse_args()

    report = gather()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
