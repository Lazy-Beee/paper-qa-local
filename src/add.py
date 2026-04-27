"""Add a paper to the library by DOI, arXiv ID, URL, or local path.

Resolves metadata (CrossRef for DOI, arXiv API for arXiv) and an open-access
PDF (Unpaywall for DOI), saves the PDF + sidecar JSON into the paper
directory, and triggers an incremental paper-qa index update.

Usage:
  python src/add.py 10.1234/foo                # DOI
  python src/add.py 2401.12345                 # arXiv id
  python src/add.py arxiv:hep-th/0603001       # legacy arXiv id
  python src/add.py https://arxiv.org/abs/...  # arXiv URL
  python src/add.py https://example.com/x.pdf  # direct PDF URL
  python src/add.py path/to/local.pdf          # local file
"""
import argparse
import asyncio
import json
import re
import sys
import urllib.parse
import urllib.request
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


USER_AGENT = "paper-qa-local (https://github.com/Lazy-Beee/paper-qa-local)"
HTTP_TIMEOUT = 30
MAX_TITLE_LEN = 60
MAX_ID_LEN = 40

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_OLD_RE = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)


def _sanitize(text: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("._-")
    return cleaned[:max_len]


def _http_get(url: str, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept},
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


def _http_get_json(url: str) -> dict:
    return json.loads(_http_get(url, accept="application/json"))


def _arxiv_id_from_url(url: str) -> str | None:
    m = re.search(
        r"arxiv\.org/(?:abs|pdf)/([\w./\-]+?)(?:v\d+)?(?:\.pdf)?(?:[?#]|$)",
        url,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def _doi_from_url(url: str) -> str | None:
    if "doi.org/" in url.lower():
        path = urllib.parse.urlsplit(url).path.lstrip("/")
        if DOI_RE.match(path):
            return path
    return None


def classify(target: str) -> tuple[str, str]:
    """Return ``(kind, identifier)``. kind is one of: file, arxiv, doi, url."""
    if Path(target).exists():
        return "file", str(Path(target).resolve())
    if target.lower().startswith("arxiv:"):
        return "arxiv", target.split(":", 1)[1]
    if ARXIV_NEW_RE.match(target) or ARXIV_OLD_RE.match(target):
        return "arxiv", target
    if DOI_RE.match(target):
        return "doi", target
    if target.lower().startswith(("http://", "https://")):
        aid = _arxiv_id_from_url(target)
        if aid:
            return "arxiv", aid
        doi = _doi_from_url(target)
        if doi:
            return "doi", doi
        return "url", target
    raise ValueError(
        f"Could not classify input: {target!r}. "
        "Expected DOI, arXiv id, http(s) URL, or existing file path."
    )


def fetch_arxiv(arxiv_id: str) -> tuple[bytes, dict]:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  Downloading PDF: {pdf_url}")
    pdf = _http_get(pdf_url, accept="application/pdf")

    meta = {"source": "arxiv", "arxiv_id": arxiv_id, "pdf_source": pdf_url}
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        atom = _http_get(api_url).decode("utf-8", errors="replace")
        entry_match = re.search(r"<entry>(.*?)</entry>", atom, re.DOTALL)
        entry = entry_match.group(1) if entry_match else atom
        title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        authors = re.findall(r"<name>(.*?)</name>", entry)
        published = re.search(r"<published>(\d{4})", entry)
        if title:
            meta["title"] = re.sub(r"\s+", " ", title.group(1)).strip()
        if authors:
            meta["authors"] = authors
        if published:
            meta["year"] = published.group(1)
    except Exception as e:
        print(f"  (arXiv metadata fetch failed: {e})")
    return pdf, meta


def fetch_doi(doi: str, unpaywall_email: str) -> tuple[bytes, dict]:
    quoted = urllib.parse.quote(doi, safe="/")
    meta = {"source": "doi", "doi": doi}

    print(f"  Resolving DOI via CrossRef: {doi}")
    try:
        cr = _http_get_json(f"https://api.crossref.org/works/{quoted}")
        msg = cr.get("message", {})
        titles = msg.get("title") or []
        if titles:
            meta["title"] = titles[0]
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in msg.get("author", [])
        ]
        if authors:
            meta["authors"] = authors
        date_parts = (msg.get("issued", {}).get("date-parts") or [[]])[0]
        if date_parts:
            meta["year"] = str(date_parts[0])
        journals = msg.get("container-title") or []
        if journals:
            meta["journal"] = journals[0]
    except Exception as e:
        print(f"  (CrossRef lookup failed: {e})")

    print("  Looking up open-access PDF via Unpaywall...")
    try:
        up_url = (
            f"https://api.unpaywall.org/v2/{quoted}"
            f"?email={urllib.parse.quote(unpaywall_email)}"
        )
        up = _http_get_json(up_url)
        loc = up.get("best_oa_location") or {}
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if not pdf_url:
            raise RuntimeError("no open-access PDF available")
        print(f"  Downloading PDF: {pdf_url}")
        pdf = _http_get(pdf_url, accept="application/pdf")
        meta["pdf_source"] = pdf_url
        return pdf, meta
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch an open-access PDF for DOI {doi}: {e}. "
            "If you have the PDF locally, run:\n"
            "  python src/add.py path/to/file.pdf"
        ) from e


def fetch_url(url: str) -> tuple[bytes, dict]:
    print(f"  Downloading: {url}")
    pdf = _http_get(url, accept="application/pdf,*/*")
    return pdf, {"source": "url", "pdf_source": url}


def fetch_local(path: str) -> tuple[bytes, dict]:
    print(f"  Reading local file: {path}")
    pdf = Path(path).read_bytes()
    return pdf, {"source": "file", "original_path": path}


def make_filename(kind: str, identifier: str, meta: dict) -> str:
    if kind == "file":
        return Path(identifier).name
    title_part = _sanitize(meta.get("title", ""), max_len=MAX_TITLE_LEN)
    if kind == "arxiv":
        id_part = "arxiv_" + _sanitize(identifier, max_len=MAX_ID_LEN)
    elif kind == "doi":
        id_part = _sanitize(identifier, max_len=MAX_ID_LEN)
    else:
        last = urllib.parse.urlsplit(identifier).path.rsplit("/", 1)[-1]
        id_part = _sanitize(last or "url", max_len=MAX_ID_LEN)
    base = f"{title_part}__{id_part}" if title_part else id_part
    return f"{base}.pdf"


def add_paper(
    target: str,
    paper_dir: Path,
    unpaywall_email: str,
    overwrite: bool = False,
) -> Path:
    kind, identifier = classify(target)
    print(f"Detected: kind={kind}, id={identifier}")

    fetcher = {
        "arxiv": lambda: fetch_arxiv(identifier),
        "doi": lambda: fetch_doi(identifier, unpaywall_email),
        "url": lambda: fetch_url(identifier),
        "file": lambda: fetch_local(identifier),
    }[kind]
    pdf, meta = fetcher()

    if not pdf.startswith(b"%PDF"):
        raise RuntimeError(
            "Downloaded content does not look like a PDF "
            f"(first bytes: {pdf[:32]!r}). Aborting."
        )

    paper_dir.mkdir(parents=True, exist_ok=True)
    filename = make_filename(kind, identifier, meta)
    out_path = paper_dir / filename
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass --overwrite to replace it."
        )

    out_path.write_bytes(pdf)
    print(f"Saved: {out_path}  ({len(pdf):,} bytes)")

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Metadata: {meta_path}")
    return out_path


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
        description=(
            "Add a paper to the library by DOI, arXiv id, URL, or local file, "
            "then update the index."
        ),
    )
    parser.add_argument(
        "target",
        help="DOI (10.x/y), arXiv id (2401.12345 or arxiv:...), URL, or local PDF path.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Save the PDF only; skip the incremental index update.",
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
    unpaywall_email = (
        cfg.get("network", {}).get("unpaywall_email") or "anonymous@example.com"
    )

    add_paper(
        args.target,
        paper_dir=paper_dir,
        unpaywall_email=unpaywall_email,
        overwrite=args.overwrite,
    )

    if args.no_index:
        print("\nSkipping index update (--no-index). Run build_index.py later.")
        return

    health_check()
    settings = make_settings(rebuild_index=False)
    asyncio.run(run_index(settings))


if __name__ == "__main__":
    main()
