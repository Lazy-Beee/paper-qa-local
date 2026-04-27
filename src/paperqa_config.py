"""Load config.toml and build a paper-qa Settings object."""
import datetime
import json
import logging
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from paperqa import Settings
from paperqa.settings import (
    AgentSettings,
    AnswerSettings,
    IndexSettings,
    ParsingSettings,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config.example.toml"
LOG_DIR = PROJECT_ROOT / "log"

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class _Tee:
    """Forward writes to multiple streams (e.g. terminal + log file).

    The first stream receives raw data so an interactive terminal still
    shows ANSI colors. Subsequent streams (log files) receive data with
    ANSI escape sequences stripped, keeping log files readable.

    Unknown attributes (isatty, fileno, encoding, ...) fall through to the
    first stream so libraries that introspect the stream still work
    (e.g. Rich detects TTY via the terminal stream and emits colors,
    which we then strip on the way to the file).
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        if not self.streams:
            return
        self.streams[0].write(data)
        self.streams[0].flush()
        if len(self.streams) > 1:
            stripped = ANSI_ESCAPE_RE.sub("", data)
            for s in self.streams[1:]:
                s.write(stripped)
                s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


class _DropPatternFilter(logging.Filter):
    """Drop log records whose message contains any of the given substrings."""

    def __init__(self, *needles: str) -> None:
        super().__init__()
        self._needles = needles

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(n in msg for n in self._needles)


def _silence_litellm_noise() -> None:
    """Suppress repetitive litellm warnings that have no actionable content."""
    logging.getLogger("LiteLLM").addFilter(
        _DropPatternFilter("MAX_CALLBACKS")
    )


def health_check() -> None:
    """Verify the LLM endpoint is reachable and required models are loaded.

    Raises RuntimeError with an actionable message on any problem.
    """
    cfg = load_config()
    llm = cfg["llm"]
    url = f"{llm['api_base'].rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach LLM endpoint at {url} ({e.reason}). "
            "Is LM Studio (or your OpenAI-compatible server) running?"
        ) from e

    available = {m["id"] for m in data.get("data", [])}
    needed = {llm["chat_model"], llm["embedding_model"]}
    rr = cfg.get("reranker") or {}
    if rr.get("enabled") and rr.get("model"):
        needed.add(rr["model"])
    missing = needed - available
    if missing:
        raise RuntimeError(
            f"Endpoint {url} is up but missing model(s): {sorted(missing)}. "
            f"Available: {sorted(available)}. "
            "Load the missing model(s) in LM Studio or fix config.toml."
        )
    print(f"Health check OK: {url}")


def setup_run_log(prefix: str) -> Path:
    """Create log/<prefix>_<timestamp>.log and tee stdout/stderr to it.

    Also installs a logging filter that drops noisy litellm callback
    warnings. The filter is attached by logger name, so it remains
    effective regardless of when litellm imports its verbose_logger.
    """
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOG_DIR / f"{prefix}_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    _silence_litellm_noise()
    return log_path


def _ensure_config_exists() -> None:
    """If config.toml is missing, seed it from config.example.toml.

    Lets a fresh checkout (where config.toml is gitignored) just work
    instead of crashing on first run.
    """
    if CONFIG_PATH.exists():
        return
    if not CONFIG_EXAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Neither {CONFIG_PATH} nor {CONFIG_EXAMPLE_PATH} exists. "
            "Restore one of them to proceed."
        )
    CONFIG_PATH.write_bytes(CONFIG_EXAMPLE_PATH.read_bytes())
    print(f"Created {CONFIG_PATH.name} from {CONFIG_EXAMPLE_PATH.name} (edit to customize).")


def load_config() -> dict:
    _ensure_config_exists()
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def _resolve_path(p: str) -> str:
    """Resolve a path from config: relative paths are anchored at PROJECT_ROOT."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


_RERANK_PATCHED = False


def _install_reranker_patch(cfg: dict) -> None:
    """Monkey-patch Docs.retrieve_texts to oversample then cross-encoder rerank.

    Idempotent: only patches once per process. Reads the current [reranker]
    section every call so config changes (enable/disable, oversample) take
    effect on the next retrieval without a restart.
    """
    global _RERANK_PATCHED

    from paperqa.docs import Docs

    from reranker import RerankerConfig, rerank_scores

    def _build_cfg() -> RerankerConfig | None:
        rr = (load_config().get("reranker") or {})
        if not rr.get("enabled"):
            return None
        llm = load_config()["llm"]
        return RerankerConfig(
            enabled=True,
            model=rr["model"],
            api_base=llm["api_base"],
            api_key=llm["api_key"],
            oversample=int(rr.get("oversample", 3)),
            max_concurrent_requests=int(rr.get("max_concurrent_requests", 8)),
            instruct=rr.get("instruct", "Given a search query, retrieve relevant passages."),
        )

    if _RERANK_PATCHED:
        return

    original = Docs.retrieve_texts

    async def patched(self, query, k, settings=None, embedding_model=None,
                      partitioning_fn=None):
        rcfg = _build_cfg()
        if rcfg is None or rcfg.oversample <= 1:
            return await original(self, query, k, settings, embedding_model,
                                  partitioning_fn)

        fetch_k = max(k, k * rcfg.oversample)
        candidates = await original(self, query, fetch_k, settings,
                                    embedding_model, partitioning_fn)
        if len(candidates) <= k:
            return candidates

        docs = [c.text for c in candidates]
        try:
            scores = await rerank_scores(rcfg, query, docs)
        except Exception as e:
            print(f"Reranker failed ({e}); falling back to embedding order.")
            return candidates[:k]

        # Stable sort by -score, preserving original embedding order on ties.
        indexed = sorted(
            range(len(candidates)),
            key=lambda i: (-scores[i], i),
        )
        kept = [candidates[i] for i in indexed[:k]]
        kept_yes = sum(1 for i in indexed[:k] if scores[i] >= 1.0)
        print(f"Reranker: {len(candidates)} candidates → kept {len(kept)} "
              f"(yes={kept_yes}, model={rcfg.model})")
        return kept

    Docs.retrieve_texts = patched
    _RERANK_PATCHED = True
    print("Reranker patch installed on paperqa.docs.Docs.retrieve_texts")


def make_settings(rebuild_index: bool) -> Settings:
    cfg = load_config()
    llm = cfg["llm"]
    paths = cfg["paths"]
    index_cfg = cfg["index"]
    answer_cfg = cfg["answer"]
    parsing_cfg = cfg["parsing"]
    _install_reranker_patch(cfg)

    os.environ["OPENAI_API_KEY"] = llm["api_key"]
    os.environ["OPENAI_BASE_URL"] = llm["api_base"]

    chat_model_id = f"openai/{llm['chat_model']}"
    chat_litellm_params = {
        "model": chat_model_id,
        "api_base": llm["api_base"],
        "api_key": llm["api_key"],
    }
    llm_router_config = {
        "model_list": [{
            "model_name": llm["chat_model"],
            "litellm_params": chat_litellm_params,
        }]
    }
    embed_router_config = {
        "model_list": [{
            "model_name": llm["embedding_alias"],
            "litellm_params": {
                "model": f"openai/{llm['embedding_model']}",
                "api_base": llm["api_base"],
                "api_key": llm["api_key"],
            },
        }]
    }

    return Settings(
        llm=chat_model_id,
        summary_llm=chat_model_id,
        embedding=llm["embedding_alias"],
        llm_config=llm_router_config,
        summary_llm_config=llm_router_config,
        embedding_config=embed_router_config,
        agent=AgentSettings(
            agent_llm=chat_model_id,
            agent_llm_config=llm_router_config,
            rebuild_index=rebuild_index,
            index=IndexSettings(
                paper_directory=_resolve_path(paths["paper_dir"]),
                index_directory=_resolve_path(paths["index_dir"]),
                recurse_subdirectories=index_cfg["recurse_subdirectories"],
            ),
        ),
        answer=AnswerSettings(
            evidence_k=answer_cfg["evidence_k"],
            answer_max_sources=answer_cfg["answer_max_sources"],
            max_concurrent_requests=answer_cfg["max_concurrent_requests"],
        ),
        parsing=ParsingSettings(
            multimodal=parsing_cfg["multimodal"],
            use_doc_details=parsing_cfg["use_doc_details"],
            disable_doc_valid_check=parsing_cfg["disable_doc_valid_check"],
            enrichment_llm=chat_model_id,
            enrichment_llm_config=llm_router_config,
        ),
    )
