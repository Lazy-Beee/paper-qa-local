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
    missing = needed - available
    if missing:
        raise RuntimeError(
            f"Endpoint {url} is up but missing model(s): {sorted(missing)}. "
            f"Available: {sorted(available)}. "
            "Load the missing model(s) in LM Studio or fix [llm] in config.toml."
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


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def _resolve_path(p: str) -> str:
    """Resolve a path from config: relative paths are anchored at PROJECT_ROOT."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def make_settings(rebuild_index: bool) -> Settings:
    cfg = load_config()
    llm = cfg["llm"]
    paths = cfg["paths"]
    index_cfg = cfg["index"]
    answer_cfg = cfg["answer"]
    parsing_cfg = cfg["parsing"]

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
