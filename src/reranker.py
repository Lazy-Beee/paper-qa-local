"""Qwen3-style cross-encoder reranker over an OpenAI-compatible endpoint.

The Qwen3-Reranker family (e.g. ``qwen3-reranker-0.6b``) is a generative
yes/no judge: given a query and a candidate passage, the model is prompted
with a fixed template and emits the literal token ``Yes`` or ``No``.

LM Studio (and most OpenAI-compatible servers) do not return token
log-probabilities for these models, so we fall back to a binary score:
``1.0`` for ``Yes``, ``0.0`` for ``No``. The original embedding-retrieval
order is preserved as a tiebreaker by the caller.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: {instruct}\n"
    "<Query>: {query}\n"
    "<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)

# Cap each document at this many characters to keep the rerank prompt
# bounded. Qwen3-reranker is trained on snippet-sized inputs.
_DOC_CHAR_LIMIT = 4000


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool
    model: str
    api_base: str
    api_key: str
    oversample: int
    max_concurrent_requests: int
    instruct: str


def _truncate(s: str, limit: int = _DOC_CHAR_LIMIT) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


async def _score_one(
    client: httpx.AsyncClient,
    cfg: RerankerConfig,
    query: str,
    document: str,
    sem: asyncio.Semaphore,
) -> float:
    prompt = _PROMPT_TEMPLATE.format(
        instruct=cfg.instruct,
        query=query,
        document=_truncate(document),
    )
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "max_tokens": 2,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    async with sem:
        try:
            r = await client.post(
                f"{cfg.api_base.rstrip('/')}/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("Reranker call failed (%s) — defaulting score to 0.0", e)
            return 0.0
    text = (data.get("choices") or [{}])[0].get("text", "")
    answer = text.strip().lower()
    if answer.startswith("yes"):
        return 1.0
    if answer.startswith("no"):
        return 0.0
    logger.debug("Reranker emitted unexpected token %r — treating as 0.0", text)
    return 0.0


async def rerank_scores(
    cfg: RerankerConfig,
    query: str,
    documents: list[str],
) -> list[float]:
    """Score each document against the query (1.0 = relevant, 0.0 = not)."""
    if not documents:
        return []
    sem = asyncio.Semaphore(max(1, cfg.max_concurrent_requests))
    async with httpx.AsyncClient() as client:
        tasks = [_score_one(client, cfg, query, doc, sem) for doc in documents]
        return await asyncio.gather(*tasks)
