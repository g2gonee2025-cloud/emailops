"""
Reranking Logic (Lightweight & Cross-Encoder).

Implements §8.7 of the Canonical Blueprint.
"""

import asyncio
import ipaddress
import json
import logging
from urllib.parse import urlparse

import httpx
from cortex.config.loader import get_config
from cortex.retrieval.results import SearchResultItem

logger = logging.getLogger(__name__)

# Constants
RERANK_TRUNCATION_CHARS = 4096
RERANK_TIMEOUT_SECONDS = 15.0
MAX_RERANK_RESPONSE_BYTES = 1_000_000
MAX_MMR_CANDIDATES = 200


def _safe_score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_unit(value: object, name: str, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        logger.warning("%s is invalid; defaulting to %.2f", name, default)
        return default
    if 0.0 <= numeric <= 1.0:
        return numeric
    logger.warning("%s %.3f is out of range; clamping to [0, 1]", name, numeric)
    return min(max(numeric, 0.0), 1.0)


def _candidate_summary_text(item: SearchResultItem) -> str:
    """
    Format chunk with metadata for the Cross-Encoder.

    Instead of reranking just "Meeting at 5pm", we rerank:
    "From: Alice | Date: 2024-01-01 | Subject: Sync | Content: Meeting at 5pm"

    This helps the model resolve navigational intent (dates, people) within the content.
    """
    meta = item.metadata if isinstance(item.metadata, dict) else {}
    parts = []

    # Extract metadata fields (normalized by ingestion)
    sender = meta.get("sender") or meta.get("from")
    date = meta.get("date") or meta.get("sent_at")
    subject = meta.get("subject")

    if sender:
        parts.append(f"From: {sender}")
    if date:
        parts.append(f"Date: {date}")
    if subject:
        parts.append(f"Subject: {subject}")

    # Content is the most important part
    content = item.content or item.snippet or ""
    parts.append(f"Content: {content}")

    return " | ".join(parts)


def rerank_results(
    results: list[SearchResultItem],
    alpha: float,
) -> list[SearchResultItem]:
    """
    Apply lightweight reranker blending lexical/vector evidence.
    Used when external reranker is disabled.
    """
    if not results:
        return results

    alpha = _normalize_unit(alpha, "alpha", default=0.5)

    max_lex = max((item.lexical_score or 0.0 for item in results), default=0.0)
    max_vec = max((item.vector_score or 0.0 for item in results), default=0.0)

    for item in results:
        if not isinstance(getattr(item, "metadata", None), dict):
            item.metadata = {}
        lex_component = (item.lexical_score or 0.0) / max_lex if max_lex > 0 else 0.0
        vec_component = (item.vector_score or 0.0) / max_vec if max_vec > 0 else 0.0
        rerank_score = alpha * vec_component + (1 - alpha) * lex_component
        item.rerank_score = rerank_score
        base_score = _safe_score(item.score)
        item.score = 0.5 * base_score + 0.5 * rerank_score
        item.metadata["rerank_score"] = rerank_score

    return sorted(results, key=lambda itm: itm.score, reverse=True)


async def call_external_reranker(
    endpoint: str, query: str, results: list[SearchResultItem], top_n: int = 50
) -> list[SearchResultItem]:
    """
    Call external reranker API (vLLM with mxbai-rerank-large-v2).
    """
    if not results:
        return []

    if top_n is None:
        top_n = len(results)
    if top_n <= 0:
        logger.warning("top_n must be positive; skipping external rerank")
        return results

    config = get_config()
    allowed_endpoint = getattr(config.search, "reranker_endpoint", None)
    if not allowed_endpoint:
        raise ValueError("Reranker endpoint not configured")
    if _normalize_endpoint(endpoint) != _normalize_endpoint(str(allowed_endpoint)):
        raise ValueError("Reranker endpoint is not allowed")

    # Ensure metadata is a dict for all items to avoid None-safety issues
    for item in results:
        if getattr(item, "metadata", None) is None or not isinstance(
            item.metadata, dict
        ):
            item.metadata = {}

    # Rerank only the top N candidates to save time/compute
    candidates = results[:top_n]

    # PREPARE DOCUMENTS WITH SUMMARY CONTEXT
    # We truncate content to ensure low latency
    documents = [
        _candidate_summary_text(c)[:RERANK_TRUNCATION_CHARS] for c in candidates
    ]

    try:
        # Use /v1/rerank endpoint (Jina/Cohere compatible)
        rerank_url = endpoint.rstrip("/")
        if not rerank_url.endswith("/v1/rerank"):
            rerank_url = f"{rerank_url}/v1/rerank"

        allowed_host = urlparse(str(allowed_endpoint)).hostname
        if not allowed_host:
            raise ValueError("Invalid reranker endpoint host")

        await _validate_reranker_url(rerank_url, {allowed_host})

        async with httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
        ) as client:
            resp = await client.post(
                rerank_url,
                json={
                    "model": "mixedbread-ai/mxbai-rerank-large-v2",
                    "query": query,
                    "documents": documents,
                },
                timeout=RERANK_TIMEOUT_SECONDS,  # Allow time for cross-encoder inference
            )
            resp.raise_for_status()

        # Response is {"results": [{"index": int, "relevance_score": float}, ...]}
        content_length = resp.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_RERANK_RESPONSE_BYTES:
                    raise ValueError("Reranker response too large")
            except ValueError:
                raise ValueError("Invalid reranker response length") from None
        content = await resp.aread()
        if len(content) > MAX_RERANK_RESPONSE_BYTES:
            raise ValueError("Reranker response too large")

        response_data = json.loads(content)
        if not isinstance(response_data, dict):
            raise ValueError("Invalid reranker response format")
        rankings = response_data.get("results", [])
        if not isinstance(rankings, list):
            raise ValueError("Invalid reranker response format")

        reranked = []
        for r in rankings:
            if not isinstance(r, dict):
                continue
            idx = r.get("index")
            if idx is None:
                continue
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if idx < 0:
                continue
            if idx >= len(candidates):
                continue
            item = candidates[idx]

            # Update score with the high-fidelity cross-encoder score
            rerank_score = _safe_score(r.get("relevance_score", r.get("score", 0.0)))
            item.rerank_score = rerank_score
            item.score = rerank_score
            item.metadata["rerank_score"] = rerank_score
            item.metadata["rerank_model"] = "mxbai-rerank-large-v2"
            reranked.append(item)

        # Sort by rerank score descending
        reranked.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return reranked

    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        logger.exception("External reranker failed (falling back to fusion scores)")
        return results


def _normalize_endpoint(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    scheme = (parsed.scheme or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    return f"{scheme}://{host}{port}"


async def _validate_reranker_url(url: str, allowed_hosts: set[str]) -> None:
    """
    Validate the reranker URL to prevent SSRF vulnerabilities.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            raise ValueError("Reranker endpoint must use HTTPS")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Missing hostname")
        if hostname not in allowed_hosts:
            raise ValueError("Reranker host is not allowlisted")

        loop = asyncio.get_running_loop()
        addr_info = await loop.getaddrinfo(hostname, None)
        if not addr_info:
            raise ValueError("Unable to resolve hostname")

        for _family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if not ip.is_global:
                raise ValueError("Resolved IP is not global")

    except Exception as e:
        logger.error(
            "Invalid reranker URL for host '%s': %s",
            urlparse(url).hostname,
            e,
        )
        raise ValueError("Invalid reranker URL") from e


def _tokenize_for_similarity(text: str) -> set[str]:
    tokens = [t.lower() for t in text.split() if t]
    return set(tokens)


def _text_similarity(a: SearchResultItem, b: SearchResultItem) -> float:
    """Compute simple Jaccard similarity between two result snippets/contents."""
    text_a = a.content or a.snippet or ""
    text_b = b.content or b.snippet or ""
    if not text_a or not text_b:
        return 0.0
    tokens_a = _tokenize_for_similarity(text_a[:RERANK_TRUNCATION_CHARS])
    tokens_b = _tokenize_for_similarity(text_b[:RERANK_TRUNCATION_CHARS])
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def apply_mmr(
    results: list[SearchResultItem],
    lambda_param: float,
    limit: int | None = None,
) -> list[SearchResultItem]:
    """Apply Maximal Marginal Relevance to diversify top results."""

    if not results:
        return results

    lambda_param = _normalize_unit(lambda_param, "lambda_param", default=0.5)
    if limit is None:
        limit = len(results)
    elif limit <= 0:
        logger.warning("limit must be positive; skipping MMR")
        return results

    if len(results) > MAX_MMR_CANDIDATES:
        candidates_sorted = sorted(results, key=lambda r: r.score or 0.0, reverse=True)
        candidates = candidates_sorted[:MAX_MMR_CANDIDATES]
        remainder = candidates_sorted[MAX_MMR_CANDIDATES:]
    else:
        candidates = results.copy()
        remainder = []

    limit = min(limit, len(candidates))
    selected: list[SearchResultItem] = []

    while candidates and len(selected) < limit:
        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(candidates):
            relevance = _safe_score(candidate.score)
            if selected:
                redundancy = max(
                    _text_similarity(candidate, chosen) for chosen in selected
                )
            else:
                redundancy = 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(candidates.pop(best_idx))

    # Append any remaining candidates to preserve ordering info
    selected.extend(candidates)
    selected.extend(remainder)
    return selected
