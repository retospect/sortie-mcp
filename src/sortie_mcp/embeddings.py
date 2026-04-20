"""LiteLLM-backed embedding client for semantic note search.

Sortie stores note embeddings in a ``vector(N)`` pgvector column so that
:func:`sortie_mcp.db.DB.search_notes` can rank by cosine distance. This
module is the *producer* side of that pipeline: given text, ask LiteLLM
for a float vector.

Feature-flag
------------
The client is strictly **opt-in** via ``$SORTIE_EMBEDDINGS_ENABLED``.
When unset (the default), every call short-circuits to ``None`` so the
tool surface works identically to v0.1 — notes get recorded without an
embedding, and ``search_notes`` falls back to recency-ordered listing.

This matters because:

1. The MCP server must start cleanly on hosts with no LiteLLM reachable
   (CI, dev laptops, integration tests).
2. An accidentally wrong ``SORTIE_EMBEDDING_MODEL`` would silently write
   vectors of the wrong dimension into pgvector — catastrophic. Opt-in
   forces an explicit deployment step.

Failure modes
-------------
Once enabled, the client is **fail-open**: a network error, timeout, or
dimension mismatch is logged at WARNING and yields ``None``. The caller
inserts the note anyway. This is deliberate — losing a semantic link is
cheap; losing a coordinator's finding because LiteLLM bounced is not.

Config
------
- ``SORTIE_EMBEDDINGS_ENABLED`` — ``1`` / ``true`` / ``yes`` to enable.
- ``LITELLM_URL`` — e.g. ``http://melchior:4000`` (no trailing slash).
- ``LITELLM_KEY`` — optional, sent as ``Authorization: Bearer``.
- ``SORTIE_EMBEDDING_MODEL`` — model id LiteLLM knows about. Default
  ``nomic-embed-text``. **Must match the pgvector column dimension.**
  The schema currently expects 384 — so configure LiteLLM to route this
  model name to a 384-dim provider (e.g. SentenceTransformers
  ``all-MiniLM-L6-v2``).
- ``SORTIE_EMBEDDING_DIM`` — expected dim, default 384. Mismatch → warn
  and return ``None`` (the DB insert would fail otherwise).
- ``SORTIE_EMBEDDING_TIMEOUT`` — HTTP timeout seconds, default 10.
"""

from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)


def embeddings_enabled() -> bool:
    """Return True iff the embedding pipeline should be active."""
    raw = (os.environ.get("SORTIE_EMBEDDINGS_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def embedding_model() -> str:
    """LiteLLM model name for the embedding endpoint."""
    return os.environ.get("SORTIE_EMBEDDING_MODEL") or "nomic-embed-text"


def embedding_dim() -> int:
    """Expected output dimension (must match pgvector column)."""
    raw = os.environ.get("SORTIE_EMBEDDING_DIM") or "384"
    try:
        return int(raw)
    except ValueError:
        return 384


def _litellm_url() -> str | None:
    url = (os.environ.get("LITELLM_URL") or "").rstrip("/")
    return url or None


def _litellm_key() -> str | None:
    return os.environ.get("LITELLM_KEY") or None


def _timeout_sec() -> float:
    raw = os.environ.get("SORTIE_EMBEDDING_TIMEOUT") or "10"
    try:
        return float(raw)
    except ValueError:
        return 10.0


class EmbeddingClient:
    """Async embedding client targeting LiteLLM's OpenAI-compatible API.

    Instances are cheap; share a single one per process so the underlying
    ``httpx.AsyncClient`` can pool connections to LiteLLM.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        expected_dim: int | None = None,
        timeout: float | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url if base_url is not None else _litellm_url()
        self._api_key = api_key if api_key is not None else _litellm_key()
        self._model = model or embedding_model()
        self._expected_dim = (
            expected_dim if expected_dim is not None else embedding_dim()
        )
        self._timeout = timeout if timeout is not None else _timeout_sec()
        self._http = http  # may be None — lazy-init on first call

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def embed(self, text: str) -> list[float] | None:
        """Return the embedding vector for ``text`` or ``None`` on failure.

        Fail-open: any error (LiteLLM unreachable, HTTP != 200, malformed
        payload, dimension mismatch) is logged at WARNING and returns
        ``None``. Callers must handle the ``None`` case.

        An empty / whitespace-only ``text`` short-circuits to ``None``
        without hitting the network.
        """
        if not text or not text.strip():
            return None
        if not self._base_url:
            log.warning(
                "sortie embeddings enabled but $LITELLM_URL is not set — "
                "skipping embedding for this note"
            )
            return None

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            http = await self._get_http()
            resp = await http.post(
                f"{self._base_url}/v1/embeddings",
                headers=headers,
                json={"model": self._model, "input": text},
            )
        except httpx.HTTPError as exc:  # network / timeout / DNS
            log.warning("LiteLLM embedding call failed: %s", exc)
            return None

        if resp.status_code != 200:
            log.warning(
                "LiteLLM embedding returned HTTP %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            return None

        try:
            payload = resp.json()
            vec = payload["data"][0]["embedding"]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            log.warning("LiteLLM embedding payload malformed: %s", exc)
            return None

        if not isinstance(vec, list) or not vec:
            log.warning("LiteLLM embedding returned empty vector")
            return None

        if len(vec) != self._expected_dim:
            log.warning(
                "LiteLLM embedding dim mismatch: got %d, expected %d "
                "(check SORTIE_EMBEDDING_MODEL / SORTIE_EMBEDDING_DIM)",
                len(vec),
                self._expected_dim,
            )
            return None

        # Coerce numeric types — LiteLLM providers occasionally return
        # ints or numpy types depending on the backend.
        try:
            return [float(x) for x in vec]
        except (TypeError, ValueError) as exc:
            log.warning("LiteLLM embedding contained non-numeric value: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Module-level singleton — shared by add_note / search_notes
# ---------------------------------------------------------------------------

_client: EmbeddingClient | None = None


def get_client() -> EmbeddingClient:
    """Return the process-wide embedding client."""
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


async def embed_text(text: str) -> list[float] | None:
    """Convenience wrapper: return an embedding if enabled, else ``None``.

    This is the single entry point the server layer should call.
    Feature-flag check is inside — callers never need to ask.
    """
    if not embeddings_enabled():
        return None
    return await get_client().embed(text)


def reset_client() -> None:
    """Drop the cached client. Tests use this between env-var changes."""
    global _client
    _client = None
