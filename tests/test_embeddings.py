"""Tests for sortie_mcp.embeddings — the LiteLLM embedding client + feature flag."""

from __future__ import annotations

import httpx
import pytest
import respx

from sortie_mcp.embeddings import (
    EmbeddingClient,
    embed_text,
    embedding_dim,
    embedding_model,
    embeddings_enabled,
    reset_client,
)


@pytest.fixture(autouse=True)
def _reset_embedding_singleton() -> None:
    """Drop the module-level client between tests so env-var monkeypatches
    actually take effect on fresh instances."""
    reset_client()
    yield
    reset_client()


# ---------------------------------------------------------------------------
# Feature flag parsing
# ---------------------------------------------------------------------------


class TestEmbeddingsEnabled:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " true "])
    def test_truthy(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("SORTIE_EMBEDDINGS_ENABLED", val)
        assert embeddings_enabled() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "banana"])
    def test_falsy(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("SORTIE_EMBEDDINGS_ENABLED", val)
        assert embeddings_enabled() is False

    def test_unset_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SORTIE_EMBEDDINGS_ENABLED", raising=False)
        assert embeddings_enabled() is False


class TestEmbeddingModel:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SORTIE_EMBEDDING_MODEL", raising=False)
        assert embedding_model() == "nomic-embed-text"

    def test_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_EMBEDDING_MODEL", "custom/model-v1")
        assert embedding_model() == "custom/model-v1"


class TestEmbeddingDim:
    def test_default_is_384(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SORTIE_EMBEDDING_DIM", raising=False)
        assert embedding_dim() == 384

    def test_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_EMBEDDING_DIM", "1024")
        assert embedding_dim() == 1024

    def test_invalid_falls_back_to_384(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_EMBEDDING_DIM", "not-a-number")
        assert embedding_dim() == 384


# ---------------------------------------------------------------------------
# EmbeddingClient — happy paths & failure modes (fail-open contract)
# ---------------------------------------------------------------------------


def _vec(n: int) -> list[float]:
    """Deterministic float vector of dim ``n``."""
    return [round(0.01 * i, 4) for i in range(n)]


class TestEmbeddingClientHappyPath:
    @respx.mock
    async def test_calls_litellm_and_returns_vector(self) -> None:
        route = respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": _vec(384), "index": 0}],
                    "model": "test-model",
                },
            )
        )
        client = EmbeddingClient(
            base_url="http://litellm.test",
            api_key="sk-test",
            model="test-model",
            expected_dim=384,
        )
        try:
            out = await client.embed("hello world")
        finally:
            await client.aclose()
        assert out is not None
        assert len(out) == 384
        # Auth header was forwarded.
        assert route.called
        headers = route.calls[0].request.headers
        assert headers["authorization"] == "Bearer sk-test"
        # Payload shape — parse the JSON rather than string-match so the
        # test is robust to whitespace differences in httpx serialization.
        import json as _json

        body = _json.loads(route.calls[0].request.content)
        assert body == {"model": "test-model", "input": "hello world"}

    @respx.mock
    async def test_no_auth_header_when_key_unset(self) -> None:
        route = respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"embedding": _vec(384)}]})
        )
        client = EmbeddingClient(
            base_url="http://litellm.test",
            api_key=None,
            expected_dim=384,
        )
        try:
            await client.embed("hi")
        finally:
            await client.aclose()
        assert "authorization" not in route.calls[0].request.headers


class TestEmbeddingClientFailOpen:
    async def test_empty_text_short_circuits(self) -> None:
        client = EmbeddingClient(base_url="http://litellm.test", expected_dim=384)
        try:
            assert await client.embed("") is None
            assert await client.embed("   ") is None
        finally:
            await client.aclose()

    async def test_no_base_url_returns_none(self) -> None:
        client = EmbeddingClient(base_url=None, expected_dim=384)
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()

    @respx.mock
    async def test_http_error_returns_none(self) -> None:
        respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(500, text="internal server error")
        )
        client = EmbeddingClient(base_url="http://litellm.test", expected_dim=384)
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()

    @respx.mock
    async def test_network_error_returns_none(self) -> None:
        respx.post("http://litellm.test/v1/embeddings").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        client = EmbeddingClient(base_url="http://litellm.test", expected_dim=384)
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()

    @respx.mock
    async def test_timeout_returns_none(self) -> None:
        respx.post("http://litellm.test/v1/embeddings").mock(
            side_effect=httpx.ReadTimeout("too slow")
        )
        client = EmbeddingClient(
            base_url="http://litellm.test", expected_dim=384, timeout=0.01
        )
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()

    @respx.mock
    async def test_malformed_payload_returns_none(self) -> None:
        respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"nope": "no data here"})
        )
        client = EmbeddingClient(base_url="http://litellm.test", expected_dim=384)
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()

    @respx.mock
    async def test_dim_mismatch_returns_none(self) -> None:
        """If LiteLLM returns 768 dims but the column wants 384, the DB
        insert would crash. Better to fail-open here so we log the
        deployment misconfig but still record the note."""
        respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"embedding": _vec(768)}]})
        )
        client = EmbeddingClient(base_url="http://litellm.test", expected_dim=384)
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()

    @respx.mock
    async def test_empty_vector_returns_none(self) -> None:
        respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"embedding": []}]})
        )
        client = EmbeddingClient(base_url="http://litellm.test", expected_dim=384)
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()

    @respx.mock
    async def test_non_numeric_values_return_none(self) -> None:
        bad = [0.1] * 383 + ["not-a-float"]  # type: ignore[list-item]
        respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"embedding": bad}]})
        )
        client = EmbeddingClient(base_url="http://litellm.test", expected_dim=384)
        try:
            assert await client.embed("hello") is None
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Module-level embed_text() — enforces the feature flag
# ---------------------------------------------------------------------------


class TestEmbedTextGate:
    async def test_disabled_returns_none_without_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the flag off, embed_text must NOT touch the network even
        if LITELLM_URL is set. This is the v0.1-compat guarantee."""
        monkeypatch.delenv("SORTIE_EMBEDDINGS_ENABLED", raising=False)
        monkeypatch.setenv("LITELLM_URL", "http://litellm.test")
        # Intentionally NOT starting respx — any HTTP call would raise.
        out = await embed_text("hello")
        assert out is None

    @respx.mock
    async def test_enabled_hits_the_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SORTIE_EMBEDDINGS_ENABLED", "1")
        monkeypatch.setenv("LITELLM_URL", "http://litellm.test")
        monkeypatch.setenv("SORTIE_EMBEDDING_DIM", "384")
        route = respx.post("http://litellm.test/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"embedding": _vec(384)}]})
        )
        out = await embed_text("hello")
        assert out is not None
        assert len(out) == 384
        assert route.called
