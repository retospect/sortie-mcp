"""Tests for sortie_mcp.session — env-derived session bindings."""

from __future__ import annotations

from uuid import uuid4

import pytest

from sortie_mcp.session import (
    resolve_step_id,
    session_campaign_id,
    session_claim_token,
    session_role,
    session_step_id,
)


class TestSessionStepId:
    def test_returns_int_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_STEP_ID", "42")
        assert session_step_id() == 42

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SORTIE_STEP_ID", raising=False)
        assert session_step_id() is None

    def test_returns_none_when_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_STEP_ID", "not-an-int")
        assert session_step_id() is None


class TestSessionClaimToken:
    def test_returns_uuid_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tok = uuid4()
        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(tok))
        assert session_claim_token() == tok

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SORTIE_CLAIM_TOKEN", raising=False)
        assert session_claim_token() is None

    def test_returns_none_when_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", "not-a-uuid")
        assert session_claim_token() is None


class TestSessionCampaignId:
    def test_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cid = uuid4()
        monkeypatch.setenv("SORTIE_CAMPAIGN_ID", str(cid))
        assert session_campaign_id() == cid


class TestSessionRole:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("coordinator", "coordinator"),
            ("worker", "worker"),
            ("both", "both"),
            ("COORDINATOR", "coordinator"),  # case-insensitive
            (" worker ", "worker"),  # strip
        ],
    )
    def test_valid_roles(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
    ) -> None:
        monkeypatch.setenv("SORTIE_ROLE", raw)
        assert session_role() == expected

    def test_unset_defaults_to_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SORTIE_ROLE", raising=False)
        assert session_role() == "both"

    def test_invalid_defaults_to_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_ROLE", "bogus")
        assert session_role() == "both"


class TestResolveStepId:
    def test_explicit_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_STEP_ID", "999")
        assert resolve_step_id(42) == 42

    def test_env_used_when_explicit_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SORTIE_STEP_ID", "42")
        assert resolve_step_id(None) == 42

    def test_returns_none_when_neither_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SORTIE_STEP_ID", raising=False)
        assert resolve_step_id(None) is None
