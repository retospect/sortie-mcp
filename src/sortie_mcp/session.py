"""Session binding — environment-derived defaults for worker agents.

When the :mod:`sortie_mcp.runner` dispatches a step to an OpenClaw agent,
it spawns (or RPCs to) a process that will in turn talk to this MCP
server. To avoid making the LLM restate its own ``step_id`` on every
tool call (pure token waste, and a failure mode if the model hallucinates
a different id), the runner seeds a handful of environment variables
that every tool can fall back on:

================================  ==============================================
``SORTIE_STEP_ID``                The integer step id this agent owns.
``SORTIE_CLAIM_TOKEN``            UUID issued by the runner at claim time.
``SORTIE_CAMPAIGN_ID``            UUID of the parent campaign (convenience).
``SORTIE_ROLE``                   ``coordinator`` / ``worker`` / ``both``.
================================  ==============================================

The values are read lazily on each call (not cached at import) so that
tests can monkey-patch ``os.environ`` without restarting the server.

Security note: the claim_token is only ever passed DOWN from trusted
sources (the runner). Agents never see it or construct it themselves —
they merely forward it to the DB so the DB can verify that an auto-
complete coming from a resurrected zombie is rejected.
"""

from __future__ import annotations

import os
from typing import Literal, cast
from uuid import UUID

Role = Literal["coordinator", "worker", "both"]

_VALID_ROLES: frozenset[str] = frozenset({"coordinator", "worker", "both"})


def session_step_id() -> int | None:
    """Return ``$SORTIE_STEP_ID`` as int, or ``None`` if unset/invalid."""
    raw = os.environ.get("SORTIE_STEP_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def session_claim_token() -> UUID | None:
    """Return ``$SORTIE_CLAIM_TOKEN`` as UUID, or ``None`` if unset/invalid."""
    raw = os.environ.get("SORTIE_CLAIM_TOKEN")
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def session_campaign_id() -> UUID | None:
    """Return ``$SORTIE_CAMPAIGN_ID`` as UUID, or ``None`` if unset/invalid."""
    raw = os.environ.get("SORTIE_CAMPAIGN_ID")
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def session_role() -> Role:
    """Return the role this MCP server instance is serving.

    Default is ``both`` — backwards compatible with pre-v0.2 servers
    where every tool was available to every client. The runner should
    set ``SORTIE_ROLE=worker`` on agent subprocesses and
    ``SORTIE_ROLE=coordinator`` on Asa's server.
    """
    raw = (os.environ.get("SORTIE_ROLE") or "both").strip().lower()
    if raw not in _VALID_ROLES:
        return "both"
    return cast(Role, raw)


def resolve_step_id(explicit: int | None) -> int | None:
    """Pick explicit step_id if provided, else session default."""
    return explicit if explicit is not None else session_step_id()
