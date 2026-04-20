"""Tests for $SORTIE_ROLE gating of MCP tool registration.

The server reads ``SORTIE_ROLE`` **once at import time** because FastMCP
registers tools eagerly via decorators. Tests therefore run each role in
a fresh subprocess with the env var set, then assert on the captured
tool set.

A subprocess is the only reliable way — you cannot re-import
``sortie_mcp.server`` in-process because the module cache (and the
module-level ``mcp`` FastMCP singleton) would already be populated by
the default ``both`` mode from earlier tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def _run_role(role: str) -> set[str]:
    """Spawn a subprocess with ``SORTIE_ROLE=role`` and return the tool names."""
    code = textwrap.dedent(
        """
        import json, os, sys
        from sortie_mcp.server import mcp

        # Mirror the reflection logic in test_server.py::_tool_names
        def _tool_names():
            if hasattr(mcp, "_tool_manager"):
                tm = mcp._tool_manager
                if hasattr(tm, "_tools"):
                    return list(tm._tools.keys())
                if hasattr(tm, "tools"):
                    return list(tm.tools.keys())
            if hasattr(mcp, "_tools"):
                return list(mcp._tools.keys())
            return []

        print(json.dumps(sorted(_tool_names())))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**__import__("os").environ, "SORTIE_ROLE": role},
        capture_output=True,
        check=True,
        text=True,
    )
    return set(json.loads(out.stdout.strip()))


COORD_TOOLS = {
    "check_success",
    "create_campaign",
    "list_campaigns",
    "get_campaign",
    "get_updates",
    "steer_campaign",
    "pause_campaign",
    "resume_campaign",
    "provide_input",
    "cancel_campaign",
}

WORKER_TOOLS = {
    "get_my_context",
    "add_note",
    "search_notes",
    "get_notes",
    "complete_step",
    "fail_step",
    "request_input",
    "spawn_and_continue",
    "abort_branch",
    "read_step_output",
    "heartbeat",
}


class TestRoleGating:
    def test_coordinator_role_hides_worker_tools(self) -> None:
        tools = _run_role("coordinator")
        assert tools >= COORD_TOOLS, f"missing coordinator tools: {COORD_TOOLS - tools}"
        # None of the worker tools are exposed on the coordinator server.
        leaked = tools & WORKER_TOOLS
        assert not leaked, f"worker tools leaked to coordinator role: {leaked}"

    def test_worker_role_hides_coordinator_tools(self) -> None:
        tools = _run_role("worker")
        assert tools >= WORKER_TOOLS, f"missing worker tools: {WORKER_TOOLS - tools}"
        leaked = tools & COORD_TOOLS
        assert not leaked, f"coordinator tools leaked to worker role: {leaked}"

    def test_both_role_exposes_everything(self) -> None:
        tools = _run_role("both")
        assert tools >= COORD_TOOLS | WORKER_TOOLS

    def test_invalid_role_falls_back_to_both(self) -> None:
        tools = _run_role("bogus-role")
        assert tools >= COORD_TOOLS | WORKER_TOOLS
