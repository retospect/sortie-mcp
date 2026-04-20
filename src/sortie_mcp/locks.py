"""Lock helpers — owner identity, key vocabulary, conflict matrix.

Used by the multi-runner claim layer (migration 0002) and the resource
lease layer (migration 0003).
"""

from __future__ import annotations

import os
import socket
from enum import StrEnum

# ---------------------------------------------------------------------------
# Owner identity — used for ``campaign_steps.claim_owner`` and
# ``resource_leases.owner``. Format: ``<host>/<runner-pid>``.
# ---------------------------------------------------------------------------


def default_owner() -> str:
    """Return a stable owner string for this process.

    Honours ``SORTIE_OWNER`` if set (useful for tests and Ansible
    deploys that want to embed the cluster role). Falls back to
    ``<hostname>/runner-pid-<pid>``.
    """
    explicit = os.environ.get("SORTIE_OWNER")
    if explicit:
        return explicit
    host = socket.gethostname().split(".")[0]  # short hostname
    return f"{host}/runner-pid-{os.getpid()}"


# ---------------------------------------------------------------------------
# Lock key vocabulary
# ---------------------------------------------------------------------------

# Hierarchical separator. ``§`` chosen because it is unambiguous in URIs
# and never appears in real filesystem paths or DOIs.
KEY_SEP = "§"


def make_lock_key(kind: str, path: str, slug: str | None = None) -> str:
    """Compose a structured lock key.

    Examples:
        >>> make_lock_key("file", "content/books/nanobuds/ch01.tex")
        'file:content/books/nanobuds/ch01.tex'
        >>> make_lock_key("file", "content/books/nanobuds/ch01.tex", "PLXDX")
        'file:content/books/nanobuds/ch01.tex§PLXDX'
        >>> make_lock_key("campaign", "abc-uuid", "strategy")
        'campaign:abc-uuid§strategy'
    """
    if KEY_SEP in kind or ":" in kind:
        raise ValueError(f"kind may not contain ':' or {KEY_SEP!r}: {kind!r}")
    base = f"{kind}:{path}"
    return f"{base}{KEY_SEP}{slug}" if slug else base


def key_parent(key: str) -> str | None:
    """Return the immediate parent of a hierarchical key, or ``None``.

    >>> key_parent("file:a/b.tex§PLXDX")
    'file:a/b.tex'
    >>> key_parent("file:a/b.tex")  # already at top
    """
    if KEY_SEP not in key:
        return None
    return key.rsplit(KEY_SEP, 1)[0]


def key_ancestors(key: str) -> list[str]:
    """Return the list of ancestor keys, root-most first, excluding self.

    >>> key_ancestors("file:a/b.tex§sec1§PLXDX")
    ['file:a/b.tex', 'file:a/b.tex§sec1']
    """
    if KEY_SEP not in key:
        return []
    parts = key.split(KEY_SEP)
    return [KEY_SEP.join(parts[: i + 1]) for i in range(len(parts) - 1)]


def key_is_descendant_of(child: str, ancestor: str) -> bool:
    """True if ``child`` is a strict descendant of ``ancestor``."""
    return child.startswith(ancestor + KEY_SEP)


# ---------------------------------------------------------------------------
# Lease modes
# ---------------------------------------------------------------------------


class LockMode(StrEnum):
    """Resource lease access mode."""

    EXCLUSIVE = "exclusive"
    SHARED = "shared"


# Conflict matrix — does a held lease block a new request?
#
#   held \ requested  | EXCL | SHARED
#   EXCLUSIVE         | yes  | yes
#   SHARED            | yes  | no
def lease_conflicts(held_mode: LockMode, requested_mode: LockMode) -> bool:
    """Return True if a held lease blocks a new request on the same key."""
    if held_mode is LockMode.EXCLUSIVE:
        return True
    # held SHARED: blocks only EXCLUSIVE requests
    return requested_mode is LockMode.EXCLUSIVE


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------


STALE_CLAIM = "stale_claim"
"""Status returned by ``complete_step`` / ``fail_step`` / ``request_input``
when the caller's ``claim_token`` does not match the current row.

The caller should NOT retry — its claim has been revoked (e.g. by
``reset_zombies``) and another runner has likely re-claimed the step."""
