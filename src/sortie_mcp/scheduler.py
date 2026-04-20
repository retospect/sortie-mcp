"""Weighted-Deficit Round-Robin (WDRR) scheduler for sortie-mcp.

Replaces the v0.1 per-tier priority-fraction allocator with a single
picker that persists compute usage across ticks on
``campaigns.slot_seconds_used`` (migration 0004).

Design
------
For each campaign we compute a scalar *virtual time*::

    virtual_time = slot_seconds_used / weight

where ``weight`` is derived from ``priority`` (see
:data:`sortie_mcp.models.PRIORITY_WEIGHTS`). The picker serves the
candidate with the lowest virtual time on every slot. Because
``slot_seconds_used`` only ever goes up, a single greedy campaign's
virtual time climbs and it stops being picked until siblings catch up.

Properties
----------
- **New urgent campaign gets prompt service.** With ``weight=8`` and
  ``slot_seconds_used=0``, virtual time stays 0 for the first 8s of
  compute; meanwhile a background campaign (``weight=0.5``) hits vt=1.0
  after just 0.5s. Urgent dominates until it has been served ~16x more
  than background.
- **Greedy background can't monopolise.** Every completed step charges
  the campaign's ledger. A long-running background step accumulates
  vt fast (low weight → steep slope) and gets deprioritised.
- **Explicit and debuggable.** vt is a single float per campaign; the
  Discord `/sortie` command can render a leaderboard.
- **No per-tick memory required.** The picker is stateless between
  calls because the state lives in the DB column.

Non-goals
---------
- Aging / decay. If a campaign sits at vt=9999 forever, we don't auto-
  decrement. If the user cares, they can reset the ledger via
  ``UPDATE campaigns SET slot_seconds_used = 0``. The plan (§8) proposes
  an aging term only if monitoring shows a starvation pattern.
- Preemption. An in-flight step runs to completion regardless of new
  arrivals.
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import PRIORITY_WEIGHTS, Campaign, Priority

# Tiebreaker when virtual times are equal (e.g. two fresh campaigns).
# Higher-priority tier wins. Same order as
# :data:`sortie_mcp.models.PRIORITY_ORDER`.
_TIER_ORDINAL: dict[Priority, int] = {
    Priority.URGENT: 0,
    Priority.HIGH: 1,
    Priority.NORMAL: 2,
    Priority.LOW: 3,
    Priority.BACKGROUND: 4,
}


def pick_next_campaign(
    candidates: Iterable[Campaign],
    *,
    exclude: set | None = None,
) -> Campaign | None:
    """Return the campaign most entitled to the next slot, or ``None``.

    Args:
        candidates: Active campaigns returned by
            :meth:`sortie_mcp.db.DB.get_due_campaigns`. Callers typically
            pass the whole list; the picker does not filter by readiness
            itself (that's :meth:`DB.get_ready_steps`). The runner
            handles a picked campaign that turns out to be lock-busy by
            marking it "unavailable this tick" and picking again.
        exclude: Optional set of campaign IDs to skip — used by the
            runner to retry after a lock-busy result without a mid-tick
            DB round-trip.

    Returns:
        The chosen :class:`Campaign`, or ``None`` if the candidate list
        is empty (modulo ``exclude``).

    Sort key, in order:

    1. ``virtual_time = slot_seconds_used / max(weight, 1e-6)`` ASC
    2. ``next_action_at`` ASC (older first — cron-friendly)
    3. Priority tier ordinal ASC (URGENT before BACKGROUND on ties)
    4. ``created_at`` ASC (deterministic final tiebreaker)
    """
    skip = exclude or set()
    best: Campaign | None = None
    best_key: tuple[float, ...] | None = None

    for c in candidates:
        if c.id in skip:
            continue
        key = (
            c.virtual_time,
            # None sorts weird in tuples — coerce to a large sentinel so
            # rows with NULL next_action_at land at the end.
            (
                c.next_action_at.timestamp()
                if c.next_action_at is not None
                else float("inf")
            ),
            _TIER_ORDINAL.get(c.priority, _TIER_ORDINAL[Priority.NORMAL]),
            (c.created_at.timestamp() if c.created_at is not None else float("inf")),
        )
        if best_key is None or key < best_key:
            best = c
            best_key = key

    return best


def weight_for(priority: Priority) -> float:
    """Re-exported convenience; see :func:`sortie_mcp.models.priority_weight`."""
    return PRIORITY_WEIGHTS.get(priority, PRIORITY_WEIGHTS[Priority.NORMAL])
