"""Tests for sortie_mcp.scheduler — WDRR picker and fairness invariants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sortie_mcp.models import (
    PRIORITY_WEIGHTS,
    Campaign,
    CampaignStatus,
    Priority,
    priority_weight,
)
from sortie_mcp.scheduler import pick_next_campaign, weight_for

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def mkc(
    *,
    slot_seconds_used: float = 0.0,
    priority: Priority = Priority.NORMAL,
    next_action_at: datetime | None = None,
    created_at: datetime | None = None,
    weight: float | None = None,
    cid: UUID | None = None,
) -> Campaign:
    """Make a Campaign with controlled fair-share state.

    ``weight`` defaults from ``priority_weight(priority)`` — mirroring
    the production code path where ``DB.create_campaign`` pins it."""
    return Campaign(
        id=cid or uuid4(),
        name="c",
        goal="g",
        status=CampaignStatus.ACTIVE,
        priority=priority,
        slot_seconds_used=slot_seconds_used,
        weight=weight if weight is not None else priority_weight(priority),
        next_action_at=next_action_at or _now(),
        created_at=created_at or _now(),
    )


# ---------------------------------------------------------------------------
# Priority → weight mapping
# ---------------------------------------------------------------------------


class TestPriorityWeights:
    def test_mapping_matches_migration(self) -> None:
        """The Python mapping MUST match the SQL CASE in
        ``migrations/0004.fair-share.sql`` — otherwise new-vs-old rows
        drift."""
        assert PRIORITY_WEIGHTS[Priority.URGENT] == 8.0
        assert PRIORITY_WEIGHTS[Priority.HIGH] == 4.0
        assert PRIORITY_WEIGHTS[Priority.NORMAL] == 2.0
        assert PRIORITY_WEIGHTS[Priority.LOW] == 1.0
        assert PRIORITY_WEIGHTS[Priority.BACKGROUND] == 0.5

    def test_priority_weight_helper(self) -> None:
        assert priority_weight(Priority.URGENT) == 8.0
        assert priority_weight(Priority.BACKGROUND) == 0.5

    def test_weight_for_reexport(self) -> None:
        """``scheduler.weight_for`` is a convenience alias for
        ``models.priority_weight``; must return identical values."""
        for p in Priority:
            assert weight_for(p) == priority_weight(p)


class TestVirtualTime:
    def test_fresh_campaign_has_zero_vt(self) -> None:
        assert mkc().virtual_time == 0.0

    def test_vt_scales_inversely_with_weight(self) -> None:
        urgent = mkc(slot_seconds_used=8, priority=Priority.URGENT)  # 8/8 = 1
        normal = mkc(slot_seconds_used=2, priority=Priority.NORMAL)  # 2/2 = 1
        bg = mkc(slot_seconds_used=0.5, priority=Priority.BACKGROUND)  # 0.5/0.5 = 1
        # All three spent enough to have vt=1 — invariant is that each
        # spent a slot-second scaled to its weight.
        assert urgent.virtual_time == 1.0
        assert normal.virtual_time == 1.0
        assert bg.virtual_time == 1.0

    def test_zero_weight_does_not_divide_by_zero(self) -> None:
        """A badly-migrated row with ``weight=0`` must not crash the
        picker. It yields a huge vt (starved forever) — acceptable
        since such a row is a deployment bug."""
        broken = mkc(slot_seconds_used=1, weight=0.0)
        assert broken.virtual_time > 1e5


# ---------------------------------------------------------------------------
# Core picker behaviour
# ---------------------------------------------------------------------------


class TestPickNextCampaign:
    def test_empty_returns_none(self) -> None:
        assert pick_next_campaign([]) is None

    def test_single_candidate_is_picked(self) -> None:
        c = mkc()
        assert pick_next_campaign([c]) is c

    def test_lowest_vt_wins(self) -> None:
        """Fairness core: whoever has lowest slot_seconds/weight goes
        next, independent of priority."""
        hungry = mkc(slot_seconds_used=0, priority=Priority.BACKGROUND)
        greedy = mkc(slot_seconds_used=100, priority=Priority.URGENT)
        # hungry.vt = 0/0.5 = 0; greedy.vt = 100/8 = 12.5
        assert pick_next_campaign([greedy, hungry]) is hungry

    def test_exclude_is_honoured(self) -> None:
        """The lock-busy retry path skips a campaign for the remainder
        of a tick. Picker must respect the exclude set."""
        a = mkc(slot_seconds_used=0)
        b = mkc(slot_seconds_used=10)
        assert pick_next_campaign([a, b], exclude={a.id}) is b

    def test_exclude_all_returns_none(self) -> None:
        a = mkc()
        b = mkc()
        assert pick_next_campaign([a, b], exclude={a.id, b.id}) is None

    def test_tiebreak_by_next_action_at(self) -> None:
        """Two fresh campaigns — older next_action_at wins so cron-
        triggered work doesn't get pushed around by newly-woken rows."""
        older = mkc(next_action_at=_now() - timedelta(minutes=5))
        newer = mkc(next_action_at=_now())
        assert pick_next_campaign([newer, older]) is older

    def test_tiebreak_by_priority_when_vt_and_next_action_identical(self) -> None:
        """When vt and next_action_at match, priority tier breaks the
        tie — URGENT beats BACKGROUND on a fresh start."""
        t = _now()
        urgent = mkc(priority=Priority.URGENT, next_action_at=t, created_at=t)
        # Match the URGENT weight so their virtual_time is identical.
        bg = mkc(
            priority=Priority.BACKGROUND,
            weight=priority_weight(Priority.URGENT),
            next_action_at=t,
            created_at=t,
        )
        # Both have vt=0, same next_action_at → URGENT tier ordinal is lower
        assert pick_next_campaign([bg, urgent]) is urgent

    def test_tiebreak_by_created_at_as_final_deterministic_key(self) -> None:
        """Two campaigns with everything equal — oldest created_at wins
        to make the picker deterministic across ticks."""
        t = _now()
        older = mkc(created_at=t - timedelta(hours=1), next_action_at=t)
        newer = mkc(created_at=t, next_action_at=t)
        assert pick_next_campaign([newer, older]) is older


# ---------------------------------------------------------------------------
# Multi-tick fairness simulation
# ---------------------------------------------------------------------------


class TestFairnessSimulation:
    """These run the picker in a loop mimicking the runner's dispatch
    cycle and assert invariants about the distribution."""

    def test_four_equal_campaigns_get_balanced_slots(self) -> None:
        """With 4 normal-priority campaigns and 100 slots, each should
        get ~25 slots (allowing small integer rounding). This is the
        baseline fairness property from the plan."""
        campaigns = [mkc(priority=Priority.NORMAL) for _ in range(4)]
        counts = {c.id: 0 for c in campaigns}

        for _ in range(100):
            pick = pick_next_campaign(campaigns)
            assert pick is not None
            counts[pick.id] += 1
            # Charge 1 slot-second.
            pick.slot_seconds_used += 1

        # With identical weights, each should have gotten 25.
        for n in counts.values():
            assert 24 <= n <= 26, f"unbalanced: {counts}"

    def test_urgent_gets_more_slots_than_background(self) -> None:
        """8x weight -> roughly 8x slots over a long horizon."""
        urgent = mkc(priority=Priority.URGENT)
        bg = mkc(priority=Priority.BACKGROUND)
        counts = {urgent.id: 0, bg.id: 0}

        for _ in range(1000):
            pick = pick_next_campaign([urgent, bg])
            assert pick is not None
            counts[pick.id] += 1
            pick.slot_seconds_used += 1

        # urgent weight 8, bg weight 0.5 -> ratio 16:1
        ratio = counts[urgent.id] / max(counts[bg.id], 1)
        assert 12 <= ratio <= 20, f"expected ~16:1, got {ratio:.1f}:1"

    def test_greedy_background_cannot_starve_normal(self) -> None:
        """Even with a BACKGROUND campaign that already spent 1000
        slot-seconds, a fresh NORMAL campaign is served first (its vt=0
        vs the background's vt=2000)."""
        hungry_normal = mkc(slot_seconds_used=0, priority=Priority.NORMAL)
        greedy_bg = mkc(slot_seconds_used=1000, priority=Priority.BACKGROUND)
        # greedy_bg.vt = 1000/0.5 = 2000, hungry_normal.vt = 0
        assert pick_next_campaign([hungry_normal, greedy_bg]) is hungry_normal

    def test_new_urgent_campaign_joins_and_is_served(self) -> None:
        """Scenario: three normal campaigns have been running, then an
        urgent one arrives. The urgent should get served for many
        consecutive slots until its vt catches up."""
        normals = [
            mkc(slot_seconds_used=100, priority=Priority.NORMAL) for _ in range(3)
        ]
        # All normals are at vt = 100/2 = 50.
        urgent = mkc(slot_seconds_used=0, priority=Priority.URGENT)
        all_ = [*normals, urgent]

        urgent_wins = 0
        first_loss_slot = None
        for slot in range(500):
            pick = pick_next_campaign(all_)
            assert pick is not None
            if pick.id == urgent.id:
                urgent_wins += 1
            else:
                if first_loss_slot is None:
                    first_loss_slot = slot
            pick.slot_seconds_used += 1

        # Urgent wins the first batch: it has to spend 50*8 = 400
        # slot-seconds before its vt catches up to the normals at 50.
        # So urgent should win about the first ~400 slots.
        assert urgent_wins >= 390
        assert first_loss_slot is not None and first_loss_slot >= 390
