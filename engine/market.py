"""
engine/market.py
================
Self-contained infrastructure for dynamic auction analysis.
No imports from the rest of the engine — fully decoupled.

Public API:
  MarketAnalyzer   — tracks live price history; computes market efficiency signals
  SquadBuilder     — greedy squad completion; target priority scoring
  SaleRecord       — one completed transaction
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ── Sale record ───────────────────────────────────────────────────────────────
@dataclass
class SaleRecord:
    player_name: str
    role: str
    tier: int
    projected_points: float
    vorp: float
    price: float
    buyer: str
    vorp_fair_price: float  # 0.5 + vorp * cr_per_vorp at time of sale


# ── MarketAnalyzer ────────────────────────────────────────────────────────────
class MarketAnalyzer:
    """
    Tracks live auction history to derive three market signals:

    market_premium_ratio
        mean(price / vorp_fair_price) over last WINDOW sales.
        > 1.3 → field is overbidding (sit back)
        < 0.9 → buyer's market (go aggressive)

    role_cliff(role, remaining)
        pts gap between #1 and #2 remaining players in this role.
        Large cliff → losing #1 is catastrophic → should overpay.

    role_depth(role, remaining, threshold)
        Count of remaining players in `role` above `threshold` pts.
        Low depth → scarcity premium warranted.
    """

    WINDOW = 10  # rolling window for premium ratio

    def __init__(self):
        self._sales: list[SaleRecord] = []
        self._recent: deque[SaleRecord] = deque(maxlen=self.WINDOW)

    # ── Ingest ────────────────────────────────────────────────────────────────
    def record_sale(
        self,
        player: dict,
        price: float,
        buyer: str,
        cr_per_vorp: float,
    ) -> None:
        vorp = player.get("vorp", 0.0)
        fair = 0.5 + vorp * cr_per_vorp
        rec = SaleRecord(
            player_name=player["player_name"],
            role=player["role"],
            tier=player.get("tier", 3),
            projected_points=player["projected_points"],
            vorp=vorp,
            price=price,
            buyer=buyer,
            vorp_fair_price=max(0.5, fair),
        )
        self._sales.append(rec)
        self._recent.append(rec)

    # ── Signals ───────────────────────────────────────────────────────────────
    @property
    def market_premium_ratio(self) -> float:
        """Rolling mean of price / vorp_fair over last WINDOW sales."""
        if not self._recent:
            return 1.0
        ratios = [r.price / r.vorp_fair_price for r in self._recent if r.vorp_fair_price > 0]
        return sum(ratios) / len(ratios) if ratios else 1.0

    @property
    def market_mode(self) -> str:
        """'overbid' | 'fair' | 'buyer'"""
        r = self.market_premium_ratio
        if r > 1.3:
            return "overbid"
        if r < 0.9:
            return "buyer"
        return "fair"

    def role_cliff(self, role: str, remaining: list[dict]) -> float:
        """
        Pts gap between the best and second-best remaining player in `role`.
        Returns 0 if fewer than 2 players remain in this role.
        """
        pts = sorted(
            (p["projected_points"] for p in remaining if p["role"] == role),
            reverse=True,
        )
        if len(pts) < 2:
            return 0.0
        return pts[0] - pts[1]

    def role_depth(
        self,
        role: str,
        remaining: list[dict],
        threshold: float = 60.0,
    ) -> int:
        """Count of remaining players in `role` above `threshold` pts."""
        return sum(
            1 for p in remaining
            if p["role"] == role and p["projected_points"] >= threshold
        )

    def premium_ratio_for_role(self, role: str) -> float:
        """Mean premium ratio restricted to a given role (last 6 sales)."""
        recent_role = [r for r in reversed(self._sales) if r.role == role][:6]
        if not recent_role:
            return self.market_premium_ratio
        ratios = [r.price / r.vorp_fair_price for r in recent_role if r.vorp_fair_price > 0]
        return sum(ratios) / len(ratios) if ratios else 1.0

    def overdue_roles(self, remaining: list[dict]) -> list[str]:
        """
        Roles with depth <= 2 above threshold — signal that scarcity is real,
        not just depletion noise.
        """
        return [
            role for role in ["BAT", "BOWL", "AR", "WK"]
            if self.role_depth(role, remaining) <= 2
        ]


# ── SquadBuilder ─────────────────────────────────────────────────────────────
class SquadBuilder:
    """
    Greedy squad completion model.

    Answers two questions:
      1. target_priority_score(player, roster, remaining, budget, slots)
             → 0.0–1.0  how important is this player to optimal completion?
      2. best_completion_pts(roster, remaining, budget, slots)
             → float  expected season_score if we greedily fill remaining slots
    """

    # Role slot requirements used for completion scoring.
    # Real fantasy rules: minimum 1 of each type.
    # Higher minimums inflated role_need and distorted target_priority_score.
    ROLE_MIN = {"BAT": 1, "BOWL": 1, "AR": 1, "WK": 1}
    QUALITY_THRESHOLD = 55.0   # minimum pts to count as a "real" contribution

    # ── Target priority ───────────────────────────────────────────────────────
    def target_priority_score(
        self,
        player: dict,
        roster: list[dict],
        remaining: list[dict],
        budget: float,
        slots: int,
    ) -> float:
        """
        0.0 = skip (blocks a better option)
        0.5 = useful but not critical
        1.0 = must target

        Logic:
          - Players that appear in nearly all greedy completions → high priority
          - Players for whom there is no viable alternative in remaining pool → high priority
          - Players whose role is already well-filled → low priority
          - Players whose pts < QUALITY_THRESHOLD when better options exist → low priority
        """
        if slots <= 0:
            return 0.0

        role = player["role"]
        role_counts = _count_roles(roster)
        already_have = role_counts.get(role, 0)
        role_need = max(0, self.ROLE_MIN.get(role, 2) - already_have)

        # How many alternatives exist that are at least 85% as good?
        alternatives = [
            p for p in remaining
            if p["role"] == role
            and p["player_name"] != player["player_name"]
            and p["projected_points"] >= player["projected_points"] * 0.85
        ]

        # Cliff: how much better is this player than next-best?
        pool_peers = sorted(
            [p for p in remaining if p["role"] == role],
            key=lambda p: p["projected_points"],
            reverse=True,
        )
        cliff = 0.0
        if pool_peers and pool_peers[0]["player_name"] == player["player_name"] and len(pool_peers) >= 2:
            cliff = (player["projected_points"] - pool_peers[1]["projected_points"]) / max(1, player["projected_points"])

        # Compute score
        score = 0.5  # neutral baseline

        # Boost: role genuinely needed
        if role_need > 0:
            score += 0.2

        # Boost: very few alternatives
        if len(alternatives) <= 1:
            score += 0.25
        elif len(alternatives) <= 3:
            score += 0.12

        # Boost: significant cliff (best-in-class)
        if cliff > 0.15:
            score += 0.15
        elif cliff > 0.08:
            score += 0.08

        # Penalise: role fully covered and player is T3
        if role_need == 0 and player.get("tier", 3) == 3:
            score -= 0.3

        # Penalise: player below quality threshold when alternatives exist
        if (player["projected_points"] < self.QUALITY_THRESHOLD
                and len(alternatives) >= 3):
            score -= 0.25

        # Penalise: can't afford reasonable alternatives (preserve budget signals)
        est_price = player.get("base_price", 1.0) * 2.2
        if budget < est_price and slots > 3:
            score -= 0.1

        # C/VC team synergy: the scoring model awards Captain/VC bonuses per match
        # for the top-2 players from a team. The 2nd player from a team is most
        # valuable (creates C/VC pair); the 3rd+ are diminishing returns.
        team = player.get("ipl_team")
        if team:
            team_count = sum(1 for p in roster if p.get("ipl_team") == team)
            if team_count == 1:
                score += 0.18   # completing C/VC pair — high value
            elif team_count == 0:
                score += 0.06   # first player from this team — small speculative boost
            else:
                score -= 0.10   # 3rd+ from same team: C/VC already covered, penalise concentration

        return max(0.0, min(1.0, score))

    # ── Completion model ──────────────────────────────────────────────────────
    def best_completion_pts(
        self,
        roster: list[dict],
        remaining: list[dict],
        budget: float,
        slots: int,
    ) -> float:
        """
        Greedy fill: at each slot, pick the highest projected_points player
        we can afford (at estimated fair price) that helps meet role minimums first.
        Returns total projected_points of completed squad.
        """
        if slots <= 0:
            return sum(p["projected_points"] for p in roster)

        role_counts = dict(_count_roles(roster))
        current_pts = sum(p["projected_points"] for p in roster)
        remaining_sorted = sorted(remaining, key=lambda p: p["projected_points"], reverse=True)
        available = list(remaining_sorted)
        purse = budget

        for _ in range(slots):
            if not available or purse <= 0:
                break

            # First pass: try to meet role minimums
            needed_roles = {
                r for r, mn in self.ROLE_MIN.items()
                if role_counts.get(r, 0) < mn
            }

            candidate = None
            for p in available:
                est = p.get("base_price", 1.0) * 1.8
                if purse < est:
                    continue
                if needed_roles and p["role"] not in needed_roles:
                    continue
                candidate = p
                break

            # If no role-specific candidate, take best affordable
            if not candidate:
                for p in available:
                    est = p.get("base_price", 1.0) * 1.8
                    if purse >= est:
                        candidate = p
                        break

            if not candidate:
                break

            available.remove(candidate)
            current_pts += candidate["projected_points"]
            purse -= candidate.get("base_price", 1.0) * 1.8
            role_counts[candidate["role"]] = role_counts.get(candidate["role"], 0) + 1

        return current_pts

    # ── Relative standing ─────────────────────────────────────────────────────
    def relative_standing(
        self,
        my_roster: list[dict],
        my_budget: float,
        my_slots: int,
        remaining: list[dict],
        opponent_states: dict,  # name -> {roster, purse, slots}
    ) -> float:
        """
        my_projected / mean(opponent_projected)
        > 1.12  → ahead
        0.88–1.12 → parity
        < 0.88  → behind
        """
        my_proj = self.best_completion_pts(my_roster, remaining, my_budget, my_slots)

        opp_projs = []
        for opp in opponent_states.values():
            opp_roster = opp.get("roster_players", [])
            opp_proj = self.best_completion_pts(
                opp_roster, remaining, opp["purse"], opp["slots"]
            )
            opp_projs.append(opp_proj)

        if not opp_projs or sum(opp_projs) == 0:
            return 1.0
        return my_proj / (sum(opp_projs) / len(opp_projs))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _count_roles(roster: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in roster:
        r = p["role"]
        counts[r] = counts.get(r, 0) + 1
    return counts
