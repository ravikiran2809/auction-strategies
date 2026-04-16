"""
engine/strategies.py
=====================
All 9 bidding strategies with parameterised @dataclass configs.
Every magic number from full_simulation_claude.py is now a named field
with explicit (lo, hi) bounds used by the evolutionary optimizer.

Public API:
  STRATEGY_REGISTRY   dict[str, tuple[type[Manager], type]]
  Manager             base class, importable for custom strategies
  ParamBounds         helper returned by get_param_bounds()

Adding a new strategy:
  1. Define YourParams(StrategyParams) with @dataclass fields + bounds
  2. Define YourStrategy(Manager) implementing willingness_to_pay()
  3. Register:  STRATEGY_REGISTRY["YourStrategy"] = (YourStrategy, YourParams)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

_DEFAULT_PARAMS_PATH  = Path(__file__).parent.parent / "strategy_params.json"
_SCHEDULE_PATH        = Path(__file__).parent.parent / "ipl_2025_schedule.json"

POOL_SIZE = 130  # used by ValueInvestor urgency scaler


def _load_schedule() -> list[dict]:
    if _SCHEDULE_PATH.exists():
        with open(_SCHEDULE_PATH) as f:
            return json.load(f)
    return []


SCHEDULE: list[dict] = _load_schedule()


# ── Param bounds metadata ────────────────────────────────────────────────────
@dataclass
class ParamField:
    """Carries the (lo, hi) bounds alongside the default value."""
    default: float
    lo: float
    hi: float


# ── Base strategy params ─────────────────────────────────────────────────────
@dataclass
class StrategyParams:
    """Base class — subclasses add concrete fields."""

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return (lo, hi) per field in dataclass field order."""
        raise NotImplementedError("Subclass must define get_bounds()")

    def to_vector(self) -> list[float]:
        return [getattr(self, f.name) for f in fields(self) if isinstance(getattr(self, f.name), (int, float))]

    @classmethod
    def from_vector(cls, vec: list[float]) -> "StrategyParams":
        scalar_fields = [f for f in fields(cls) if f.type in ("float", "int", float, int)]
        kwargs = {f.name: v for f, v in zip(scalar_fields, vec)}
        return cls(**kwargs)


# ── Manager base class ───────────────────────────────────────────────────────
@dataclass
class Manager:
    """
    Base manager class. All 9 strategies inherit from this.
    Holds purse/roster state and shared helper methods.
    """
    name: str
    params: StrategyParams
    total_purse: float = 120.0
    min_roster: int = 13
    max_roster: int = 16
    purse: float = field(init=False)
    roster: list = field(init=False)
    role_counts: dict = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.purse = self.total_purse
        self.roster = []
        self.role_counts = {"BAT": 0, "BOWL": 0, "AR": 0, "WK": 0}

    @property
    def slots(self) -> int:
        return max(0, self.max_roster - len(self.roster))

    @property
    def mandatory(self) -> int:
        return max(0, self.min_roster - len(self.roster))

    @property
    def max_bid(self) -> float:
        # Reserve 1 Cr (T3 base price) per remaining mandatory slot except the current one
        return max(0.0, self.purse - 1.0 * max(0, self.mandatory - 1))

    @property
    def cash_per_slot(self) -> float:
        return self.purse / self.slots if self.slots > 0 else 0.0

    def buy(self, player: dict, price: float) -> None:
        self.roster.append({"player": player, "price": price})
        self.purse -= price
        self.role_counts[player["role"]] += 1

    def fire_sale(self) -> Optional[dict]:
        """Sell the most expensive player at 75% recovery when out of money."""
        if self.mandatory > 0 and self.max_bid <= 0.5 and self.roster:
            self.roster.sort(key=lambda x: x["price"], reverse=True)
            sold = self.roster.pop(0)
            self.purse += sold["price"] * 0.75
            self.role_counts[sold["player"]["role"]] -= 1
            return sold["player"]
        return None

    def season_score(self) -> float:
        """C/VC-aware fitness: base pts + Captain(2×)/VC(1.5×) bonus per match."""
        base = sum(e["player"]["projected_points"] for e in self.roster)
        if not SCHEDULE:
            return base
        cv_bonus = 0.0
        for match in SCHEDULE:
            eligible = sorted(
                [e["player"] for e in self.roster
                 if e["player"].get("ipl_team") in (match["team1"], match["team2"])],
                key=lambda p: p["projected_points"],
                reverse=True,
            )
            if len(eligible) >= 1:
                cv_bonus += eligible[0]["projected_points"] * 1.0  # Captain +1×
            if len(eligible) >= 2:
                cv_bonus += eligible[1]["projected_points"] * 0.5  # VC +0.5×
        return base + cv_bonus

    # ── Shared helpers ───────────────────────────────────────────────────────
    def _desperation(self, wtp: float, state: dict, rnd: int) -> float:
        scarcity = self.slots / max(1, state["players_remaining"])
        if scarcity > 0.15:
            wtp *= 1.0 + (scarcity * 3) ** 2
        if rnd > 1 and self.mandatory > 0:
            wtp = max(wtp, self.purse / max(1, self.slots))
        return min(wtp, self.max_bid)

    def _alt_mean(self, player: dict, state: dict) -> float:
        pool = state["pool_by_role"].get(player["role"], [])
        alts = pool[: max(1, 4 - self.role_counts[player["role"]])]
        return sum(p["projected_points"] for p in alts) / len(alts) if alts else 75.0

    def _premium_wtp(self, player: dict, state: dict, rnd: int, exponent: float = 1.6) -> float:
        premium = player["projected_points"] / max(1.0, self._alt_mean(player, state))
        return self._desperation(self.cash_per_slot * (premium ** exponent), state, rnd)

    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        raise NotImplementedError

    def bid(self, player: dict, state: dict, rnd: int) -> float:
        """Called by auction.py. Applies team diversity premium on top of WTP."""
        wtp = self.willingness_to_pay(player, state, rnd)
        return self._apply_team_premium(wtp, player)

    def _apply_team_premium(self, wtp: float, player: dict) -> float:
        """Boosts WTP for new teams; discounts 3rd+ player from same franchise."""
        premium = getattr(self.params, "new_team_premium", 1.0)
        if premium == 1.0:
            return wtp
        team = player.get("ipl_team")
        if not team:
            return wtp
        my_teams = {e["player"].get("ipl_team") for e in self.roster}
        if team not in my_teams:
            return min(wtp * premium, self.max_bid)
        team_count = sum(1 for e in self.roster if e["player"].get("ipl_team") == team)
        if team_count >= 3:
            return wtp * max(0.5, 2.0 - premium)  # discount when oversaturated
        return wtp

    def _team_spread(self) -> int:
        """Number of unique IPL franchises in current roster."""
        return len({e["player"].get("ipl_team") for e in self.roster
                    if e["player"].get("ipl_team")})

    def _is_new_team(self, player: dict) -> bool:
        """True if this player's IPL team is not yet represented in roster."""
        my_teams = {e["player"].get("ipl_team") for e in self.roster}
        return player.get("ipl_team") not in my_teams

    def _cv_saturation(self, player: dict) -> bool:
        """True if roster already has 3+ players from this player's IPL team."""
        team = player.get("ipl_team")
        if not team:
            return False
        return sum(1 for e in self.roster if e["player"].get("ipl_team") == team) >= 3
        return (
            f"{self.name:<30} | {len(self.roster):>2}/{self.max_roster}"
            f" | ₹{self.purse:>5.1f}Cr | {self.season_score():>6.0f}pts"
        )


# ═══════════════════════════════════════════════════════════
# STRATEGY 0 — StarChaser
# ═══════════════════════════════════════════════════════════
@dataclass
class StarChaserParams(StrategyParams):
    elite_cap: float = 0.35          # max fraction of total_purse for first elite
    t1_solo_mult: float = 2.4        # cash_per_slot multiplier for first T1
    t1_extra_mult: float = 1.2       # cash_per_slot multiplier for subsequent T1
    t2_mult: float = 1.0             # cash_per_slot multiplier for T2
    depth_mult: float = 0.55         # cash_per_slot multiplier for T3
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.20, 0.60), (1.5, 4.0), (0.8, 2.0), (0.7, 1.5), (0.3, 0.8), (0.8, 2.0)]


class StarChaser(Manager):
    """Stars & Scrubs: one elite player, depth everywhere else."""

    params: StarChaserParams

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        tier = player.get("tier", 2)
        own_t1 = sum(1 for e in self.roster if e["player"].get("tier") == 1)
        cap = self.total_purse * p.elite_cap
        if tier == 1 and own_t1 == 0:
            wtp = min(cap, self.cash_per_slot * p.t1_solo_mult)
        elif tier == 1:
            wtp = self.cash_per_slot * p.t1_extra_mult
        elif tier == 2:
            wtp = self.cash_per_slot * p.t2_mult
        else:
            wtp = self.cash_per_slot * p.depth_mult
        return self._desperation(wtp, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 1 — ValueInvestor
# ═══════════════════════════════════════════════════════════
@dataclass
class ValueInvestorParams(StrategyParams):
    urgency_min: float = 0.85
    urgency_max: float = 1.35
    fair_base: float = 0.5           # base cost offset in ₹Cr
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.5, 1.0), (1.0, 2.0), (0.3, 1.0), (0.8, 2.0)]


class ValueInvestor(Manager):
    """VORP-based fair price with urgency scaler — guarantees full purse deployment."""

    params: ValueInvestorParams

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        fair = p.fair_base + player["vorp"] * state["cr_per_vorp"]
        cash_ratio = self.purse / max(1.0, self.total_purse)
        pool_ratio = state["players_remaining"] / max(1, POOL_SIZE)
        urgency = max(p.urgency_min, min(p.urgency_max, 1.0 + (cash_ratio - pool_ratio)))
        return self._desperation(fair * urgency, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 2 — DynamicMaximizer
# ═══════════════════════════════════════════════════════════
@dataclass
class DynamicMaximizerParams(StrategyParams):
    base_exp: float = 1.4            # exponent at 4 agents
    exp_per_agent: float = 0.07      # exponent increase per additional agent
    exp_cap: float = 2.0             # maximum exponent
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(1.0, 2.0), (0.02, 0.15), (1.5, 3.0), (0.8, 2.0)]


class DynamicMaximizer(Manager):
    """Exponential premium scaler. Exponent grows with field size."""

    params: DynamicMaximizerParams

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        n_agents = len(state["agent_states"])
        exponent = min(p.exp_cap, p.base_exp + (n_agents - 4) * p.exp_per_agent)
        return self._premium_wtp(player, state, rnd, exponent)


# ═══════════════════════════════════════════════════════════
# STRATEGY 3 — ApexPredator
# ═══════════════════════════════════════════════════════════
@dataclass
class ApexPredatorParams(StrategyParams):
    wc_discount: float = 0.90        # winner's-curse discount (Milgrom/Thaler)
    efficiency_exp: float = 1.5      # p_eff/a_eff ratio exponent
    adj_min: float = 0.8             # wallet-leverage clamp min
    adj_max: float = 2.0             # wallet-leverage clamp max
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.70, 1.00), (1.0, 2.5), (0.5, 1.2), (1.5, 3.0), (0.8, 2.0)]


class ApexPredator(Manager):
    """Efficiency-ratio bidder with winner's-curse discount."""

    params: ApexPredatorParams

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        cper = state["cr_per_vorp"]
        exp = 0.5 + player["vorp"] * cper
        pool = state["pool_by_role"].get(player["role"], [])
        alts = pool[: max(1, 4 - self.role_counts[player["role"]])]
        if not alts:
            return self._desperation(0.5, state, rnd)
        alt_pts = sum(a["projected_points"] for a in alts) / len(alts)
        alt_cost = 0.5 + max(0, alt_pts - 40) * cper
        p_eff = player["projected_points"] / max(0.5, exp)
        a_eff = alt_pts / max(0.5, alt_cost)
        ratio = (p_eff / max(0.01, a_eff)) ** p.efficiency_exp
        adj = min(p.adj_max, max(p.adj_min, self.cash_per_slot / max(1.0, exp)))
        return self._desperation(exp * ratio * adj * p.wc_discount, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 4 — BarbellStrategist
# ═══════════════════════════════════════════════════════════
@dataclass
class BarbellStrategistParams(StrategyParams):
    hi_cost_threshold: float = 0.15  # cost_impact > this → penalise volatility
    lo_cost_threshold: float = 0.05  # cost_impact < this → reward upside
    hi_penalty_mult: float = 0.15    # std_dev multiplier for expensive picks (season avg dominates)
    lo_reward_mult: float = 0.15     # std_dev multiplier for cheap picks
    mid_penalty_mult: float = 0.05   # std_dev multiplier for mid-range picks
    base_exp: float = 1.6            # _premium_wtp exponent
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.05, 0.30), (0.01, 0.10), (0.5, 2.0), (0.5, 2.0), (0.0, 1.0), (1.0, 2.5), (0.8, 2.0)]


class BarbellStrategist(Manager):
    """Risk-adjusted bidder: pays less for volatile stars, more for cheap high-upside."""

    params: BarbellStrategistParams

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        exp_price = 0.5 + player["vorp"] * state["cr_per_vorp"]
        ci = exp_price / max(1.0, self.purse)
        std = player.get("std_dev", 50)
        if ci > p.hi_cost_threshold:
            adj_pts = player["projected_points"] - std * p.hi_penalty_mult
        elif ci < p.lo_cost_threshold:
            adj_pts = player["projected_points"] + std * p.lo_reward_mult
        else:
            adj_pts = player["projected_points"] - std * p.mid_penalty_mult
        copy = {**player, "projected_points": max(10.0, adj_pts)}
        return self._premium_wtp(copy, state, rnd, p.base_exp)


# ═══════════════════════════════════════════════════════════
# STRATEGY 5 — VampireMaximizer
# ═══════════════════════════════════════════════════════════
@dataclass
class VampireMaximizerParams(StrategyParams):
    shade_margin: float = 0.5        # ₹Cr above max_opp to win cleanly
    bluff_floor_ratio: float = 0.40  # bluff only if my_wtp > max_opp * floor
    bluff_cap_ratio: float = 0.60    # price-enforce to my_wtp / cap (60%)
    purse_healthy_pct: float = 0.40  # require purse > this fraction to bluff
    min_slots_to_bluff: int = 4      # require at least this many slots to bluff
    elite_pts_threshold: float = 90  # projected_points above this → elite opp mult
    elite_mult: float = 1.4          # opp cash_per_slot multiplier for elite players
    normal_mult: float = 0.9         # opp cash_per_slot multiplier for normal players
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.25, 1.5), (0.20, 0.60), (0.40, 0.80), (0.20, 0.60),
                (2.0, 7.0), (70.0, 120.0), (1.0, 2.0), (0.6, 1.2), (0.8, 2.0)]


class VampireMaximizer(Manager):
    """Price-enforcer: bluffs to drain opponent purses when it's profitable."""

    params: VampireMaximizerParams

    def _opp_wtp(self, player: dict, opp: dict) -> float:
        p = self.params
        if opp["slots"] == 0:
            return 0.0
        avg = opp["purse"] / max(1, opp["slots"])
        cap = opp["purse"] - 0.5 * max(0, opp["mandatory"] - 1)
        mult = p.elite_mult if player["projected_points"] > p.elite_pts_threshold else p.normal_mult
        return min(avg * mult, cap)

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        my_wtp = self._premium_wtp(player, state, rnd, 1.6)
        max_opp = max(
            (self._opp_wtp(player, o)
             for n, o in state["agent_states"].items() if n != self.name),
            default=0.0,
        )
        if my_wtp >= max_opp + p.shade_margin:
            return self._desperation(my_wtp, state, rnd)
        healthy = (self.purse / self.total_purse) > p.purse_healthy_pct and self.slots > p.min_slots_to_bluff
        if my_wtp < max_opp and my_wtp > max_opp * p.bluff_floor_ratio and healthy:
            return min(my_wtp / p.bluff_cap_ratio, self.max_bid)
        return self._desperation(my_wtp, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 6 — TierSniper
# ═══════════════════════════════════════════════════════════
@dataclass
class TierSniperParams(StrategyParams):
    t1_cap: float = 0.45                    # max fraction of total_purse for first T1
    t1_solo_mult: float = 2.8               # cash_per_slot multiplier for first T1
    t1_price_enforce_mult: float = 1.05     # multiplier when enforcing on last few T1s
    t1_sit_back_mult: float = 0.5           # multiplier when already have T1
    t1_enforce_when_remaining: int = 3      # enforce when ≤ this many T1s left
    t2_mult: float = 1.1
    depth_mult: float = 0.65
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.25, 0.65), (1.5, 4.0), (0.8, 1.5), (0.2, 0.8),
                (1.0, 6.0), (0.7, 1.5), (0.3, 1.0), (0.8, 2.0)]


class TierSniper(Manager):
    """Tier-lock: one elite player, then disciplined T2/T3 sweep."""

    params: TierSniperParams

    def _t1_remaining(self, state) -> int:
        return sum(1 for ps in state["pool_by_role"].values() for pl in ps if pl.get("tier") == 1)

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        tier = player.get("tier", 2)
        own_t1 = sum(1 for e in self.roster if e["player"].get("tier") == 1)
        t1_rem = self._t1_remaining(state)
        if tier == 1:
            if own_t1 == 0:
                wtp = min(self.total_purse * p.t1_cap, self.cash_per_slot * p.t1_solo_mult)
            elif t1_rem <= p.t1_enforce_when_remaining:
                wtp = self.cash_per_slot * p.t1_price_enforce_mult
            else:
                wtp = self.cash_per_slot * p.t1_sit_back_mult
        elif tier == 2:
            wtp = self.cash_per_slot * p.t2_mult
        else:
            wtp = self.cash_per_slot * p.depth_mult
        return self._desperation(wtp, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 7 — NominationGambler
# ═══════════════════════════════════════════════════════════
@dataclass
class NominationGamblerParams(StrategyParams):
    phase_a_threshold: int = 8       # T1 remaining ≥ this → Phase A (conserve)
    phase_b_threshold: int = 3       # T1 remaining ≥ this → Phase B (selective)
    phase_a_t1_mult: float = 1.5
    phase_a_other_mult: float = 0.25
    phase_b_t1_exp: float = 1.5
    phase_b_t2_mult: float = 0.9
    phase_b_depth_mult: float = 0.40
    phase_c_exp: float = 2.0
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(4.0, 12.0), (1.0, 5.0), (0.8, 2.5), (0.1, 0.5),
                (1.0, 2.5), (0.5, 1.5), (0.2, 0.7), (1.5, 3.0), (0.8, 2.0)]


class NominationGambler(Manager):
    """Phase-aware: conserve early, compete mid, liquidate late."""

    params: NominationGamblerParams

    def _t1_remaining(self, state) -> int:
        return sum(1 for ps in state["pool_by_role"].values() for pl in ps if pl.get("tier") == 1)

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        t1_rem = self._t1_remaining(state)
        tier = player.get("tier", 2)
        if t1_rem >= p.phase_a_threshold:
            wtp = self.cash_per_slot * (p.phase_a_t1_mult if tier == 1 else p.phase_a_other_mult)
        elif t1_rem >= p.phase_b_threshold:
            if tier == 1:
                wtp = self._premium_wtp(player, state, rnd, p.phase_b_t1_exp)
            elif tier == 2:
                wtp = self.cash_per_slot * p.phase_b_t2_mult
            else:
                wtp = self.cash_per_slot * p.phase_b_depth_mult
        else:
            wtp = self._premium_wtp(player, state, rnd, p.phase_c_exp)
        return self._desperation(wtp, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 8 — PositionalArbitrageur
# ═══════════════════════════════════════════════════════════
@dataclass
class PositionalArbitrageurParams(StrategyParams):
    target_bat: int = 5
    target_bowl: int = 4
    target_ar: int = 2
    target_wk: int = 2
    scarcity_cap: float = 2.5
    filled_discount: float = 0.7     # scarcity multiplier when role is filled
    scarcity_exp: float = 0.8        # (need/supply)^scarcity_exp
    premium_exp: float = 1.4         # (pts/alt_mean)^premium_exp
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(3.0, 8.0), (2.0, 6.0), (1.0, 4.0), (1.0, 3.0),
                (1.5, 4.0), (0.4, 1.0), (0.5, 1.5), (1.0, 2.0), (0.8, 2.0)]


class PositionalArbitrageur(Manager):
    """ILP-inspired: pays premium proportional to role scarcity need vs supply."""

    params: PositionalArbitrageurParams

    def _target(self) -> dict[str, int]:
        p = self.params
        return {"BAT": p.target_bat, "BOWL": p.target_bowl, "AR": p.target_ar, "WK": p.target_wk}

    def _scarcity(self, role: str, state: dict) -> float:
        p = self.params
        need = max(0, self._target()[role] - self.role_counts[role])
        supply = len(state["pool_by_role"].get(role, []))
        if need == 0:
            return p.filled_discount
        return min(p.scarcity_cap, 1.0 + (need / max(1, supply)) ** p.scarcity_exp)

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        sc = self._scarcity(player["role"], state)
        premium = (player["projected_points"] / max(1.0, self._alt_mean(player, state))) ** p.premium_exp
        return self._desperation(self.cash_per_slot * premium * sc, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 9 — BudgetSweeper
# Philosophy: hoard budget while the pool is full, then flood
# the late market when opponents are purse-constrained.
# An explicit deployment urgency term guarantees near-full
# budget usage — directly minimising leftover purse waste.
# ═══════════════════════════════════════════════════════════
@dataclass
class BudgetSweeperParams(StrategyParams):
    hoard_until: float = 0.55       # sit back when > this fraction of pool remains
    sweep_from: float = 0.30        # go aggressive when < this fraction remains
    hoard_mult: float = 0.30        # cash_per_slot multiplier in hoard phase
    transition_mult: float = 0.80   # multiplier in transition phase
    sweep_exp: float = 1.6          # _premium_wtp exponent in sweep phase
    sweep_boost: float = 1.25       # extra multiplier on top of premium in sweep
    underspend_boost: float = 1.60  # scale-up when falling behind budget pace
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.40, 0.70), (0.20, 0.45), (0.15, 0.55),
                (0.60, 1.10), (1.2, 2.5), (1.0, 2.0), (1.0, 2.5), (0.8, 2.0)]


class BudgetSweeper(Manager):
    """
    Patience-then-aggression.  Conserves early to outbid a cash-starved
    field in the final third, and uses a budget-deployment urgency term
    to ensure the purse is fully deployed.
    """

    params: BudgetSweeperParams

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        pool_ratio = state["players_remaining"] / max(1, POOL_SIZE)

        # How far behind are we on spending vs. roster-fill pace?
        spent_pct = 1.0 - (self.purse / self.total_purse)
        roster_pct = len(self.roster) / max(1, self.max_roster)
        underspend = max(0.0, roster_pct - spent_pct)        # 0 → on track
        urgency = 1.0 + underspend * p.underspend_boost

        if pool_ratio > p.hoard_until:
            wtp = self.cash_per_slot * p.hoard_mult
        elif pool_ratio > p.sweep_from:
            wtp = self.cash_per_slot * p.transition_mult
        else:
            wtp = self._premium_wtp(player, state, rnd, p.sweep_exp) * p.sweep_boost

        return self._desperation(wtp * urgency, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 10 — FieldAwareAdaptor
# Philosophy: reads the room.  When this team holds a purse-
# per-slot advantage over the field it plays patiently;
# when disadvantaged it bids aggressively.  A budget-pace
# tracker then closes the gap between ideal and actual spend.
# ═══════════════════════════════════════════════════════════
@dataclass
class FieldAwareAdaptorParams(StrategyParams):
    advantage_threshold: float = 1.15   # cps / field_cps ratio above which we sit back
    disadvantage_threshold: float = 0.85 # ratio below which we bid hard
    advantage_mult: float = 0.82        # bid conservatively when ahead
    parity_mult: float = 1.00           # neutral multiplier
    disadvantage_mult: float = 1.22     # bid aggressively when behind
    base_exp: float = 1.45              # _premium_wtp exponent baseline
    t1_premium: float = 1.25            # extra multiplier for Tier-1 players
    budget_urgency_scale: float = 1.30  # amplifies spend-lag into bid boost
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(1.05, 1.30), (0.70, 0.95), (0.60, 1.00),
                (0.85, 1.15), (1.05, 1.50), (1.1, 2.0),
                (1.0, 1.8), (1.0, 2.0), (0.8, 2.0)]


class FieldAwareAdaptor(Manager):
    """
    Relative-position bidder.  Uses agent_states to estimate who is
    cash-rich / cash-poor, then inverts the field's position into
    a bid multiplier.  Budget-pace term prevents unspent purse.
    """

    params: FieldAwareAdaptorParams

    def _field_avg_cps(self, state: dict) -> float:
        others = [
            (v["purse"], v["slots"])
            for n, v in state["agent_states"].items()
            if n != self.name
        ]
        if not others:
            return self.cash_per_slot
        return sum(p for p, _ in others) / max(1, sum(s for _, s in others))

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params

        relative = self.cash_per_slot / max(1.0, self._field_avg_cps(state))
        if relative > p.advantage_threshold:
            pos_mult = p.advantage_mult
        elif relative < p.disadvantage_threshold:
            pos_mult = p.disadvantage_mult
        else:
            pos_mult = p.parity_mult

        # Budget-pace correction
        slots_done = self.max_roster - self.slots
        spent = self.total_purse - self.purse
        ideal_spent = (slots_done / max(1, self.max_roster)) * self.total_purse
        spend_lag = max(0.0, ideal_spent - spent) / self.total_purse
        pos_mult *= 1.0 + spend_lag * p.budget_urgency_scale

        tier = player.get("tier", 2)
        if tier == 1:
            wtp = self._premium_wtp(player, state, rnd, p.base_exp) * p.t1_premium
        else:
            wtp = self._premium_wtp(player, state, rnd, p.base_exp)

        return self._desperation(wtp * pos_mult, state, rnd)


# ═══════════════════════════════════════════════════════════
# STRATEGY 11 — ContrarianSnapper
# Philosophy: avoids crowded bidding wars.  Estimates how
# many opponents can realistically compete for each player
# and backs off when competition is high (T1), snapping up
# under-contested depth instead.  Runs a late-auction sweep
# to deploy any remaining budget.
# ═══════════════════════════════════════════════════════════
@dataclass
class ContrarianSnapperParams(StrategyParams):
    competition_threshold: int = 3  # back off T1 if more than this many rivals can pay
    t1_contested_mult: float = 0.70  # T1 bid when heavily contested
    t1_uncontested_mult: float = 1.50 # T1 bid when few rivals can compete
    t2_mult: float = 1.10             # T2 bid multiplier
    t3_snap_mult: float = 1.35        # depth snap multiplier
    t3_min_pts: float = 50.0          # only snap Tier-3 if projected_points ≥ this
    late_pool_threshold: float = 0.25 # "late auction" when this fraction of pool remains
    late_sweep_exp: float = 1.75      # premium exponent for late sweep
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(1.0, 6.0), (0.40, 1.00), (1.10, 2.20),
                (0.80, 1.40), (0.90, 1.80), (30.0, 80.0),
                (0.15, 0.40), (1.2, 2.5), (0.8, 2.0)]


class ContrarianSnapper(Manager):
    """
    Contrarian: avoids T1 wars, targets under-contested value.
    Key insight: existing strategies all fight for the same players →
    large price distortions. Buying where competition is thin beats
    paying full price for consensus picks.
    """

    params: ContrarianSnapperParams

    def _competition_count(self, player: dict, state: dict) -> int:
        """Count opponents who appear able to win this player at fair value."""
        fair = 0.5 + player["vorp"] * state["cr_per_vorp"]
        return sum(
            1
            for n, o in state["agent_states"].items()
            if n != self.name
            and o["slots"] > 0
            and (o["purse"] / max(1, o["slots"])) >= fair * 0.65
        )

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        tier = player.get("tier", 2)
        pool_ratio = state["players_remaining"] / max(1, POOL_SIZE)

        # Late-auction: deploy remaining budget aggressively
        if pool_ratio < p.late_pool_threshold:
            return self._desperation(
                self._premium_wtp(player, state, rnd, p.late_sweep_exp), state, rnd
            )

        competition = self._competition_count(player, state)

        if tier == 1:
            mult = (
                p.t1_contested_mult
                if competition > p.competition_threshold
                else p.t1_uncontested_mult
            )
            wtp = self.cash_per_slot * mult
        elif tier == 2:
            wtp = self.cash_per_slot * p.t2_mult
        elif player["projected_points"] >= p.t3_min_pts:
            wtp = self.cash_per_slot * p.t3_snap_mult
        else:
            wtp = self.cash_per_slot * 0.75

        return self._desperation(wtp, state, rnd)


# ═══════════════════════════════════════════════════════════
# HYBRID STRATEGIES
# A hybrid blends two parent strategies' WTPs with a learnable
# α, then applies a budget-deployment urgency kicker so the
# full purse is deployed regardless of which parents are used.
#
# Architecture:
#   1. HybridBase syncs its live state (purse / roster /
#      role_counts) into both parent sub-instances before
#      each willingness_to_pay call.
#   2. raw_wtp = α·wtp_A + (1-α)·wtp_B
#   3. urgency boost = 1 + (spend_lag / total_purse) × kicker
#   4. final = min(raw_wtp × urgency, max_bid)
#
# Only α and budget_kicker are numeric → evolved by GA/CMA-ES.
# The parent strategy types are fixed per subclass.
# ═══════════════════════════════════════════════════════════
@dataclass
class _HybridBase(Manager):
    """Shared blend-and-kick logic for all hybrid strategies."""

    # Subclasses set these at class level
    _SA_CLS: type[Manager] = field(init=False, repr=False, default=None)
    _SA_PARAMS_CLS: type[StrategyParams] = field(init=False, repr=False, default=None)
    _SB_CLS: type[Manager] = field(init=False, repr=False, default=None)
    _SB_PARAMS_CLS: type[StrategyParams] = field(init=False, repr=False, default=None)

    _sa: Manager = field(init=False, repr=False, default=None)
    _sb: Manager = field(init=False, repr=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self._sa = self._SA_CLS(name="_sa", params=self._SA_PARAMS_CLS())
        self._sb = self._SB_CLS(name="_sb", params=self._SB_PARAMS_CLS())

    def _sync(self, sub: Manager) -> None:
        """Mirror live auction state into a sub-strategy instance."""
        sub.purse = self.purse
        sub.roster = self.roster
        sub.role_counts = self.role_counts

    def _blend(self, player: dict, state: dict, rnd: int) -> float:
        """α·wtp_A + (1-α)·wtp_B with budget urgency kicker."""
        if self.slots == 0:
            return 0.0
        self._sync(self._sa)
        self._sync(self._sb)
        wtp_a = self._sa.willingness_to_pay(player, state, rnd)
        wtp_b = self._sb.willingness_to_pay(player, state, rnd)
        alpha = self.params.alpha
        raw = alpha * wtp_a + (1.0 - alpha) * wtp_b

        # Budget urgency: penalise falling behind spend pace
        slots_done = self.max_roster - self.slots
        spent = self.total_purse - self.purse
        ideal = (slots_done / max(1, self.max_roster)) * self.total_purse
        lag = max(0.0, ideal - spent) / self.total_purse
        boost = 1.0 + lag * self.params.budget_kicker

        return min(raw * boost, self.max_bid)

    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        return self._blend(player, state, rnd)


# ── Hybrid 1: FieldSniper ────────────────────────────────────────────────────
# FieldAwareAdaptor × TierSniper
# Field-reading position logic combined with disciplined T1 locking.
# High quality picks + intelligent timing.
@dataclass
class FieldSniperParams(StrategyParams):
    alpha: float = 0.55          # weight on FieldAwareAdaptor (vs TierSniper)
    budget_kicker: float = 1.80  # urgency multiplier per unit spend-lag
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.1, 0.9), (0.5, 4.0), (0.8, 2.0)]


class FieldSniper(_HybridBase):
    """Hybrid: FieldAwareAdaptor (α) × TierSniper (1-α) + budget urgency."""
    _SA_CLS = FieldAwareAdaptor
    _SA_PARAMS_CLS = FieldAwareAdaptorParams
    _SB_CLS = TierSniper
    _SB_PARAMS_CLS = TierSniperParams
    params: FieldSniperParams


# ── Hybrid 2: VampireSweeper ─────────────────────────────────────────────────
# VampireMaximizer × BudgetSweeper
# Price-enforcement to drain rivals combined with late-auction budget sweep.
# Aggressive blocker that guarantees full spend.
@dataclass
class VampireSweeperParams(StrategyParams):
    alpha: float = 0.60          # weight on VampireMaximizer (vs BudgetSweeper)
    budget_kicker: float = 2.20  # stronger urgency — Vampire can sit back too long
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.1, 0.9), (0.5, 4.0), (0.8, 2.0)]


class VampireSweeper(_HybridBase):
    """Hybrid: VampireMaximizer (α) × BudgetSweeper (1-α) + budget urgency."""
    _SA_CLS = VampireMaximizer
    _SA_PARAMS_CLS = VampireMaximizerParams
    _SB_CLS = BudgetSweeper
    _SB_PARAMS_CLS = BudgetSweeperParams
    params: VampireSweeperParams


# ── Hybrid 3: ContrarianArbitrageur ──────────────────────────────────────────
# ContrarianSnapper × PositionalArbitrageur
# Avoids crowded auctions while filling roles by scarcity-need.
# Efficient, under-contested picks by role with near-full deployment.
@dataclass
class ContrarianArbitrageurParams(StrategyParams):
    alpha: float = 0.50          # equal blend by default
    budget_kicker: float = 1.60
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.1, 0.9), (0.5, 4.0), (0.8, 2.0)]


class ContrarianArbitrageur(_HybridBase):
    """Hybrid: ContrarianSnapper (α) × PositionalArbitrageur (1-α) + budget urgency."""
    _SA_CLS = ContrarianSnapper
    _SA_PARAMS_CLS = ContrarianSnapperParams
    _SB_CLS = PositionalArbitrageur
    _SB_PARAMS_CLS = PositionalArbitrageurParams
    params: ContrarianArbitrageurParams


# ── Hybrid 4: NominationFieldAdaptor ─────────────────────────────────────────
# NominationGambler × FieldAwareAdaptor
# Phase-timing discipline from NG + field-position awareness from FAA.
# NominationGambler conserves early and times T1 entries; FieldAwareAdaptor
# continuously reads who is cash-rich/poor and corrects spend lag.
# Together they close both gaps: phase-blind FAA & field-blind NG.
@dataclass
class NominationFieldAdaptorParams(StrategyParams):
    alpha: float = 0.50          # weight on NominationGambler (vs FieldAwareAdaptor)
    budget_kicker: float = 1.70  # urgency multiplier per unit spend-lag
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.1, 0.9), (0.5, 4.0), (0.8, 2.0)]


class NominationFieldAdaptor(_HybridBase):
    """Hybrid: NominationGambler (α) × FieldAwareAdaptor (1-α) + budget urgency."""
    _SA_CLS = NominationGambler
    _SA_PARAMS_CLS = NominationGamblerParams
    _SB_CLS = FieldAwareAdaptor
    _SB_PARAMS_CLS = FieldAwareAdaptorParams
    params: NominationFieldAdaptorParams


# ── Hybrid 5: VampireGambler ─────────────────────────────────────────────────
# VampireMaximizer × NominationGambler
# Price-enforcement bleed with phase-timing gate:
# early auction → conserve budget (NG phase-A), mid → bleed rivals (Vampire),
# late → both strategies agree to sweep aggressively.
# Fixes VampireMaximizer's weakness of sitting back without a timing signal.
@dataclass
class VampireGamblerParams(StrategyParams):
    alpha: float = 0.60          # weight on VampireMaximizer (vs NominationGambler)
    budget_kicker: float = 1.90  # stronger urgency — Vampire-style wait can over-save
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [(0.1, 0.9), (0.5, 4.0), (0.8, 2.0)]


class VampireGambler(_HybridBase):
    """Hybrid: VampireMaximizer (α) × NominationGambler (1-α) + budget urgency."""
    _SA_CLS = VampireMaximizer
    _SA_PARAMS_CLS = VampireMaximizerParams
    _SB_CLS = NominationGambler
    _SB_PARAMS_CLS = NominationGamblerParams
    params: VampireGamblerParams


# ═══════════════════════════════════════════════════
# ContextualEnsemble — scalable 4-parent mixture model
# ═══════════════════════════════════════════════════
# Architecture:
#   Parents: FieldAwareAdaptor, VampireMaximizer,
#            NominationGambler, PositionalArbitrageur
#   Blending: softmax over log-weights, then two layers
#             of context modifiers (phase × tier).
#   Evolution: GA/CMA-ES learns which parent to trust,
#              and how much that shifts per context cell.
#   Scalability: adding a new parent = one new log-weight.
# ═══════════════════════════════════════════════════
@dataclass
class ContextualEnsembleParams(StrategyParams):
    # Base log-weights (exp → softmax-normalised at call time)
    lw_field:     float = 0.0   # FieldAwareAdaptor
    lw_vampire:   float = 0.0   # VampireMaximizer
    lw_gambler:   float = 0.0   # NominationGambler
    lw_positional: float = 0.0  # PositionalArbitrageur

    # Phase context multipliers
    early_gambler_boost:  float = 1.50  # up-weight NG when pool >50% (conserve phase)
    late_field_boost:     float = 1.40  # up-weight FAA when pool <25% (read cash)
    late_vampire_boost:   float = 1.20  # up-weight Vampire when pool <25% (drain)

    # Tier context multipliers
    t1_field_boost:   float = 1.30  # FAA T1 premium logic is strong
    t1_vampire_boost: float = 1.10  # Vampire enforcement on sought-after T1s

    budget_kicker: float = 1.50     # spend-lag urgency
    new_team_premium: float = 1.0

    def get_bounds(self):
        return [
            (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0),  # log-weights
            (0.8, 3.0), (0.8, 3.0), (0.8, 2.5),                   # phase boosts
            (0.8, 2.5), (0.8, 2.0),                                # tier boosts
            (0.5, 4.0),                                            # budget_kicker
            (0.8, 2.0),                                            # new_team_premium
        ]


class ContextualEnsemble(Manager):
    """
    4-parent mixture model with context-dependent softmax blending.

    The GA/CMA-ES evolves log-weights and context modifiers so the
    ensemble learns *which parent strategy to trust* at each point in
    the auction (early / mid / late phase, T1 vs. other tier).  This
    is more scalable than fixed-α pairs: adding a parent adds one param.
    """

    params: ContextualEnsembleParams

    def __post_init__(self):
        super().__post_init__()
        self._fa = FieldAwareAdaptor(name="_fa", params=FieldAwareAdaptorParams())
        self._vm = VampireMaximizer(name="_vm", params=VampireMaximizerParams())
        self._ng = NominationGambler(name="_ng", params=NominationGamblerParams())
        self._pa = PositionalArbitrageur(name="_pa", params=PositionalArbitrageurParams())

    def _sync_all(self) -> None:
        for sub in (self._fa, self._vm, self._ng, self._pa):
            sub.purse = self.purse
            sub.roster = self.roster
            sub.role_counts = self.role_counts

    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        if self.slots == 0:
            return 0.0
        p = self.params
        self._sync_all()

        # Gather parent WTPs
        wtp_fa = self._fa.willingness_to_pay(player, state, rnd)
        wtp_vm = self._vm.willingness_to_pay(player, state, rnd)
        wtp_ng = self._ng.willingness_to_pay(player, state, rnd)
        wtp_pa = self._pa.willingness_to_pay(player, state, rnd)

        # Base softmax weights via log-weights
        w = [
            math.exp(p.lw_field),
            math.exp(p.lw_vampire),
            math.exp(p.lw_gambler),
            math.exp(p.lw_positional),
        ]

        # Phase context
        pool_ratio = state["players_remaining"] / max(1, POOL_SIZE)
        if pool_ratio > 0.50:     # early auction — NominationGambler timing
            w[2] *= p.early_gambler_boost
        elif pool_ratio < 0.25:   # late auction — field-read + drain
            w[0] *= p.late_field_boost
            w[1] *= p.late_vampire_boost

        # Tier context
        tier = player.get("tier", 2)
        if tier == 1:
            w[0] *= p.t1_field_boost
            w[1] *= p.t1_vampire_boost

        # Normalise
        total = sum(w)
        w = [x / total for x in w]

        # Weighted blend
        raw = w[0] * wtp_fa + w[1] * wtp_vm + w[2] * wtp_ng + w[3] * wtp_pa

        # Budget-pace urgency kicker
        slots_done = self.max_roster - self.slots
        spent = self.total_purse - self.purse
        ideal = (slots_done / max(1, self.max_roster)) * self.total_purse
        lag = max(0.0, ideal - spent) / self.total_purse
        boost = 1.0 + lag * p.budget_kicker

        return min(raw * boost, self.max_bid)


# ═══════════════════════════════════════════════════════════════════════════
# TeamDiversifier — control strategy for isolating team diversity value
# Pure team-spread signal: bonus for new franchises, discount once 3+
# players from the same team are acquired.  Allows direct measurement of
# how much C/VC team diversity is worth vs. ignoring squad composition.
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TeamDiversifierParams(StrategyParams):
    team_spread_target: int = 7      # aim for this many unique franchises
    spread_bonus: float = 1.5        # WTP multiplier for first player from new team
    saturation_discount: float = 0.7 # WTP multiplier when 3+ from same team
    base_exp: float = 1.4            # _premium_wtp exponent
    new_team_premium: float = 1.0    # handled directly in WTP; bid() is a no-op

    def get_bounds(self):
        return [(3.0, 10.0), (1.1, 2.5), (0.4, 0.9), (1.0, 2.0), (0.8, 2.0)]


class TeamDiversifier(Manager):
    """Control strategy: maximises IPL franchise diversity for C/VC coverage."""

    params: TeamDiversifierParams

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        p = self.params
        team = player.get("ipl_team")
        my_teams = {e["player"].get("ipl_team") for e in self.roster}
        team_count = sum(
            1 for e in self.roster if e["player"].get("ipl_team") == team
        )
        base_wtp = self._premium_wtp(player, state, rnd, p.base_exp)
        if team and team not in my_teams:
            mult = p.spread_bonus
        elif team_count >= 3:
            mult = p.saturation_discount
        elif len(my_teams) < p.team_spread_target:
            mult = 1.1
        else:
            mult = 1.0
        return self._desperation(base_wtp * mult, state, rnd)


# ── Registry ─────────────────────────────────────────────────────────────────
STRATEGY_REGISTRY: dict[str, tuple[type[Manager], type[StrategyParams]]] = {
    "StarChaser":                (StarChaser,                StarChaserParams),
    "ValueInvestor":             (ValueInvestor,              ValueInvestorParams),
    "DynamicMaximizer":          (DynamicMaximizer,           DynamicMaximizerParams),
    "ApexPredator":              (ApexPredator,               ApexPredatorParams),
    "BarbellStrategist":         (BarbellStrategist,          BarbellStrategistParams),
    "VampireMaximizer":          (VampireMaximizer,           VampireMaximizerParams),
    "TierSniper":                (TierSniper,                 TierSniperParams),
    "NominationGambler":         (NominationGambler,          NominationGamblerParams),
    "PositionalArbitrageur":     (PositionalArbitrageur,      PositionalArbitrageurParams),
    "BudgetSweeper":             (BudgetSweeper,              BudgetSweeperParams),
    "FieldAwareAdaptor":         (FieldAwareAdaptor,          FieldAwareAdaptorParams),
    "ContrarianSnapper":         (ContrarianSnapper,          ContrarianSnapperParams),
    # Hybrids (fixed-α, 2-parent)
    "FieldSniper":               (FieldSniper,                FieldSniperParams),
    "VampireSweeper":            (VampireSweeper,             VampireSweeperParams),
    "ContrarianArbitrageur":     (ContrarianArbitrageur,      ContrarianArbitrageurParams),
    "NominationFieldAdaptor":    (NominationFieldAdaptor,    NominationFieldAdaptorParams),
    "VampireGambler":            (VampireGambler,             VampireGamblerParams),
    # Ensemble (4-parent, context-weighted)
    "ContextualEnsemble":        (ContextualEnsemble,         ContextualEnsembleParams),
    # Control strategy for team diversity measurement
    "TeamDiversifier":           (TeamDiversifier,            TeamDiversifierParams),
}


# ── Param persistence ────────────────────────────────────────────────────────
def load_params(
    strategy_name: str,
    path: str | Path | None = None,
    evolved: bool = False,
) -> StrategyParams:
    """
    Load params for strategy_name from JSON file.
    Falls back to dataclass defaults if key not found.
    Set evolved=True to prefer the "evolved" sub-key over "default".
    """
    _, ParamsCls = STRATEGY_REGISTRY[strategy_name]
    p = Path(path) if path else _DEFAULT_PARAMS_PATH
    if not p.exists():
        return ParamsCls()
    with open(p) as f:
        data = json.load(f)
    entry = data.get(strategy_name, {})
    source = entry.get("evolved") if evolved and "evolved" in entry else entry.get("default", {})
    return ParamsCls.from_dict(source)


def save_params(
    strategy_name: str,
    params: StrategyParams,
    key: str = "default",
    path: str | Path | None = None,
) -> None:
    """Persist params under strategy_name.{key} in the JSON file."""
    p = Path(path) if path else _DEFAULT_PARAMS_PATH
    data: dict = {}
    if p.exists():
        with open(p) as f:
            data = json.load(f)
    data.setdefault(strategy_name, {})[key] = params.to_dict()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def instantiate(
    strategy_name: str,
    evolved: bool = False,
    params_path: str | Path | None = None,
    **manager_kwargs,
) -> Manager:
    """Instantiate a strategy by name with params loaded from JSON."""
    StrategyCls, _ = STRATEGY_REGISTRY[strategy_name]
    params = load_params(strategy_name, params_path, evolved=evolved)
    return StrategyCls(name=strategy_name, params=params, **manager_kwargs)
