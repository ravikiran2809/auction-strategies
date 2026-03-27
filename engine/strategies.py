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
import random
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

_DEFAULT_PARAMS_PATH = Path(__file__).parent.parent / "strategy_params.json"

POOL_SIZE = 80  # used by ValueInvestor urgency scaler


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
    max_roster: int = 15
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
        return max(0.0, self.purse - 0.5 * max(0, self.mandatory - 1))

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
        """Sum of top-11 projected_points — unified fitness metric."""
        pts = sorted(
            (e["player"]["projected_points"] for e in self.roster), reverse=True
        )
        return sum(pts[:11])

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

    def __repr__(self) -> str:
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

    def get_bounds(self):
        return [(0.20, 0.60), (1.5, 4.0), (0.8, 2.0), (0.7, 1.5), (0.3, 0.8)]


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

    def get_bounds(self):
        return [(0.5, 1.0), (1.0, 2.0), (0.3, 1.0)]


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

    def get_bounds(self):
        return [(1.0, 2.0), (0.02, 0.15), (1.5, 3.0)]


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

    def get_bounds(self):
        return [(0.70, 1.00), (1.0, 2.5), (0.5, 1.2), (1.5, 3.0)]


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
    hi_penalty_mult: float = 1.0     # std_dev multiplier for expensive picks
    lo_reward_mult: float = 1.0      # std_dev multiplier for cheap picks
    mid_penalty_mult: float = 0.25   # std_dev multiplier for mid-range picks
    base_exp: float = 1.6            # _premium_wtp exponent

    def get_bounds(self):
        return [(0.05, 0.30), (0.01, 0.10), (0.5, 2.0), (0.5, 2.0), (0.0, 1.0), (1.0, 2.5)]


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

    def get_bounds(self):
        return [(0.25, 1.5), (0.20, 0.60), (0.40, 0.80), (0.20, 0.60),
                (2.0, 7.0), (70.0, 120.0), (1.0, 2.0), (0.6, 1.2)]


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

    def get_bounds(self):
        return [(0.25, 0.65), (1.5, 4.0), (0.8, 1.5), (0.2, 0.8),
                (1.0, 6.0), (0.7, 1.5), (0.3, 1.0)]


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

    def get_bounds(self):
        return [(4.0, 12.0), (1.0, 5.0), (0.8, 2.5), (0.1, 0.5),
                (1.0, 2.5), (0.5, 1.5), (0.2, 0.7), (1.5, 3.0)]


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

    def get_bounds(self):
        return [(3.0, 8.0), (2.0, 6.0), (1.0, 4.0), (1.0, 3.0),
                (1.5, 4.0), (0.4, 1.0), (0.5, 1.5), (1.0, 2.0)]


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


# ── Registry ─────────────────────────────────────────────────────────────────
STRATEGY_REGISTRY: dict[str, tuple[type[Manager], type[StrategyParams]]] = {
    "StarChaser":             (StarChaser,             StarChaserParams),
    "ValueInvestor":          (ValueInvestor,           ValueInvestorParams),
    "DynamicMaximizer":       (DynamicMaximizer,        DynamicMaximizerParams),
    "ApexPredator":           (ApexPredator,            ApexPredatorParams),
    "BarbellStrategist":      (BarbellStrategist,       BarbellStrategistParams),
    "VampireMaximizer":       (VampireMaximizer,        VampireMaximizerParams),
    "TierSniper":             (TierSniper,              TierSniperParams),
    "NominationGambler":      (NominationGambler,       NominationGamblerParams),
    "PositionalArbitrageur":  (PositionalArbitrageur,   PositionalArbitrageurParams),
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
