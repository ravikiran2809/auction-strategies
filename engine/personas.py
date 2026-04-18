"""
engine/personas.py
==================
Persona-based bidding strategies derived from real past auction data.

Each PersonaStrategy replicates an observed human manager's bidding behaviour
using their inferred pay ratios, aggression curve, and archetype.

Public API:
  load_personas(path)            → dict[team_name, persona_dict]
  register_persona_strategies()  → registers all personas into STRATEGY_REGISTRY
  PersonaStrategy                → Manager subclass parameterised by a persona profile

How it works:
  WTP = fair_value × tier_ratio × aggression_scale × scarcity_boost
  where:
    fair_value      = base_price + vorp * cr_per_vorp  (same as ValueInvestor)
    tier_ratio      = observed pay ratio for this player's tier (T1/T2/T3)
    aggression_scale= early_ratio → late_ratio blend based on auction progress
    scarcity_boost  = _desperation() inherited from Manager

Counter strategies: see PersonaCounter at the bottom — a SCB variant that
adapts its strategy based on detected persona patterns in the field.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from .strategies import Manager, StrategyParams, STRATEGY_REGISTRY

_PERSONAS_PATH = Path(__file__).parent.parent / "personas.json"

# ── Persona params dataclass ─────────────────────────────────────────────────
@dataclass
class PersonaParams(StrategyParams):
    """
    Bidding profile extracted from real auction data.

    All ratio fields: 1.0 = pay exactly at model fair value.
    """
    overall_ratio: float = 1.0    # fallback ratio if tier unknown
    t1_ratio: float = 0.6         # pay ratio for Tier-1 (marquee) players
    t2_ratio: float = 0.9         # pay ratio for Tier-2 players
    t3_ratio: float = 1.5         # pay ratio for Tier-3 players (often overbid!)
    early_ratio: float = 0.7      # aggression scaler in first third of auction
    mid_ratio: float = 1.0        # aggression scaler in middle third
    late_ratio: float = 1.5       # aggression scaler in final third
    budget_pct: float = 0.90      # target % of purse to spend (controls floor bids)
    role_bat: float = 1.0         # role-specific pay ratio for BAT
    role_bowl: float = 1.0        # role-specific pay ratio for BOWL
    role_ar: float = 1.0          # role-specific pay ratio for AR
    role_wk: float = 1.0          # role-specific pay ratio for WK

    def get_bounds(self):
        # Not evolved; all ranges kept wide for safety
        return [(0.1, 3.0)] * 12


def _make_persona_params(profile: dict) -> PersonaParams:
    """Build PersonaParams from a persona profile dict (from personas.json)."""
    rr = profile.get("role_ratios", {})
    return PersonaParams(
        overall_ratio = profile.get("overall_ratio", 1.0),
        t1_ratio      = profile.get("t1_ratio",      0.6),
        t2_ratio      = profile.get("t2_ratio",      0.9),
        t3_ratio      = profile.get("t3_ratio",      1.5),
        early_ratio   = profile.get("early_ratio",   0.7),
        mid_ratio     = profile.get("mid_ratio",     1.0),
        late_ratio    = profile.get("late_ratio",    1.5),
        budget_pct    = profile.get("budget_pct",    0.90),
        role_bat      = rr.get("BAT",  1.0),
        role_bowl     = rr.get("BOWL", 1.0),
        role_ar       = rr.get("AR",   1.0),
        role_wk       = rr.get("WK",   1.0),
    )


def _fair_value(player: dict, state: dict) -> float:
    """Model fair value: base_price + vorp * cr_per_vorp."""
    return player.get("base_price", 1.0) + player.get("vorp", 0.0) * state.get("cr_per_vorp", 0.3)


def _auction_progress(state: dict, total_pool: int = 125) -> float:
    """0.0 = start of auction, 1.0 = end. Based on players remaining."""
    remaining = state.get("players_remaining", total_pool)
    return max(0.0, min(1.0, 1.0 - remaining / total_pool))


# ── PersonaStrategy ───────────────────────────────────────────────────────────
@dataclass
class PersonaStrategy(Manager):
    """
    Simulates a human manager's bidding behaviour using their observed
    pay ratios and aggression curve from past auction data.

    WTP = fair_value × tier_ratio × phase_ratio × role_ratio × budget_urgency
    """
    params: PersonaParams

    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        if self.slots == 0:
            return 0.0

        p = self.params
        fv = _fair_value(player, state)

        # ── Tier ratio ────────────────────────────────────────────────────────
        tier = player.get("tier", 2)
        if tier == 1:
            tier_ratio = p.t1_ratio
        elif tier == 2:
            tier_ratio = p.t2_ratio
        else:
            tier_ratio = p.t3_ratio

        # ── Role ratio ────────────────────────────────────────────────────────
        role = player.get("role", "BAT")
        role_ratio = {
            "BAT":  p.role_bat,
            "BOWL": p.role_bowl,
            "AR":   p.role_ar,
            "WK":   p.role_wk,
        }.get(role, 1.0)

        # ── Phase (aggression curve) ──────────────────────────────────────────
        # Blend early→mid→late based on auction progress
        progress = _auction_progress(state)
        if progress < 0.33:
            phase_ratio = p.early_ratio
        elif progress < 0.67:
            # linear interpolation mid
            t = (progress - 0.33) / 0.34
            phase_ratio = p.early_ratio + t * (p.mid_ratio - p.early_ratio)
        else:
            # linear interpolation late
            t = (progress - 0.67) / 0.33
            phase_ratio = p.mid_ratio + t * (p.late_ratio - p.mid_ratio)

        # ── Budget urgency (ensure they deploy their budget_pct) ─────────────
        target_spend = self.total_purse * p.budget_pct
        spent = self.total_purse - self.purse
        spend_ratio = spent / max(1.0, target_spend)
        # If underspending significantly, bid higher proportionally
        urgency = 1.0
        if spend_ratio < 0.5 and self.slots <= 6:
            urgency = 1.0 + (0.5 - spend_ratio) * 0.8

        wtp = fv * tier_ratio * role_ratio * phase_ratio * urgency
        return self._desperation(wtp, state, rnd)


# ── PersonaCounter ────────────────────────────────────────────────────────────
@dataclass
class PersonaCounterParams(StrategyParams):
    """
    Counter strategy params — tuned against the persona field from our last auction.

    Key observations from past auction analysis:
      - 5/6 teams are LateSweepers: bid low early (0.3-0.6x), spike late (1.6-3.3x)
      - T1 players are consistently UNDERPAID (mean 0.55x fair value)
      - T3 players massively OVERBID (1.5-16x) in late rounds
      - Counter: lock T1 players early before late spike, avoid T3 bidding wars
    """
    # T1 aggression — go hard on marquee players before LateSweepers engage
    t1_early_mult: float       = 1.8   # WTP mult on T1 players in first 40% of auction
    t1_late_mult: float        = 2.4   # WTP mult on T1 if we still haven't got one by 50%

    # T3 discipline — LateSweepers overbid T3; we let them
    t3_cap_ratio: float        = 1.2   # never pay more than this × fair_value for T3
    t3_floor_bid_mult: float   = 0.3   # floor bid on T3 we don't need (drain their budget)

    # T2 opportunism — buy T2 value in mid-auction when LateSweepers are quiet
    t2_mid_mult: float         = 1.3   # WTP mult on T2 in 33-67% of auction
    t2_late_mult: float        = 0.9   # back off T2 late (LateSweepers overpay then)

    # Budget pacing — deploy early, don't leave cash for late overbid battles
    early_deploy_threshold: float = 1.3  # if cash_per_slot > pace × this, boost early bids
    early_deploy_boost: float     = 1.4  # boost applied when underspending early

    # Opponent denial — when a LateSweeper is being outbid on a player WE want, push them
    deny_on_pressure: bool = True   # not a float; not evolved
    deny_mult: float       = 1.2    # extra mult when opponent pressure is high

    def get_bounds(self):
        return [
            (1.2, 3.0),   # t1_early_mult
            (1.5, 4.0),   # t1_late_mult
            (0.8, 2.0),   # t3_cap_ratio
            (0.1, 0.5),   # t3_floor_bid_mult
            (1.0, 2.0),   # t2_mid_mult
            (0.6, 1.2),   # t2_late_mult
            (1.1, 2.0),   # early_deploy_threshold
            (1.1, 2.0),   # early_deploy_boost
            (1.0, 1.6),   # deny_mult
        ]


@dataclass
class PersonaCounter(Manager):
    """
    Counter strategy specifically designed against the LateSweeper-heavy
    field observed in our last auction.

    Core logic:
      1. LOCK T1 early — LateSweepers bid 0.3-0.5x on marquee players.
         We bid 1.8x WTP to steal them cheaply.
      2. SWEEP T2 mid-auction — LateSweepers are quiet in the middle.
         Aggressively pick up T2 value at 1.3x WTP.
      3. AVOID T3 wars — LateSweepers overbid T3 massively late.
         Floor-bid T3 to drain their purses; only buy at ≤1.2x fair value.
      4. BUDGET EARLY — Deploy 50%+ of budget in first half so we're not
         competing in late rounds where opponents are irrationally aggressive.
    """
    params: PersonaCounterParams

    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        if self.slots == 0:
            return 0.0

        p = self.params
        progress = _auction_progress(state)
        tier = player.get("tier", 2)
        fv = _fair_value(player, state)
        base_wtp = self._premium_wtp(player, state, rnd, exponent=1.5)

        # ── T1: lock early, commit hard if missed ────────────────────────────
        if tier == 1:
            own_t1 = sum(1 for e in self.roster if e["player"].get("tier") == 1)
            if own_t1 == 0:
                # No T1 yet — go hard
                mult = p.t1_late_mult if progress > 0.5 else p.t1_early_mult
                wtp = max(base_wtp, fv) * mult
            else:
                # Already have T1 — treat as T2
                wtp = base_wtp * p.t2_mid_mult
            return self._desperation(min(wtp, self.max_bid), state, rnd)

        # ── T3: cap spend; floor-bid to drain LateSweepers ──────────────────
        if tier == 3:
            cap = fv * p.t3_cap_ratio
            if base_wtp > cap:
                # We wouldn't pay this — but bid floor to make LateSweepers pay
                return min(fv * p.t3_floor_bid_mult, self.max_bid)
            return self._desperation(min(base_wtp, cap), state, rnd)

        # ── T2: opportunistic mid-auction, back off late ─────────────────────
        if progress < 0.67:
            wtp = base_wtp * p.t2_mid_mult
        else:
            wtp = base_wtp * p.t2_late_mult

        # ── Early budget deployment ──────────────────────────────────────────
        if progress < 0.4:
            starting_pace = self.total_purse / max(1, self.max_roster)
            if self.cash_per_slot > starting_pace * p.early_deploy_threshold:
                wtp = min(wtp * p.early_deploy_boost, self.max_bid)

        return self._desperation(wtp, state, rnd)


# ── Load + register ───────────────────────────────────────────────────────────
def load_personas(path: Path | str = _PERSONAS_PATH) -> dict[str, dict]:
    """Load personas.json and return the personas dict."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return data.get("personas", {})


def register_persona_strategies(path: Path | str = _PERSONAS_PATH) -> list[str]:
    """
    Register one PersonaStrategy per team from personas.json into STRATEGY_REGISTRY.
    Also registers PersonaCounter.

    Persona params are stored as defaults in strategy_params.json so that the
    standard instantiate() / load_params() path works without modification.

    Returns list of registered strategy names.
    """
    from .strategies import save_params  # import here to avoid circular at module level

    registered: list[str] = []
    personas = load_personas(path)

    for team_name, persona in personas.items():
        # Sanitise name for use as strategy key
        strategy_name = f"Persona_{team_name.replace(' ', '_')}"
        profile = persona.get("profile", {})
        params = _make_persona_params(profile)

        # Build a unique class per persona (needed so each has its own name)
        persona_cls = type(
            strategy_name,
            (PersonaStrategy,),
            {"__doc__": f"Persona strategy for {team_name} ({persona.get('archetype', '?')})"},
        )
        STRATEGY_REGISTRY[strategy_name] = (persona_cls, PersonaParams)
        # Persist params so load_params() can retrieve them normally
        save_params(strategy_name, params, key="default")
        registered.append(strategy_name)

    # Register PersonaCounter
    if "PersonaCounter" not in STRATEGY_REGISTRY:
        STRATEGY_REGISTRY["PersonaCounter"] = (PersonaCounter, PersonaCounterParams)
        save_params("PersonaCounter", PersonaCounterParams(), key="default")
        registered.append("PersonaCounter")

    return registered
