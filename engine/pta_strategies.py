"""
engine/pta_strategies.py
========================
Plan-Track-Adapt (PTA) strategies — fully decoupled from existing strategies.

Imports only:
  - engine.strategies.Manager, StrategyParams  (base classes)
  - engine.market.MarketAnalyzer, SquadBuilder  (new infrastructure)
  - engine.intel.intel_mult                     (opt-in market intelligence)

New strategies:
  SquadCompletionBidder   — bids based on target priority score
  AdaptiveRecoveryManager — tracks lost targets and recovers with urgency

Both are registered into STRATEGY_REGISTRY at import time via
  `register_pta_strategies()` — called from engine/__init__.py or main.py

Market intelligence (engine/intel.py)
--------------------------------------
Both PTA strategies apply intel_mult() as a final WTP multiplier.  When no
intel file has been loaded (e.g. during simulation) the multiplier is always
1.0 and has zero effect.  Load intel once on startup via load_intel().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, asdict
from typing import Optional

from .intel import intel_mult
from .strategies import Manager, StrategyParams, STRATEGY_REGISTRY
from .market import MarketAnalyzer, SquadBuilder


# ── PTAManager base ───────────────────────────────────────────────────────────
@dataclass
class PTAManager(Manager):
    """
    Extends Manager with:
      - on_sale() hook called by auction.py after every purchase
      - on_unsold() hook called when a player goes unsold
      - Live MarketAnalyzer and SquadBuilder attached per instance
    """

    _market: MarketAnalyzer = field(init=False, repr=False, default=None)
    _squad:  SquadBuilder   = field(init=False, repr=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self._market = MarketAnalyzer()
        self._squad  = SquadBuilder()

    def reset(self):
        super().reset()
        self._market = MarketAnalyzer()
        self._squad  = SquadBuilder()

    # ── Event hooks (called by auction.py) ───────────────────────────────────
    def on_sale(
        self,
        player: dict,
        price: float,
        buyer: str,
        cr_per_vorp: float,
    ) -> None:
        """
        Called after EVERY purchase (including purchases by opponents).
        Override in subclasses for post-sale adaptation.
        """
        self._market.record_sale(player, price, buyer, cr_per_vorp)

    def on_unsold(self, player: dict) -> None:
        """Called when a player goes unsold."""

    # ── Helpers used by subclasses ────────────────────────────────────────────
    def _my_roster_players(self) -> list[dict]:
        return [e["player"] for e in self.roster]

    def _target_priority(self, player: dict, state: dict) -> float:
        remaining = list(state.get("pool_by_role", {}).get(player["role"], []))
        # add all roles to get full remaining pool
        all_remaining = [
            p for role_pool in state["pool_by_role"].values() for p in role_pool
        ]
        return self._squad.target_priority_score(
            player,
            self._my_roster_players(),
            all_remaining,
            self.purse,
            self.slots,
        )

    def _relative_standing(self, state: dict) -> float:
        """Returns my projected score / mean opponent projected score."""
        all_remaining = [
            p for role_pool in state["pool_by_role"].values() for p in role_pool
        ]
        opp_states = {}
        for name, s in state["agent_states"].items():
            if name != self.name:
                opp_states[name] = s  # s has purse, slots, mandatory
        return self._squad.relative_standing(
            self._my_roster_players(),
            self.purse,
            self.slots,
            all_remaining,
            opp_states,
        )

    def _opponent_role_pressure(self, player: dict, state: dict) -> float:
        """
        0.0–1.0: how much competitive pressure opponents have on this player.

        High value means multiple budget-healthy opponents need this role slot
        AND this player is one of their better remaining options.
        Static strategies CANNOT compute this — they don't track opponent rosters.

        Used to decide whether to bid harder to deny a specific opponent.
        """
        role = player["role"]
        role_min = self._squad.ROLE_MIN.get(role, 2)
        pressure_sum = 0.0
        opp_count = 0
        for name, opp in state["agent_states"].items():
            if name == self.name or opp["slots"] == 0:
                continue
            opp_roster = opp.get("roster_players", [])
            opp_role_count = sum(1 for p in opp_roster if p["role"] == role)
            opp_role_need = max(0, role_min - opp_role_count)
            # Budget health: can they actually compete for this player?
            opp_cps = opp["purse"] / max(1, opp["slots"])
            est_price = player.get("base_price", 1.0) * 2.0
            if opp_role_need >= 1 and opp_cps >= est_price:
                # Weight by urgency (role need) and budget leverage
                pressure_sum += opp_role_need * min(1.0, opp_cps / max(1.0, est_price))
            opp_count += 1
        if opp_count == 0:
            return 0.0
        # Normalise: max realistic pressure ≈ 4 opponents × need=2 × budget_mult=2 = 16
        return min(1.0, pressure_sum / 8.0)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PTA-1 — SquadCompletionBidder
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SquadCompletionBidderParams(StrategyParams):
    must_bid_mult: float       = 2.0   # WTP multiplier when priority >= must_bid_threshold
    useful_mult: float         = 1.1   # WTP multiplier when priority >= useful_threshold
    floor_mult: float          = 0.35  # WTP multiplier when player not needed (floor bid)
    must_bid_threshold: float  = 0.80  # priority score above which player is must-bid
    useful_threshold: float    = 0.50  # priority score above which player is useful
    overbid_dampen: float      = 0.75  # multiplier applied when market is overbidding
    buyer_boost: float         = 1.25  # multiplier applied in buyer's market
    base_exp: float            = 1.45  # exponent for _premium_wtp base calculation
    quality_floor: float       = 50.0  # never bid above floor_mult for players below this
    # Role cliff boost: when this player is MUCH better than next in role,
    # treat it as a must-bid regardless of priority score
    cliff_boost_threshold: float = 0.15  # cliff fraction above which boost activates
    cliff_must_bid_bonus: float  = 0.60  # added to mult when cliff is large
    # Opponent shading: if our WTP clearly beats all opponents, shade down to
    # save budget; if a must-bid player likely goes above our WTP, commit harder
    shade_margin: float          = 0.75  # ₹Cr above max_opp to shade down on wins
    shade_save_ratio: float      = 0.80  # WTP scaled to this fraction when shading down
    # Opponent denial: when opponents have high role pressure on a player WE also want,
    # bid harder to win it AND deny them the points — a uniquely dynamic advantage.
    denial_pressure_threshold: float = 0.55  # opp pressure above which denial activates
    denial_boost: float              = 1.35  # WTP multiplier when denying a pressured opp
    new_team_premium: float    = 1.0
    # Marquee recovery: if we keep losing T1 players, boost WTP on remaining T1/T2 lots.
    # Seed from past auction T1 pay ratios: 75th-pct human premium × 0.9 ≈ 0.674 × 0.9.
    # Calibrated via grid search in `python main.py simulate --marquee-pressure F`.
    marquee_recovery_mult: float = 1.4  # WTP boost on remaining T1 after losing lost_t1_trigger
    lost_t1_trigger: int         = 1   # enter marquee recovery after losing this many T1s

    def get_bounds(self):
        return [
            (1.2, 3.5),   # must_bid_mult
            (0.8, 1.5),   # useful_mult
            (0.15, 0.55), # floor_mult
            (0.60, 0.95), # must_bid_threshold
            (0.30, 0.70), # useful_threshold
            (0.50, 0.95), # overbid_dampen
            (1.0, 1.6),   # buyer_boost
            (1.1, 2.0),   # base_exp
            (30.0, 70.0), # quality_floor
            (0.05, 0.35), # cliff_boost_threshold
            (0.20, 1.00), # cliff_must_bid_bonus
            (0.25, 2.00), # shade_margin
            (0.50, 0.95), # shade_save_ratio
            (0.30, 0.85), # denial_pressure_threshold
            (1.0,  2.00), # denial_boost
            (0.8, 2.0),   # new_team_premium
            (1.0,  2.5),  # marquee_recovery_mult
            (1,    3),    # lost_t1_trigger (treated as float, rounded at use time)
        ]


class SquadCompletionBidder(PTAManager):
    """
    Bids proportional to how important each player is to optimal squad completion.

    High priority (must-bid): few alternatives, role needed, high cliff → pay up
    Medium priority (useful): some alternatives → fair value
    Low priority (not needed): role filled or better options exist → floor bid only
    Market mode dampens/boosts all bids reactively.
    """

    params: SquadCompletionBidderParams
    _lost_t1_count: int = field(init=False, default=0)

    def __post_init__(self):
        super().__post_init__()
        self._lost_t1_count = 0

    def reset(self):
        super().reset()
        self._lost_t1_count = 0

    def on_sale(self, player: dict, price: float, buyer: str, cr_per_vorp: float) -> None:
        super().on_sale(player, price, buyer, cr_per_vorp)
        if buyer != self.name and player.get("tier") == 1:
            self._lost_t1_count += 1

    def _opp_wtp_max(self, player: dict, state: dict) -> float:
        """Estimate the highest WTP any opponent is likely to have for this player.
        Uses the same model as VampireMaximizer: opp.purse / opp.slots * tier_mult."""
        best = 0.0
        for name, opp in state["agent_states"].items():
            if name == self.name or opp["slots"] == 0:
                continue
            avg = opp["purse"] / max(1, opp["slots"])
            cap = opp["purse"] - 0.5 * max(0, opp["mandatory"] - 1)
            tier_mult = 1.3 if player["projected_points"] > 85 else 1.0
            best = max(best, min(avg * tier_mult, cap))
        return best

    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        if self.slots == 0:
            return 0.0
        p = self.params

        priority = self._target_priority(player, state)

        # Compute raw base WTP WITHOUT internal capping so multipliers work correctly.
        premium = player["projected_points"] / max(1.0, self._alt_mean(player, state))
        raw_wtp = self.cash_per_slot * (premium ** p.base_exp)

        # Tier-based multiplier based on priority score
        if priority >= p.must_bid_threshold:
            mult = p.must_bid_mult
        elif priority >= p.useful_threshold:
            mult = p.useful_mult
        else:
            # Floor bid — still participate meaningfully, just not aggressively
            mult = (
                p.floor_mult
                if player["projected_points"] >= p.quality_floor
                else p.floor_mult * 0.5
            )

        # Role cliff boost: if this player is a large cliff above next-best in role,
        # upgrade the multiplier — they are essentially irreplaceable.
        all_remaining = [
            pl for role_pool in state["pool_by_role"].values() for pl in role_pool
        ]
        cliff = self._market.role_cliff(player["role"], all_remaining)
        if cliff > p.cliff_boost_threshold and priority >= p.useful_threshold:
            mult = min(mult + p.cliff_must_bid_bonus, p.must_bid_mult + 0.5)

        wtp = raw_wtp * mult

        # Budget deployment: if carrying more per-slot budget than starting pace,
        # deploy surplus on good players rather than leaving cash unspent.
        starting_pace = self.total_purse / self.max_roster
        if self.cash_per_slot > starting_pace * 1.2:
            deployment_boost = min(1.6, self.cash_per_slot / starting_pace)
            wtp *= deployment_boost

        # Opponent shading (VampireMaximizer insight applied to dynamic context):
        # • Must-bid player and opponents bid higher — raise to just beat them
        # • Non-critical player and we clearly win — shade down to save budget
        max_opp = self._opp_wtp_max(player, state)
        if priority >= p.must_bid_threshold and max_opp > wtp:
            # Can't afford to lose this player — set WTP just above max opponent estimate
            wtp = max_opp + p.shade_margin
        elif priority < p.useful_threshold and max_opp > 0 and wtp > max_opp + p.shade_margin:
            # We'd win comfortably — shade down to conserve budget
            wtp *= p.shade_save_ratio

        # Opponent denial: if this player is useful to us AND opponents are
        # under high role pressure for it, bid harder to deny them the player.
        # This is a uniquely dynamic insight — static strategies cannot estimate
        # opponent role gaps from their roster contents.
        if priority >= p.useful_threshold:
            opp_pressure = self._opponent_role_pressure(player, state)
            if opp_pressure >= p.denial_pressure_threshold:
                wtp *= p.denial_boost

        # Market mode adjustment — don't dampen high-priority must-bid players.
        mode = self._market.market_mode
        if mode == "overbid" and priority < p.must_bid_threshold:
            wtp *= p.overbid_dampen
        elif mode == "buyer":
            wtp *= p.buyer_boost

        # Marquee recovery: if we've lost enough T1 players, boost WTP on remaining T1/T2.
        # This compensates for human opponents bidding far above fair value on marquee lots.
        if (player.get("tier", 3) <= 2
                and self._lost_t1_count >= int(round(p.lost_t1_trigger))):
            wtp *= p.marquee_recovery_mult

        # Market intelligence: injury / form adjustment (no-op if intel not loaded).
        im = intel_mult(player["player_name"])
        if im == 0.0:
            return player.get("base_price", 0.5)  # injured/unavailable — token bid only
        wtp *= im

        return self._desperation(wtp, state, rnd)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PTA-2 — AdaptiveRecoveryManager
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class AdaptiveRecoveryManagerParams(StrategyParams):
    # Normal mode
    normal_exp: float          = 1.45   # base premium exponent
    market_sensitivity: float  = 0.40   # how much market_premium_ratio dampens bids
    # Recovery mode (triggered when behind or lost key targets)
    recovery_threshold: float  = 0.88   # relative standing below which to enter recovery
    recovery_urgency_mult: float = 1.6  # extra WTP multiplier in recovery mode
    recovery_sub_boost: float  = 2.2   # WTP mult for the nominated substitute target
    # Plan adherence
    off_plan_floor: float      = 0.40   # WTP fraction for sub-threshold players (below min_priority)
    min_priority: float        = 0.45   # ignore players below this priority entirely
    # Budget pre-allocation
    t1_budget_fraction: float  = 0.45   # fraction of purse pre-allocated to T1 targets
    new_team_premium: float    = 1.0
    # Marquee recovery: boost WTP on T1/T2 lots after losing too many marquee players.
    # Seed: 75th-pct human T1 pay ratio (0.674) × 0.9; calibrate via --marquee-pressure.
    marquee_recovery_mult: float = 1.4
    lost_t1_trigger: int         = 1

    def get_bounds(self):
        return [
            (1.1, 2.0),   # normal_exp
            (0.1, 0.8),   # market_sensitivity
            (0.75, 0.98), # recovery_threshold
            (1.1, 2.5),   # recovery_urgency_mult
            (1.5, 3.5),   # recovery_sub_boost
            (0.2, 0.7),   # off_plan_floor
            (0.3, 0.65),  # min_priority
            (0.30, 0.65), # t1_budget_fraction
            (0.8, 2.0),   # new_team_premium
            (1.0, 2.5),   # marquee_recovery_mult
            (1,   3),     # lost_t1_trigger (treated as float, rounded at use time)
        ]


class AdaptiveRecoveryManager(PTAManager):
    """
    Plan-Track-Adapt strategy with active recovery.

    Normal mode: premium WTP dampened/boosted by live market efficiency.
    Recovery mode (entered when lost key targets or falling behind opponents):
      - Identifies best substitute for each lost target
      - Bids aggressively on the substitute
      - Suppresses spending on off-plan players to preserve budget

    Key behaviours:
      - Never wastes slots on low-priority players while high-priority targets remain
      - Adapts budget allocation: pre-reserves T1 budget, releases when T1 pool depletes
      - Uses relative standing to decide whether to bid offensively or defensively
    """

    params: AdaptiveRecoveryManagerParams

    # Per-auction instance state
    _lost_targets: list[dict] = field(init=False, default_factory=list)
    _lost_pts_gap: float = field(init=False, default=0.0)
    _lost_t1_count: int = field(init=False, default=0)
    _substitute_target: Optional[str] = field(init=False, default=None)
    _t1_seen: int = field(init=False, default=0)
    _t1_remaining_est: int = field(init=False, default=20)

    def __post_init__(self):
        super().__post_init__()
        self._lost_targets = []
        self._lost_pts_gap = 0.0
        self._lost_t1_count = 0
        self._substitute_target = None
        self._t1_seen = 0
        self._t1_remaining_est = 20

    def reset(self):
        super().reset()
        self._lost_targets = []
        self._lost_pts_gap = 0.0
        self._lost_t1_count = 0
        self._substitute_target = None
        self._t1_seen = 0
        self._t1_remaining_est = 20

    # ── Event hooks ───────────────────────────────────────────────────────────
    def on_sale(self, player: dict, price: float, buyer: str, cr_per_vorp: float) -> None:
        super().on_sale(player, price, buyer, cr_per_vorp)

        # Track T1 depletion
        if player.get("tier") == 1:
            self._t1_seen += 1
            self._t1_remaining_est = max(0, self._t1_remaining_est - 1)

        # If a T1/T2 player we could have afforded was bought by an opponent,
        # log it as a lost target and accumulate the projected-points gap.
        if buyer != self.name and player.get("tier", 3) <= 2:
            pts = player["projected_points"]
            # Count as lost if we had at least 1.5× the sale price remaining
            if self.purse >= price * 1.5:
                self._lost_pts_gap += pts * 0.20  # marginal loss estimate
                if self._substitute_target is None:
                    self._substitute_target = None  # will resolve on next WTP call
        if buyer != self.name and player.get("tier") == 1:
            self._lost_t1_count += 1

    # ── Core bidding ──────────────────────────────────────────────────────────
    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        if self.slots == 0:
            return 0.0
        p = self.params

        all_remaining = [
            pl for role_pool in state["pool_by_role"].values() for pl in role_pool
        ]
        priority = self._target_priority(player, state)

        # Compute raw base WTP without internal capping (same fix as SCB: avoid
        # double-desperation so multipliers actually work)
        premium = player["projected_points"] / max(1.0, self._alt_mean(player, state))
        raw_wtp = self.cash_per_slot * (premium ** p.normal_exp)

        # Hard skip: priority too low — bid base_price as a soft floor rather
        # than completely abstaining; this deploys idle budget on cheap fillers
        if priority < p.min_priority and self.mandatory == 0:
            # Still participate if we have budget surplus; otherwise pass
            starting_pace = self.total_purse / self.max_roster
            if self.cash_per_slot > starting_pace * 1.4:
                # Significant surplus: bid conservatively but participate
                return self._desperation(raw_wtp * p.off_plan_floor, state, rnd)
            return player.get("base_price", 1.0)

        # Budget deployment: same as SCB — carry surplus budget into bids
        starting_pace = self.total_purse / self.max_roster
        budget_deployment = 1.0
        if self.cash_per_slot > starting_pace * 1.2:
            budget_deployment = min(1.6, self.cash_per_slot / starting_pace)

        # Check recovery state — multi-signal trigger:
        # 1. Relative standing below threshold (squad completion based)
        # 2. Lost too many T1/T2 players by pts gap
        # 3. PACE signal: spending significantly less per player than opponents on average
        #    This fires early in the auction before completion estimates are reliable.
        standing = self._relative_standing(state)

        opp_avg_purse_per_slot = 0.0
        opp_count = 0
        for name, opp in state["agent_states"].items():
            if name != self.name and opp["slots"] > 0:
                opp_avg_purse_per_slot += opp["purse"] / opp["slots"]
                opp_count += 1
        if opp_count > 0:
            opp_avg_purse_per_slot /= opp_count
        pace_behind = (
            opp_avg_purse_per_slot > 0
            and self.cash_per_slot < opp_avg_purse_per_slot * 0.75
            and len(self.roster) >= 3  # only meaningful once we have some data
        )

        in_recovery = (
            standing < p.recovery_threshold
            or self._lost_pts_gap > 50
            or pace_behind
        )

        if in_recovery:
            # Elect substitute target as needed
            if self._substitute_target is None:
                self._substitute_target = _find_substitute(player, all_remaining)

            if player["player_name"] == self._substitute_target:
                mult = p.recovery_sub_boost        # 2.2 — get this player at any cost
            elif priority >= 0.80:
                mult = p.recovery_urgency_mult     # 1.6 — urgently needed
            elif priority >= 0.55:
                mult = 1.2
            elif priority >= p.min_priority:
                mult = 0.70
            else:
                mult = p.off_plan_floor
        else:
            # Normal mode — PROACTIVE: compete hard for top targets;
            # dampening only applies to opportunistic bids where budget matters.
            if priority >= 0.80:
                mult = 2.2   # must-have: out-compete static strategies early
            elif priority >= 0.55:
                mult = 1.3   # useful: bid above fair value
            elif priority >= p.min_priority:
                mult = 0.80  # opportunistic: participate at a discount
            else:
                mult = p.off_plan_floor

        # Market dampening: only compress low-priority opportunistic bids.
        # High-priority decisions should not be affected by market noise.
        if priority < 0.55:
            market_premium = self._market.market_premium_ratio
            market_adj = max(0.6, 1.0 - p.market_sensitivity * max(0, market_premium - 1.0))
        else:
            market_adj = 1.0

        wtp = raw_wtp * mult * market_adj * budget_deployment

        # Marquee recovery: if we've lost enough T1 players, boost WTP on remaining T1/T2.
        if (player.get("tier", 3) <= 2
                and self._lost_t1_count >= int(round(p.lost_t1_trigger))):
            wtp *= p.marquee_recovery_mult

        # Market intelligence: injury / form adjustment (no-op if intel not loaded).
        im = intel_mult(player["player_name"])
        if im == 0.0:
            return player.get("base_price", 0.5)  # injured/unavailable — token bid only
        wtp *= im

        return self._desperation(wtp, state, rnd)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _t1_still_in_pool(remaining: list[dict]) -> int:
    return sum(1 for p in remaining if p.get("tier") == 1)


def _find_substitute(lost_player: dict, remaining: list[dict]) -> Optional[str]:
    """Find best same-role player in remaining pool."""
    candidates = [
        p for p in remaining
        if p["role"] == lost_player["role"]
        and p["player_name"] != lost_player["player_name"]
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p["projected_points"])["player_name"]


# ── Registration ──────────────────────────────────────────────────────────────
def register_pta_strategies() -> None:
    """Call once at startup to add PTA strategies to the global registry."""
    STRATEGY_REGISTRY["SquadCompletionBidder"] = (
        SquadCompletionBidder, SquadCompletionBidderParams
    )
    STRATEGY_REGISTRY["AdaptiveRecoveryManager"] = (
        AdaptiveRecoveryManager, AdaptiveRecoveryManagerParams
    )
