"""
IPL Fantasy Auction Simulator v2 — FINAL
==========================================
9 strategies (5 fixed from v1, 1 updated, 3 new):
  0  StarChaser             FIXED: 35% purse cap, elite-lock-once logic
  1  ValueInvestor          FIXED: urgency scaler for full cash deployment
  2  DynamicMaximizer       FIXED: exponent scales with number of agents
  3  ApexPredator           FIXED: winner's-curse 10% discount (Milgrom/Thaler)
  4  BarbellStrategist      FIXED: no shared-dict mutation bug
  5  VampireMaximizer       FIXED: +0.5Cr shade margin, healthy-purse guard
  6  TierSniper             NEW: tier-lock (CBS Sports/RotoBaller meta)
  7  NominationGambler      NEW: elite-count-aware phase conservation
  8  PositionalArbitrageur  NEW: ILP-inspired role-scarcity multiplier

Academic sources:
  Anagnostopoulos et al. "Bidding Strategies for Fantasy-Sports Auctions"
  Singh et al. "Dynamic Bidding Strategy for IPL" (integer-programming model)
  Milgrom "Putting Auction Theory to Work" — winner's curse
  CBS Sports / RotoBaller 2024 auction meta: tier-lock, nomination leverage

Usage:
  python ipl_auction_sim_v2.py
  python ipl_auction_sim_v2.py --quiet
  python ipl_auction_sim_v2.py --monte-carlo 1000
  python ipl_auction_sim_v2.py --strategies 2 5 6   # by index 0-8
"""

import json, random, argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ──────────────────────────────────────────────────────────
# 1.  DATA PIPELINE
# ──────────────────────────────────────────────────────────
STATS_PATH = "ipl_player_stats.csv"
RULES_PATH = "scoring_rules.json"
SEASON = "2024"
MIN_MATCHES = 5
POOL_SIZE = 80

TIER1_FLOOR = 80  # avg pts/match → elite
TIER2_FLOOR = 58  # avg pts/match → solid  (below = depth)


def load_player_pool() -> list[dict]:
    df = pd.read_csv(STATS_PATH)
    season = df[df["ipl_year"] == SEASON].copy()
    season["fp"] = season.apply(_calc_fp, axis=1)

    agg = (
        season.groupby(["player_id", "player_name"])
        .agg(
            matches=("match_id", "nunique"),
            total_pts=("fp", "sum"),
            avg_pts=("fp", "mean"),
            std_pts=("fp", "std"),
            total_runs=("runs_scored", "sum"),
            total_wkts=("wickets", "sum"),
            total_catches=("catches_caught", "sum"),
            total_balls_bowled=("balls_bowled", "sum"),
        )
        .reset_index()
    )

    agg["std_pts"] = agg["std_pts"].fillna(30)
    agg = agg[agg["matches"] >= MIN_MATCHES].copy()
    agg["role"] = agg.apply(_classify_role, axis=1)

    replacement = {
        r: agg[agg["role"] == r]["avg_pts"].quantile(0.25)
        for r in ["BAT", "BOWL", "AR", "WK"]
    }
    agg["vorp"] = agg.apply(
        lambda r: max(0.0, r["avg_pts"] - replacement.get(r["role"], 40)), axis=1
    )

    top = agg.nlargest(POOL_SIZE, "avg_pts")
    players = []
    for _, r in top.iterrows():
        pts = round(r["avg_pts"], 1)
        tier = 1 if pts > TIER1_FLOOR else (2 if pts > TIER2_FLOOR else 3)
        players.append(
            {
                "player_name": r["player_name"],
                "role": r["role"],
                "projected_points": pts,
                "std_dev": round(r["std_pts"], 1),
                "vorp": round(r["vorp"], 1),
                "tier": tier,
                "matches": int(r["matches"]),
            }
        )
    return players


def _calc_fp(row) -> float:
    runs, balls = row["runs_scored"], row["balls_faced"]
    wkts, bowl = row["wickets"], row["balls_bowled"]
    role = row.get("role", "")
    overs = bowl / 6 if bowl > 0 else 0
    eco = row["runs_given"] / overs if overs > 0 else 0
    sr = (runs / balls * 100) if balls > 0 else 0
    pts = (
        runs
        + row["four_count"]
        + row["six_count"] * 2
        + row["catches_caught"] * 10
        + row["runouts"] * 20
        + row.get("stumpings", 0) * 20
        + wkts * 40
        + row["bowled_lbw_wickets"] * 10
        + row["maiden_count"] * 50
        + row.get("dot_ball_count", 0) * 1
        + row.get("wides_bowled", 0) * (-1)
        + row.get("no_balls_bowled", 0) * (-1)
    )
    # Duck penalty: only for BAT, AR, WK — not pure bowlers
    if runs == 0 and row["is_out"] == 1 and role != "BOWL":
        pts -= 10
    if balls >= 10:
        if runs >= 150:
            pts += 150
        elif runs >= 100:
            pts += 50
        elif runs >= 75:
            pts += 25
        elif runs >= 50:
            pts += 20
        elif runs >= 30:
            pts += 5
        if sr >= 200:
            pts += 35
        elif sr >= 150:
            pts += 25
        elif sr >= 125:
            pts += 10
        elif sr < 100:
            pts -= 15
    if wkts >= 5:
        pts += 150
    elif wkts >= 3:
        pts += 50
    # Economy penalties: only for BOWL and AR
    if bowl >= 12 and role in ("BOWL", "AR"):
        if eco < 4:
            pts += 50
        elif eco < 6:
            pts += 35
        elif eco < 8:
            pts += 15
        elif 10 <= eco < 12:
            pts -= 10
        elif eco >= 12:
            pts -= 20
    elif bowl >= 12 and role not in ("BOWL", "AR"):
        # BAT/WK still get economy bonuses, just not penalties
        if eco < 4:
            pts += 50
        elif eco < 6:
            pts += 35
        elif eco < 8:
            pts += 15
    return pts


def _classify_role(r) -> str:
    if r["total_catches"] >= 8 and r["total_runs"] < 150 and r["total_wkts"] == 0:
        return "WK"
    if r["total_runs"] >= 250 and r["total_wkts"] >= 8:
        return "AR"
    if r["total_balls_bowled"] >= 120 and r["total_runs"] < 200:
        return "BOWL"
    if (
        r["total_balls_bowled"] >= 60
        and r["total_wkts"] >= 6
        and r["total_runs"] >= 100
    ):
        return "AR"
    return "BAT"


# ──────────────────────────────────────────────────────────
# 2.  BASE MANAGER
# ──────────────────────────────────────────────────────────


@dataclass
class Manager:
    name: str
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
        return max(0.0, self.purse - 0.5 * max(0, self.mandatory - 1))

    @property
    def cash_per_slot(self) -> float:
        return self.purse / self.slots if self.slots > 0 else 0.0

    def buy(self, player: dict, price: float):
        self.roster.append({"player": player, "price": price})
        self.purse -= price
        self.role_counts[player["role"]] += 1

    def fire_sale(self) -> Optional[dict]:
        if self.mandatory > 0 and self.max_bid <= 0.5 and self.roster:
            self.roster.sort(key=lambda x: x["price"], reverse=True)
            sold = self.roster.pop(0)
            self.purse += sold["price"] * 0.75
            self.role_counts[sold["player"]["role"]] -= 1
            return sold["player"]
        return None

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

    def _premium_wtp(
        self, player: dict, state: dict, rnd: int, exponent: float = 1.6
    ) -> float:
        """Core DynamicMaximizer logic — reusable across strategies."""
        premium = player["projected_points"] / max(1.0, self._alt_mean(player, state))
        return self._desperation(self.cash_per_slot * (premium**exponent), state, rnd)

    def willingness_to_pay(self, player: dict, state: dict, rnd: int) -> float:
        raise NotImplementedError

    def season_score(self) -> float:
        pts = sorted(
            (e["player"]["projected_points"] for e in self.roster), reverse=True
        )
        return sum(pts[:11])

    def __repr__(self):
        return (
            f"{self.name:<30} | {len(self.roster):>2}/{self.max_roster}"
            f" | ₹{self.purse:>5.1f}Cr | {self.season_score():>6.0f}pts"
        )


# ──────────────────────────────────────────────────────────
# 3.  THE 9 STRATEGIES
# ──────────────────────────────────────────────────────────


class StarChaser(Manager):
    """
    FIXED: Caps elite spend at 35% of total purse. Goes aggressive exactly
    ONCE (first tier-1 pick), then backs off so depth is always funded.
    Avoids v1's 'dry on depth' failure mode while keeping the Stars & Scrubs
    philosophy intact.
    """

    ELITE_CAP = 0.35

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        tier = player.get("tier", 2)
        own_t1 = sum(1 for e in self.roster if e["player"].get("tier") == 1)
        cap = self.total_purse * self.ELITE_CAP
        if tier == 1 and own_t1 == 0:
            wtp = min(cap, self.cash_per_slot * 2.4)
        elif tier == 1:
            wtp = self.cash_per_slot * 1.2
        elif tier == 2:
            wtp = self.cash_per_slot * 1.0
        else:
            wtp = self.cash_per_slot * 0.55
        return self._desperation(wtp, state, rnd)


class ValueInvestor(Manager):
    """
    FIXED: Urgency scaler from 0.85→1.35× fair price as remaining cash
    outpaces remaining pool. Full purse deployment guaranteed.
    v1 left 15-20Cr unspent by refusing to bid above 90% of 'fair price'.
    """

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        fair = 0.5 + player["vorp"] * state["cr_per_vorp"]
        cash_ratio = self.purse / max(1.0, self.total_purse)
        pool_ratio = state["players_remaining"] / max(1, POOL_SIZE)
        urgency = max(0.85, min(1.35, 1.0 + (cash_ratio - pool_ratio)))
        return self._desperation(fair * urgency, state, rnd)


class DynamicMaximizer(Manager):
    """
    FIXED: Exponent adapts to the number of agents in the room.
    In a 4-agent race it uses 1.4; in a 9-agent race it scales to 1.75+.
    More competition = faster escalation needed to stay competitive.
    Still the cleanest mathematical core strategy.
    """

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        n_agents = len(state["agent_states"])
        exponent = min(2.0, 1.4 + (n_agents - 4) * 0.07)
        return self._premium_wtp(player, state, rnd, exponent)


class ApexPredator(Manager):
    """
    FIXED: Applies Milgrom/Thaler winner's-curse discount (10%).
    v1 over-paid trying to be clever because it didn't account for
    the fact that the auction winner is statistically the person with
    the largest estimation error.
    """

    WC = 0.90

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        cper = state["cr_per_vorp"]
        exp = 0.5 + player["vorp"] * cper
        pool = state["pool_by_role"].get(player["role"], [])
        alts = pool[: max(1, 4 - self.role_counts[player["role"]])]
        if not alts:
            return self._desperation(0.5, state, rnd)
        alt_pts = sum(p["projected_points"] for p in alts) / len(alts)
        alt_cost = 0.5 + max(0, alt_pts - 40) * cper
        p_eff = player["projected_points"] / max(0.5, exp)
        a_eff = alt_pts / max(0.5, alt_cost)
        ratio = (p_eff / max(0.01, a_eff)) ** 1.5
        adj = min(2.0, max(0.8, self.cash_per_slot / max(1.0, exp)))
        return self._desperation(exp * ratio * adj * self.WC, state, rnd)


class BarbellStrategist(Manager):
    """
    FIXED: Now uses a temporary copy so it never mutates the shared player dict.
    v1 had a subtle bug where modifying projected_points on the shared dict
    could corrupt other agents' evaluations in the same round.
    Still adjusts perceived pts by std_dev: penalise high-cost volatility,
    reward high-upside cheap picks.
    """

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        exp = 0.5 + player["vorp"] * state["cr_per_vorp"]
        ci = exp / max(1.0, self.purse)
        std = player.get("std_dev", 50)
        if ci > 0.15:
            adj = player["projected_points"] - std * 1.0
        elif ci < 0.05:
            adj = player["projected_points"] + std * 1.0
        else:
            adj = player["projected_points"] - std * 0.25
        # Use a copy — never mutate the shared dict
        copy = {**player, "projected_points": max(10, adj)}
        return self._premium_wtp(copy, state, rnd, 1.6)


class VampireMaximizer(Manager):
    """
    FIXED: Bid-shade now at exactly +0.5Cr above enemy max (v1 used 85%
    which could still lose to a bid-shade opponent). Price-enforce ONLY
    when purse >40% AND 4+ slots remain. Source: CBS Sports 'enforce up
    to 60% of your own earmark to avoid getting stuck with the player'.
    """

    def _opp_wtp(self, player, opp) -> float:
        if opp["slots"] == 0:
            return 0.0
        avg = opp["purse"] / max(1, opp["slots"])
        cap = opp["purse"] - 0.5 * max(0, opp["mandatory"] - 1)
        return min(avg * (1.4 if player["projected_points"] > 90 else 0.9), cap)

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        my_wtp = self._premium_wtp(player, state, rnd, 1.6)
        max_opp = max(
            (
                self._opp_wtp(player, o)
                for n, o in state["agent_states"].items()
                if n != self.name
            ),
            default=0.0,
        )

        if my_wtp >= max_opp + 0.5:
            return self._desperation(my_wtp, state, rnd)  # win precisely

        healthy = (self.purse / self.total_purse) > 0.4 and self.slots > 4
        if my_wtp < max_opp and my_wtp > max_opp * 0.40 and healthy:
            return min(my_wtp / 0.60, self.max_bid)  # push to 60% cap

        return self._desperation(my_wtp, state, rnd)


class TierSniper(Manager):
    """
    NEW — Tier-lock strategy (CBS Sports / RotoBaller 2024 auction meta).
    Locks ONE tier-1 (elite) player early at up to 45% of total purse.
    Then backs off to let others fight over remaining elite slots,
    sweeping tier-2 and tier-3 at disciplined prices.
    Key insight: 'Never bid on the last player left in a tier — they always
    go for more than earlier ones at the same quality level.'
    """

    T1_CAP = 0.45

    def _t1_remaining(self, state) -> int:
        return sum(
            1 for ps in state["pool_by_role"].values() for p in ps if p.get("tier") == 1
        )

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        tier = player.get("tier", 2)
        own_t1 = sum(1 for e in self.roster if e["player"].get("tier") == 1)
        t1_rem = self._t1_remaining(state)

        if tier == 1:
            if own_t1 == 0:
                wtp = min(self.total_purse * self.T1_CAP, self.cash_per_slot * 2.8)
            elif t1_rem <= 3:
                wtp = self.cash_per_slot * 1.05  # price-enforce last few
            else:
                wtp = self.cash_per_slot * 0.5  # already have one, sit back
        elif tier == 2:
            wtp = self.cash_per_slot * 1.1
        else:
            wtp = self.cash_per_slot * 0.65
        return self._desperation(wtp, state, rnd)


class NominationGambler(Manager):
    """
    NEW — Phase-aware strategy using ELITE PLAYER COUNT not % sold.
    (% sold is fragile in a multi-agent pool; elite count is signal-clear.)

    Phase A (8+ tier-1s left): Bid only on tier-1. Aggressively underbid
      everything else to preserve cash while others waste it on tier-2.
    Phase B (3-7 tier-1s left): Compete on tier-1 and strong tier-2.
    Phase C (<3 tier-1s left): Full liquidation — spend all remaining cash.

    Source: RotoBaller — 'letting others spend early gives you buying power
    late'. CBS Sports — 'the worst thing is leaving money on the table'.
    """

    def _t1_remaining(self, state) -> int:
        return sum(
            1 for ps in state["pool_by_role"].values() for p in ps if p.get("tier") == 1
        )

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        t1_rem = self._t1_remaining(state)
        tier = player.get("tier", 2)

        if t1_rem >= 8:  # Phase A: conserve
            wtp = self.cash_per_slot * (1.5 if tier == 1 else 0.25)
        elif t1_rem >= 3:  # Phase B: selective
            if tier == 1:
                wtp = self._premium_wtp(player, state, rnd, 1.5)
            elif tier == 2:
                wtp = self.cash_per_slot * 0.9
            else:
                wtp = self.cash_per_slot * 0.40
        else:  # Phase C: liquidate
            wtp = self._premium_wtp(player, state, rnd, 2.0)

        return self._desperation(wtp, state, rnd)


class PositionalArbitrageur(Manager):
    """
    NEW — ILP-inspired role-scarcity multiplier (Singh et al. 2011 / Cornell).
    Pays a premium proportional to (how much you need this role) /
    (how much supply of this role remains in the pool).
    Target composition: 5 BAT, 4 BOWL, 2 AR, 2 WK.
    Benefit: automatically over-pays for your weakest positions before
    they run out, avoiding the 'desperate scrub buy' at the end.
    """

    TARGET = {"BAT": 5, "BOWL": 4, "AR": 2, "WK": 2}

    def _scarcity(self, role: str, state: dict) -> float:
        need = max(0, self.TARGET[role] - self.role_counts[role])
        supply = len(state["pool_by_role"].get(role, []))
        if need == 0:
            return 0.7  # role filled — apply discount
        return min(2.5, 1.0 + (need / max(1, supply)) ** 0.8)

    def willingness_to_pay(self, player, state, rnd):
        if self.slots == 0:
            return 0.0
        sc = self._scarcity(player["role"], state)
        premium = (
            player["projected_points"] / max(1.0, self._alt_mean(player, state))
        ) ** 1.4
        return self._desperation(self.cash_per_slot * premium * sc, state, rnd)


# ──────────────────────────────────────────────────────────
# 4.  AUCTION ENGINE
# ──────────────────────────────────────────────────────────
BID_INCREMENT = 0.25


def build_state(agents: list, remaining: list[dict]) -> dict:
    avail = sum(a.purse - 0.5 * max(0, a.mandatory - 1) for a in agents)
    rem_v = sum(p["vorp"] for p in remaining)
    return {
        "players_remaining": len(remaining),
        "cr_per_vorp": avail / rem_v if rem_v > 0 else 0.0,
        "pool_by_role": {
            r: sorted(
                [p for p in remaining if p["role"] == r],
                key=lambda x: x["projected_points"],
                reverse=True,
            )
            for r in ["BAT", "BOWL", "AR", "WK"]
        },
        "agent_states": {
            a.name: {
                "purse": a.purse,
                "slots": a.slots,
                "mandatory": a.mandatory,
                "name": a.name,
            }
            for a in agents
        },
    }


def run_auction(pool: list[dict], agents: list, verbose=True) -> list[dict]:
    for a in agents:
        a.reset()
    random.shuffle(pool)
    unsold, lot_pool, rnd = [], list(pool), 1

    while lot_pool:
        if verbose and rnd > 1:
            print(
                f"\n{'─' * 64}\n  ♻  ROUND {rnd} — {len(lot_pool)} lots re-entered\n{'─' * 64}"
            )

        current, lot_pool = list(lot_pool), []
        bought = 0

        for player in current:
            remaining = current + unsold + lot_pool
            state = build_state(agents, remaining)
            active = [a for a in agents if a.slots > 0 and a.max_bid >= 0.5]

            wtp = {}
            for a in active:
                raw = a.willingness_to_pay(player, state, rnd)
                noise = random.uniform(-0.25, 0.5) if raw > 0.5 else 0.0
                wtp[a.name] = min(raw + noise, a.max_bid)

            if not any(v >= 0.5 for v in wtp.values()):
                unsold.append(player)
                if verbose:
                    print(
                        f"  ⛔ UNSOLD  {player['player_name']:<22} ({player['role']})"
                    )
                continue

            bid, winner, going = 0.0, None, True
            while going:
                going = False
                for a in random.sample(active, len(active)):
                    if a.name == winner:
                        continue
                    ask = bid + BID_INCREMENT if winner else 0.5
                    if wtp[a.name] >= ask and ask <= a.max_bid:
                        bid, winner, going = ask, a.name, True
                        break

            w = next(a for a in agents if a.name == winner)
            w.buy(player, bid)
            bought += 1

            if verbose:
                print(
                    f"  🏏 {player['player_name']:<22} {player['role']:<4} T{player.get('tier', '?')}"
                    f" → {w.name:<30} ₹{bid:.2f}Cr  [{player['projected_points']:.0f}pts]"
                )

            for a in agents:
                ret = a.fire_sale()
                if ret:
                    lot_pool.append(ret)
                    if verbose:
                        print(f"  🔥 {a.name} fire-sales {ret['player_name']}")

        if bought == 0:
            break
        if any(a.mandatory > 0 for a in agents) and unsold:
            lot_pool.extend(unsold)
            unsold = []
            rnd += 1
        else:
            break

    return unsold


# ──────────────────────────────────────────────────────────
# 5.  REPORTING
# ──────────────────────────────────────────────────────────


def print_results(agents: list, unsold: list):
    print("\n" + "═" * 72)
    print("  🏆  FINAL RESULTS")
    print("═" * 72)
    ranked = sorted(agents, key=lambda a: a.season_score(), reverse=True)
    for i, a in enumerate(ranked):
        m = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        spent = a.total_purse - a.purse
        eff = a.season_score() / spent if spent > 0 else 0
        print(
            f"  {m} {a.name:<30} Score:{a.season_score():>6.0f}  "
            f"Spent:₹{spent:>5.1f}  Eff:{eff:.1f}pts/Cr  {len(a.roster)}p"
        )

    w = ranked[0]
    print(f"\n  {'─' * 70}\n  🏆 WINNER: {w.name}\n  {'─' * 70}")
    for i, e in enumerate(
        sorted(w.roster, key=lambda x: x["player"]["projected_points"], reverse=True), 1
    ):
        p = e["player"]
        print(
            f"  {'✅' if i <= 11 else '  '}{i:>2}. {p['player_name']:<22} {p['role']:<4}"
            f" {p['projected_points']:>6.1f}pts  ₹{e['price']:>5.2f}Cr"
        )
    if unsold:
        print(f"\n  ⛔ {len(unsold)} unsold")
    print("═" * 72)


# ──────────────────────────────────────────────────────────
# 6.  MONTE CARLO
# ──────────────────────────────────────────────────────────

ALL_STRATEGIES = [
    StarChaser,
    ValueInvestor,
    DynamicMaximizer,
    ApexPredator,
    BarbellStrategist,
    VampireMaximizer,
    TierSniper,
    NominationGambler,
    PositionalArbitrageur,
]


def run_monte_carlo(players: list[dict], n: int, indices: list[int]):
    classes = [ALL_STRATEGIES[i] for i in indices]
    wins, scores = defaultdict(int), defaultdict(list)
    print(f"\nMonte Carlo: {n} runs × {len(classes)} strategies …", end="", flush=True)
    for i in range(n):
        agents = [cls(cls.__name__) for cls in classes]
        pool = [p.copy() for p in players]
        run_auction(pool, agents, verbose=False)
        sim = {a.name: a.season_score() for a in agents}
        top = max(sim.values())
        for nm, sc in sim.items():
            scores[nm].append(sc)
            if sc == top:
                wins[nm] += 1
        if (i + 1) % 200 == 0:
            print(f"{i + 1}…", end="", flush=True)

    print(" done!\n")
    print("═" * 72)
    print(f"  📊  MONTE CARLO  ({n} runs, {len(classes)} strategies)")
    print("═" * 72)
    print(f"  {'Strategy':<30} {'Win%':>6}  {'AvgPts':>8}  {'StdDev':>7}  Chart")
    ranked = sorted(classes, key=lambda c: wins[c.__name__], reverse=True)
    for c in ranked:
        nm = c.__name__
        wp = wins[nm] / n * 100
        avg = sum(scores[nm]) / n
        std = (sum((s - avg) ** 2 for s in scores[nm]) / n) ** 0.5
        bar = "█" * max(1, int(wp / 3))
        print(f"  {nm:<30} {wp:>5.1f}%  {avg:>8.1f}  {std:>7.1f}  {bar}")
    print("═" * 72)
    print(
        f"\n  🏆 DOMINANT STRATEGY: {ranked[0].__name__}  "
        f"({wins[ranked[0].__name__] / n * 100:.1f}% wins)\n"
    )


# ──────────────────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monte-carlo", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--strategies",
        nargs="+",
        type=int,
        help=(
            "Indices 0-8 (default: all 9).\n"
            "0 StarChaser  1 ValueInvestor  2 DynamicMaximizer\n"
            "3 ApexPredator  4 Barbell  5 Vampire\n"
            "6 TierSniper  7 NominGambler  8 PosArbitrageur"
        ),
    )
    args = p.parse_args()

    if not HAS_PANDAS:
        print("pip install pandas numpy")
        return

    print("Loading 2024 IPL player pool…")
    players = load_player_pool()
    t_counts = {t: sum(1 for p in players if p["tier"] == t) for t in [1, 2, 3]}
    print(
        f"  {len(players)} players | "
        f"Elite(T1):{t_counts[1]}  Solid(T2):{t_counts[2]}  Depth(T3):{t_counts[3]} | "
        f"top: {players[0]['player_name']} ({players[0]['projected_points']:.0f}pts)\n"
    )

    indices = args.strategies if args.strategies else list(range(len(ALL_STRATEGIES)))

    if args.monte_carlo > 0:
        run_monte_carlo(players, args.monte_carlo, indices)
    else:
        agents = [ALL_STRATEGIES[i](ALL_STRATEGIES[i].__name__) for i in indices]
        verbose = not args.quiet
        if verbose:
            print("🏏 LIVE ASCENDING-BID AUCTION")
            print(
                f"   Purse ₹120Cr | Roster 13–15 | Pool {len(players)} | {len(agents)} managers"
            )
            print("═" * 72)
        unsold = run_auction([p.copy() for p in players], agents, verbose)
        print_results(agents, unsold)


if __name__ == "__main__":
    main()
