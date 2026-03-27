"""
engine/auction.py
==================
Clean ascending-bid auction engine — single source of truth.
Replaces both `LiveIPLAuction` in baseline.ipynb and `run_auction` in full_simulation_claude.py.

Faithful to full_simulation_claude.py contract:
  - build_state() produces the exact state dict every strategy expects
  - Noise: uniform(-0.25, +0.5) per WTP, only when raw WTP > 0.5
  - Bid loop: randomised agent order each pass; increment = ₹0.25
  - Bid starts at ₹0.5 minimum
  - Fire-sale: triggered after each sale; 75% recovery
  - Unsold pool re-entered in subsequent rounds
  - Stops when zero players bought in a round

Public API:
  build_state(agents, remaining) → dict
  run_auction(pool, agents, verbose) → list[dict]  (unsold players)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .strategies import Manager

from .pool import order_pool

BID_INCREMENT = 0.25
MIN_BID = 0.5
NOISE_LO = -0.25
NOISE_HI = 0.50


def build_state(agents: list["Manager"], remaining: list[dict]) -> dict:
    """
    Compute the auction state dict delivered to every strategy's willingness_to_pay().

    Keys:
      players_remaining  int
      cr_per_vorp        float  (available spendable cash / sum of remaining VORP)
      pool_by_role       dict[role, list[player]]  sorted by projected_points desc
      agent_states       dict[name, {purse, slots, mandatory, name}]
    """
    avail = sum(
        max(0.0, a.purse - 1.0 * max(0, a.mandatory - 1)) for a in agents
    )
    rem_vorp = sum(p.get("vorp", 0.0) for p in remaining)
    return {
        "players_remaining": len(remaining),
        "cr_per_vorp": avail / rem_vorp if rem_vorp > 0 else 0.0,
        "pool_by_role": {
            role: sorted(
                [p for p in remaining if p["role"] == role],
                key=lambda x: x["projected_points"],
                reverse=True,
            )
            for role in ["BAT", "BOWL", "AR", "WK"]
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


def run_auction(
    pool: list[dict],
    agents: list["Manager"],
    verbose: bool = True,
    purchase_log: list[dict] | None = None,
) -> list[dict]:
    """
    Run a full ascending-bid auction.

    Args:
        pool:    list of player dicts (will be shuffled in-place copy)
        agents:  list of Manager instances (will be reset at start)
        verbose: whether to print lot-by-lot results

    Returns:
        list of unsold player dicts
    """
    for agent in agents:
        agent.reset()

    # Round 1: IPL-realistic set ordering (BAT→AR→WK→BOWL, capped first,
    # randomised within each base-price bracket of ~4 players).
    # Re-entry rounds (2+) use a simple shuffle — the real IPL accelerated auction.
    lot_pool = order_pool(list(pool))
    unsold: list[dict] = []
    rnd = 1

    while lot_pool:
        if verbose and rnd > 1:
            print(f"\n{'─' * 64}\n  ♻  ROUND {rnd} — {len(lot_pool)} lots re-entered\n{'─' * 64}")

        current_lots = list(lot_pool)
        lot_pool = []
        bought = 0

        for player in current_lots:
            remaining = current_lots + unsold + lot_pool
            state = build_state(agents, remaining)
            floor = player.get("base_price", MIN_BID)  # tier-based bid floor
            active = [a for a in agents if a.slots > 0 and a.max_bid >= floor]

            # Compute WTP for each active agent (with noise)
            wtp: dict[str, float] = {}
            for a in active:
                raw = a.willingness_to_pay(player, state, rnd)
                noise = random.uniform(NOISE_LO, NOISE_HI) if raw > MIN_BID else 0.0
                wtp[a.name] = min(raw + noise, a.max_bid)

            if not any(v >= floor for v in wtp.values()):
                unsold.append(player)
                if verbose:
                    print(f"  ⛔ UNSOLD  {player['player_name']:<22} ({player['role']})")
                continue

            # Ascending bid loop — starts at tier base price
            bid = 0.0
            winner: str | None = None
            going = True
            while going:
                going = False
                for a in random.sample(active, len(active)):
                    if a.name == winner:
                        continue
                    ask = bid + BID_INCREMENT if winner else floor
                    if wtp[a.name] >= ask and ask <= a.max_bid:
                        bid, winner, going = ask, a.name, True
                        break

            assert winner is not None
            winning_agent = next(a for a in agents if a.name == winner)
            winning_agent.buy(player, bid)
            if purchase_log is not None:
                purchase_log.append({
                    "strategy": winner,
                    "player": player["player_name"],
                    "role": player["role"],
                    "tier": player.get("tier", 3),
                    "price": bid,
                    "base_price": player.get("base_price", 1.0),
                    "projected_points": player["projected_points"],
                })
            bought += 1

            if verbose:
                tier_label = f"T{player.get('tier', '?')}"
                print(
                    f"  🏏 {player['player_name']:<22} {player['role']:<4} {tier_label}"
                    f" → {winner:<30} ₹{bid:.2f}Cr  [{player['projected_points']:.0f}pts]"
                )

            # Fire-sale check
            for a in agents:
                returned = a.fire_sale()
                if returned:
                    lot_pool.append(returned)
                    if verbose:
                        print(f"  🔥 {a.name} fire-sales {returned['player_name']}")

        if bought == 0:
            break
        # If any agent is still short of min_roster and there are unsold players, re-run.
        # Round 2+ = accelerated auction: randomise unsold order (real IPL practice).
        if any(a.mandatory > 0 for a in agents) and unsold:
            random.shuffle(unsold)
            lot_pool.extend(unsold)
            unsold = []
            rnd += 1
        else:
            break

    return unsold


def print_results(agents: list["Manager"], unsold: list[dict]) -> None:
    """Pretty-print final standings."""
    print("\n" + "═" * 72)
    print("  🏆  FINAL RESULTS")
    print("═" * 72)
    ranked = sorted(agents, key=lambda a: a.season_score(), reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for i, a in enumerate(ranked):
        m = medals[i] if i < 3 else "  "
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
        marker = "✅" if i <= 11 else "  "
        print(
            f"  {marker}{i:>2}. {p['player_name']:<22} {p['role']:<4}"
            f" {p['projected_points']:>6.1f}pts  ₹{e['price']:>5.2f}Cr"
        )
    if unsold:
        print(f"\n  ⛔ {len(unsold)} unsold")
    print("═" * 72)
