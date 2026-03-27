"""
engine/export.py
=================
Generate JSON files consumed by the HTML advisor and server.

Public API:
  export_pool(pool, path)               → writes player_pool.json
  export_insights(pool, mc_results, path) → writes insights.json
  load_pool(path)                         → list[dict]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import MCResult

_DEFAULT_POOL_PATH    = Path(__file__).parent.parent / "player_pool.json"
_DEFAULT_INSIGHTS_PATH = Path(__file__).parent.parent / "insights.json"


def export_pool(pool: list[dict], path: str | Path | None = None) -> Path:
    """Write the player pool to player_pool.json."""
    p = Path(path) if path else _DEFAULT_POOL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(pool, f, indent=2)
    return p


def load_pool(path: str | Path | None = None) -> list[dict]:
    """Load a previously exported player pool."""
    p = Path(path) if path else _DEFAULT_POOL_PATH
    with open(p) as f:
        return json.load(f)


def export_insights(
    pool: list[dict],
    mc_results: "dict[str, MCResult]",
    path: str | Path | None = None,
) -> Path:
    """
    Write insights.json:
      {
        "pool_size":        int,
        "tier_counts":      {1: N, 2: N, 3: N},
        "strategies": {
          "StrategyName": {
            "win_rate":   float,
            "avg_score":  float,
            "std_score":  float,
            "wins":       int,
            "runs":       int,
          }
        }
      }
    """
    p = Path(path) if path else _DEFAULT_INSIGHTS_PATH

    tier_counts = {1: 0, 2: 0, 3: 0}
    for player in pool:
        tier_counts[player.get("tier", 3)] += 1

    strategy_data = {}
    for name, r in mc_results.items():
        strategy_data[name] = {
            "win_share":             round(r.win_share, 4),
            "conditional_win_rate":  round(r.conditional_win_rate, 4),
            "avg_score":             round(r.avg_score, 2),
            "std_score":             round(r.std_score, 2),
            "wins":                  r.wins,
            "participations":        r.participations,
            "total_auctions":        r.total_auctions,
            "budget_utilization":    round(r.budget_utilization, 3),
        }

    payload = {
        "pool_size":   len(pool),
        "tier_counts": {str(k): v for k, v in tier_counts.items()},
        "strategies":  strategy_data,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    return p
