"""
IPL Auction Strategy Engine
============================
Unified engine package. Import order:
  scoring   → pool → overrides → strategies → auction → simulation → export
"""
from .scoring import score_matches, aggregate_season
from .pool import build_pool, WeightedSeasonModel, RecentFormModel, CustomFunctionModel
from .overrides import load_overrides, apply_overrides
from .strategies import STRATEGY_REGISTRY, Manager
from .auction import run_auction, build_state
from .simulation import run_monte_carlo, evolve_strategy
from .export import export_pool, export_insights

__all__ = [
    "score_matches", "aggregate_season",
    "build_pool", "WeightedSeasonModel", "RecentFormModel", "CustomFunctionModel",
    "load_overrides", "apply_overrides",
    "STRATEGY_REGISTRY", "Manager",
    "run_auction", "build_state",
    "run_monte_carlo", "evolve_strategy",
    "export_pool", "export_insights",
]
