"""
engine/pool.py
===============
Pluggable player pool generation.

ProjectionModel ABC — three concrete implementations:
  WeightedSeasonModel   season-weighted avg pts (default, mirrors full_simulation_claude.py)
  RecentFormModel       last-N-matches window
  CustomFunctionModel   caller-supplied Polars transform

Public API:
  build_pool(csv_path, model, *, min_matches, pool_size, num_teams,
             role_requirements, vorp_method, rules_path, overrides_path)
      → list[dict]  (player records, sorted by projected_points desc)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from .scoring import score_matches, load_rules
from .overrides import apply_overrides

# ── Constants (canonical from full_simulation_claude.py) ────────────────────
TIER1_FLOOR = 80.0   # avg pts/match → Elite
TIER2_FLOOR = 58.0   # avg pts/match → Solid, below = Depth

_DEFAULT_CSV   = Path(__file__).parent.parent / "ipl_player_stats.csv"
_DEFAULT_RULES = Path(__file__).parent.parent / "scoring_rules.json"


# ── Role classification (from _classify_role in full_simulation_claude.py) ──
def classify_role(
    total_runs: int,
    total_wkts: int,
    total_catches: int,
    total_balls_bowled: int,
) -> str:
    if total_catches >= 8 and total_runs < 150 and total_wkts == 0:
        return "WK"
    if total_runs >= 250 and total_wkts >= 8:
        return "AR"
    if total_balls_bowled >= 120 and total_runs < 200:
        return "BOWL"
    if total_balls_bowled >= 60 and total_wkts >= 6 and total_runs >= 100:
        return "AR"
    return "BAT"


def _classify_roles_df(df: pl.DataFrame) -> pl.DataFrame:
    """Vectorised role classification for an aggregated player DataFrame."""
    return df.with_columns(
        pl.struct(
            ["total_runs", "total_wkts", "total_catches", "total_balls_bowled"]
        )
        .map_elements(
            lambda r: classify_role(
                r["total_runs"],
                r["total_wkts"],
                r["total_catches"],
                r["total_balls_bowled"],
            ),
            return_dtype=pl.Utf8,
        )
        .alias("role")
    )


# ── VORP calculation ─────────────────────────────────────────────────────────
def _calculate_vorp(
    df: pl.DataFrame,
    method: str,
    num_teams: int,
    role_requirements: dict[str, int],
) -> pl.DataFrame:
    """
    Add a `vorp` column to an aggregated player DataFrame.

    method="quantile"   replacement level = Q25 of projected_points per role
                        (source: full_simulation_claude.py)
    method="depth"      replacement level = projected_points of the
                        (num_teams × role_req)-th best player per role
                        (source: baseline.ipynb)
    """
    roles = ["BAT", "BOWL", "AR", "WK"]
    rep_levels: dict[str, float] = {}

    for role in roles:
        role_df = df.filter(pl.col("role") == role).sort(
            "projected_points", descending=True
        )
        if method == "depth":
            depth = num_teams * role_requirements.get(role, 1)
            rep_levels[role] = (
                role_df["projected_points"][depth]
                if len(role_df) > depth
                else 50.0
            )
        else:  # quantile
            pts = role_df["projected_points"]
            rep_levels[role] = float(pts.quantile(0.25)) if len(pts) > 0 else 50.0

    rep_df = pl.DataFrame(
        {"role": list(rep_levels.keys()), "rep_level": list(rep_levels.values())}
    )
    df = df.join(rep_df, on="role", how="left")
    df = df.with_columns(
        pl.when(pl.col("projected_points") > pl.col("rep_level"))
        .then(pl.col("projected_points") - pl.col("rep_level"))
        .otherwise(0.0)
        .alias("vorp")
    )
    return df.drop("rep_level")


# ── Base scorer: raw stats → scored match log ────────────────────────────────
def _load_and_score(csv_path: Path, rules_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path, infer_schema_length=5000)
    # Ensure ipl_year is a string for consistent filtering
    if df["ipl_year"].dtype != pl.Utf8:
        df = df.with_columns(pl.col("ipl_year").cast(pl.Utf8))
    return score_matches(df, rules_path)


# ── Projection Models ────────────────────────────────────────────────────────
class ProjectionModel(ABC):
    """Base class. Subclasses produce a projected_points value per (player_name)."""

    @abstractmethod
    def project(self, scored_df: pl.DataFrame) -> pl.DataFrame:
        """
        Input:  scored match-level DataFrame (has match_fantasy_points column)
        Output: aggregated player DataFrame with at minimum:
                player_name, projected_points, std_dev,
                total_runs, total_wkts, total_catches, total_balls_bowled,
                matches_played
        """


class WeightedSeasonModel(ProjectionModel):
    """
    Weighted average of per-season avg_pts.
    season_weights: {"2024": 1.0, "2023": 0.6, "2022": 0.3}
    Most recent season should typically have weight 1.0.
    """

    def __init__(self, season_weights: dict[str, float]):
        self.season_weights = season_weights

    def project(self, scored_df: pl.DataFrame) -> pl.DataFrame:
        seasons = list(self.season_weights.keys())
        df = scored_df.filter(pl.col("ipl_year").is_in(seasons))

        season_agg = (
            df.group_by(["ipl_year", "player_name"])
            .agg(
                [
                    pl.col("match_id").n_unique().alias("matches_in_season"),
                    pl.col("match_fantasy_points").mean().alias("avg_pts"),
                    pl.col("match_fantasy_points").std().alias("std_pts"),
                    pl.col("runs_scored").sum().alias("total_runs"),
                    pl.col("wickets").sum().alias("total_wkts"),
                    pl.col("catches_caught").sum().alias("total_catches"),
                    pl.col("balls_bowled").sum().alias("total_balls_bowled"),
                ]
            )
            .with_columns(
                pl.col("ipl_year")
                .map_elements(lambda y: self.season_weights.get(y, 0.0), return_dtype=pl.Float64)
                .alias("weight")
            )
        )

        # Weighted average across seasons
        weighted = (
            season_agg.group_by("player_name")
            .agg(
                [
                    (
                        (pl.col("avg_pts") * pl.col("weight")).sum()
                        / pl.col("weight").sum()
                    ).alias("projected_points"),
                    pl.col("std_pts").mean().alias("std_dev"),
                    pl.col("matches_in_season").sum().alias("matches_played"),
                    pl.col("total_runs").sum().alias("total_runs"),
                    pl.col("total_wkts").sum().alias("total_wkts"),
                    pl.col("total_catches").sum().alias("total_catches"),
                    pl.col("total_balls_bowled").sum().alias("total_balls_bowled"),
                ]
            )
            .with_columns(pl.col("std_dev").fill_null(30.0))
        )
        return weighted


class RecentFormModel(ProjectionModel):
    """
    Uses each player's most recent `last_n_matches` to compute projected_points.
    Ideal for players with strong recent momentum.
    """

    def __init__(self, last_n_matches: int = 10):
        self.last_n = last_n_matches

    def project(self, scored_df: pl.DataFrame) -> pl.DataFrame:
        # Sort by date proxy (ipl_year + match_id) and take last N per player
        df = scored_df.sort(["ipl_year", "match_id"])
        recent = (
            df.group_by("player_name")
            .agg(
                [
                    pl.col("match_fantasy_points").tail(self.last_n).mean().alias("projected_points"),
                    pl.col("match_fantasy_points").tail(self.last_n).std().alias("std_dev"),
                    pl.len().alias("matches_played"),
                    pl.col("runs_scored").sum().alias("total_runs"),
                    pl.col("wickets").sum().alias("total_wkts"),
                    pl.col("catches_caught").sum().alias("total_catches"),
                    pl.col("balls_bowled").sum().alias("total_balls_bowled"),
                ]
            )
            .with_columns(pl.col("std_dev").fill_null(30.0))
        )
        return recent


class CustomFunctionModel(ProjectionModel):
    """
    Escape hatch: caller provides a function that transforms the scored match
    log directly into the required aggregated DataFrame.

    fn signature: (scored_df: pl.DataFrame) -> pl.DataFrame
    The output must contain: player_name, projected_points, std_dev,
    matches_played, total_runs, total_wkts, total_catches, total_balls_bowled
    """

    def __init__(self, fn):
        self.fn = fn

    def project(self, scored_df: pl.DataFrame) -> pl.DataFrame:
        return self.fn(scored_df)


# ── Main entry point ─────────────────────────────────────────────────────────
def build_pool(
    csv_path: str | Path | None = None,
    model: ProjectionModel | None = None,
    *,
    min_matches: int = 5,
    pool_size: int = 80,
    num_teams: int = 6,
    role_requirements: dict[str, int] | None = None,
    vorp_method: str = "quantile",   # "quantile" | "depth"
    rules_path: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> list[dict]:
    """
    Full pipeline: CSV → score → project → classify roles → VORP → tier → pool.
    Returns list of player dicts sorted by projected_points descending.
    """
    csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV
    rules_path = Path(rules_path) if rules_path else _DEFAULT_RULES
    role_requirements = role_requirements or {"BAT": 5, "BOWL": 4, "AR": 2, "WK": 2}

    if model is None:
        model = WeightedSeasonModel({"2024": 1.0})

    # 1. Load + score match log
    scored = _load_and_score(csv_path, rules_path)

    # 2. Project → aggregated player stats
    agg = model.project(scored)

    # 3. Filter minimum matches
    agg = agg.filter(pl.col("matches_played") >= min_matches)

    # 4. Classify roles
    agg = _classify_roles_df(agg)

    # 5. Round projected_points and std_dev
    agg = agg.with_columns(
        [
            pl.col("projected_points").round(1),
            pl.col("std_dev").round(1),
        ]
    )

    # 6. VORP
    agg = _calculate_vorp(agg, vorp_method, num_teams, role_requirements)
    agg = agg.with_columns(pl.col("vorp").round(1))

    # 7. Tier assignment
    agg = agg.with_columns(
        pl.when(pl.col("projected_points") > TIER1_FLOOR)
        .then(1)
        .when(pl.col("projected_points") > TIER2_FLOOR)
        .then(2)
        .otherwise(3)
        .alias("tier")
    )

    # 8. Take top N by projected_points
    agg = agg.sort("projected_points", descending=True).head(pool_size)

    # 9. Convert to list of dicts
    pool = agg.select(
        ["player_name", "role", "projected_points", "std_dev", "vorp", "tier", "matches_played"]
    ).rename({"matches_played": "matches"}).to_dicts()

    # 10. Apply manual overrides
    pool = apply_overrides(pool, overrides_path)

    return pool
