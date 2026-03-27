"""
engine/scoring.py
==================
Single source of truth for fantasy point calculation.
Reads scoring_rules.json via Polars vectorised expressions.
Replaces the duplicated `_calc_fp()` in full_simulation_claude.py.

Public API:
    score_matches(df, rules_path)  → pl.DataFrame with match_fantasy_points column
    aggregate_season(df)           → pl.DataFrame aggregated per (player_name, ipl_year)
    load_rules(rules_path)         → list[dict]
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

# ── Column name mapping: JSON stat key → CSV column name ────────────────────
STAT_COL_MAP: dict[str, str] = {
    "RUNS": "runs_scored",
    "FOURS": "four_count",
    "SIXES": "six_count",
    "BALLS_FACED": "balls_faced",
    "STRIKE_RATE": "strike_rate",
    "WICKETS": "wickets",
    "BOWLED_LBW": "bowled_lbw_wickets",
    "MAIDENS": "maiden_count",
    "ECONOMY": "economy",
    "BALLS_BOWLED": "balls_bowled",
    "CATCHES": "catches_caught",
    "RUNOUTS": "runouts",
}

_DEFAULT_RULES_PATH = Path(__file__).parent.parent / "scoring_rules.json"


def load_rules(rules_path: str | Path | None = None) -> list[dict]:
    path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
    with open(path) as f:
        return json.load(f)


def score_matches(
    df: pl.DataFrame,
    rules_path: str | Path | None = None,
) -> pl.DataFrame:
    """
    Add a `match_fantasy_points` column to a match-level DataFrame.
    The DataFrame must contain the columns referenced in STAT_COL_MAP.
    """
    rules = load_rules(rules_path)

    # 1. Derived stats (strike rate, economy)
    df = df.with_columns(
        [
            pl.when(pl.col("balls_faced") > 0)
            .then((pl.col("runs_scored") / pl.col("balls_faced")) * 100.0)
            .otherwise(0.0)
            .alias("strike_rate"),
            (pl.col("balls_bowled") / 6.0).alias("overs_bowled"),
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("overs_bowled") > 0)
        .then(pl.col("runs_given") / pl.col("overs_bowled"))
        .otherwise(0.0)
        .alias("economy")
    )

    # 2. Compile JSON rules → Polars expressions
    point_exprs: list[pl.Expr] = []
    for rule in rules:
        stat_key = rule["stat"]
        col_name = STAT_COL_MAP.get(stat_key)
        if not col_name:
            continue

        base_cond = pl.lit(True)
        if "condition" in rule:
            cond = rule["condition"]
            cond_col = STAT_COL_MAP.get(cond["stat"])
            if cond_col:
                if cond.get("min") is not None:
                    base_cond = base_cond & (pl.col(cond_col) >= cond["min"])
                if cond.get("max") is not None:
                    base_cond = base_cond & (pl.col(cond_col) < cond["max"])

        rule_type = rule["type"]
        rule_pts = rule["points"]

        if rule_type == "PER_UNIT":
            expr = pl.when(base_cond).then(pl.col(col_name) * rule_pts).otherwise(0)
            point_exprs.append(expr)

        elif rule_type == "RANGE":
            rng_cond = base_cond
            if rule.get("min") is not None:
                rng_cond = rng_cond & (pl.col(col_name) >= rule["min"])
            if rule.get("max") is not None:
                rng_cond = rng_cond & (pl.col(col_name) < rule["max"])
            point_exprs.append(pl.when(rng_cond).then(rule_pts).otherwise(0))

        elif rule_type == "MILESTONE":
            threshold = rule.get("threshold")
            if rule.get("exact"):
                ms_cond = base_cond & (pl.col(col_name) == threshold)
                # Duck: -10 pts only if batter actually faced a ball or was dismissed
                if stat_key == "RUNS" and threshold == 0:
                    duck_safe = (pl.col("balls_faced") > 0) | (pl.col("is_out") == 1)
                    ms_cond = ms_cond & duck_safe
                point_exprs.append(pl.when(ms_cond).then(rule_pts).otherwise(0))

    # 3. Sum all rule expressions horizontally
    df = df.with_columns(
        pl.sum_horizontal(point_exprs).alias("match_fantasy_points")
    )
    return df


def aggregate_season(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate match_fantasy_points per (ipl_year, player_id, player_name).
    Returns points_per_match, points_stddev, total stats used for role classification.
    """
    return (
        df.group_by(["ipl_year", "player_id", "player_name"])
        .agg(
            [
                pl.col("match_id").n_unique().alias("matches_played"),
                pl.col("match_fantasy_points").sum().alias("total_points"),
                pl.col("match_fantasy_points").mean().alias("points_per_match"),
                pl.col("match_fantasy_points").std().alias("points_stddev"),
                pl.col("runs_scored").sum().alias("total_runs"),
                pl.col("wickets").sum().alias("total_wkts"),
                pl.col("catches_caught").sum().alias("total_catches"),
                pl.col("balls_bowled").sum().alias("total_balls_bowled"),
            ]
        )
        .with_columns(pl.col("points_stddev").fill_null(30.0))
        .sort(["ipl_year", "total_points"], descending=[False, True])
    )


def aggregate_all_time(df: pl.DataFrame) -> pl.DataFrame:
    """Career aggregation across all seasons."""
    return (
        df.group_by(["player_id", "player_name"])
        .agg(
            [
                pl.col("match_id").n_unique().alias("total_matches"),
                pl.col("match_fantasy_points").sum().alias("all_time_points"),
                pl.col("match_fantasy_points").mean().alias("all_time_ppm"),
                pl.col("match_fantasy_points").std().alias("points_stddev"),
                pl.col("runs_scored").sum().alias("total_runs"),
                pl.col("wickets").sum().alias("total_wkts"),
                pl.col("catches_caught").sum().alias("total_catches"),
                pl.col("balls_bowled").sum().alias("total_balls_bowled"),
            ]
        )
        .with_columns(pl.col("points_stddev").fill_null(30.0))
        .sort("all_time_points", descending=True)
    )
