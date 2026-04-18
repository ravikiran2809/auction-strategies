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

import json
import math
import random
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from .scoring import score_matches, load_rules
from .overrides import apply_overrides

# ── Constants (canonical from full_simulation_claude.py) ────────────────────
TIER1_FLOOR = 80.0   # avg pts/match → Elite
TIER2_FLOOR = 58.0   # avg pts/match → Solid, below = Depth

_DEFAULT_CSV         = Path(__file__).parent.parent / "ipl_player_stats.csv"
_DEFAULT_RULES       = Path(__file__).parent.parent / "scoring_rules.json"
_DEFAULT_ROLES_PATH  = Path(__file__).parent.parent / "player_roles.json"
_DEFAULT_MASTER_PATH = Path(__file__).parent.parent / "player_master.json"

# Floor projected_points for players with no CSV history
_FLOOR_PTS = {"BAT": 40.0, "AR": 40.0, "BOWL": 38.0, "WK": 35.0}


# ── Authoritative role lookup ────────────────────────────────────────────────
def _load_role_lookup(
    roles_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load player_roles.json and build two indices:
      full_index  : {full_name_lower → role}   (e.g. "sunil narine" → "AR")
      last_index  : {last_name_lower → role}   (e.g. "narine" → "AR")

    The last-name index handles CSV abbreviated names like "SP Narine".
    Collisions in last_index (two players with the same surname, different
    roles) are resolved by marking that surname as ambiguous ("") so the
    heuristic fallback is used instead.
    """
    p = roles_path or _DEFAULT_ROLES_PATH
    import json
    if not p.exists():
        return {}, {}
    with open(p) as f:
        data = json.load(f)

    full_index: dict[str, str] = {}
    last_count: dict[str, list[str]] = {}  # last_name → [role, ...]
    for name, role in data.items():
        if name.startswith("_"):
            continue
        full_index[name.lower()] = role
        last = name.split()[-1].lower()
        last_count.setdefault(last, []).append(role)

    last_index: dict[str, str] = {}
    for last, roles in last_count.items():
        unique = set(roles)
        last_index[last] = roles[0] if len(unique) == 1 else ""  # "" = ambiguous

    return full_index, last_index


def _lookup_role(
    player_name: str,
    full_index: dict[str, str],
    last_index: dict[str, str],
) -> str | None:
    """
    Try to resolve the role for *player_name* (CSV format, e.g. "SP Narine").

    Resolution order:
      1. Exact full-name match (case-insensitive)
      2. Last-name match       (works for abbreviated CSV names)
      3. None → caller uses stat heuristic
    """
    key = player_name.strip().lower()
    if key in full_index:
        return full_index[key]
    last = key.split()[-1]
    role = last_index.get(last, "")
    return role if role else None  # "" (ambiguous) treated as miss


# ── Stat-heuristic fallback ──────────────────────────────────────────────────
def classify_role(
    total_runs: int,
    total_wkts: int,
    total_catches: int,
    total_balls_bowled: int,
) -> str:
    """Stat-heuristic fallback — only used when lookup fails."""
    if total_catches >= 8 and total_runs < 150 and total_wkts == 0:
        return "WK"
    if total_runs >= 250 and total_wkts >= 8:
        return "AR"
    if total_balls_bowled >= 120 and total_runs < 200:
        return "BOWL"
    if total_balls_bowled >= 60 and total_wkts >= 6 and total_runs >= 100:
        return "AR"
    return "BAT"


def _classify_roles_df(
    df: pl.DataFrame,
    roles_path: Path | None = None,
) -> pl.DataFrame:
    """
    Assign roles to each player in the aggregated DataFrame.

    Priority:
      1. Authoritative lookup from player_roles.json (by full name or last name)
      2. Stat heuristic as fallback for players not in the lookup
    """
    full_index, last_index = _load_role_lookup(roles_path)

    def _classify(r: dict) -> str:
        looked_up = _lookup_role(r["player_name"], full_index, last_index)
        if looked_up:
            return looked_up
        return classify_role(
            r["total_runs"],
            r["total_wkts"],
            r["total_catches"],
            r["total_balls_bowled"],
        )

    return df.with_columns(
        pl.struct(
            ["player_name", "total_runs", "total_wkts", "total_catches", "total_balls_bowled"]
        )
        .map_elements(_classify, return_dtype=pl.Utf8)
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
def _load_and_score(
    csv_path: Path,
    rules_path: Path,
    extra_csv_paths: list[Path] | None = None,
) -> pl.DataFrame:
    """Load one or more CSV files, concatenate, then score.

    extra_csv_paths: additional season CSV files to merge (e.g. 2026 partial season).
    All CSVs must have the same schema as ipl_player_stats.csv.
    """
    def _read_csv(p: Path) -> pl.DataFrame:
        df = pl.read_csv(p, infer_schema_length=5000)
        if df["ipl_year"].dtype != pl.Utf8:
            df = df.with_columns(pl.col("ipl_year").cast(pl.Utf8))
        return df

    frames = [_read_csv(csv_path)]
    for ep in (extra_csv_paths or []):
        frames.append(_read_csv(ep))

    combined = pl.concat(frames) if len(frames) > 1 else frames[0]
    return score_matches(combined, rules_path)


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


class NormalisedSeasonModel(ProjectionModel):
    """
    Combines multiple seasons with per-season normalisation for incomplete seasons.

    For incomplete seasons (e.g. 2026 mid-season) two adjustments are applied:

    1. Season-level scale: avg_pts are projected to full-season equivalent via
       full_season_matches / max_team_matches_in_season (capped at 2×).

    2. Player-level Bayesian shrinkage: players with few matches in the incomplete
       season are pulled toward the season's role-mean before scaling.

         shrunk_avg = (n * raw_avg + k * role_mean) / (n + k)

       where n = player match count, k = shrinkage_k (default 4).
       A player with n=1 match: weight 1/(1+4)=20% own data, 80% role mean.
       A player with n=6 matches: weight 6/10=60% own data.
       A player with n≥14 matches: nearly unaffected (>77% own data).

    This prevents single-match blowout scores (e.g. Dewald Brewis scoring 80pts
    in his only 2026 game) from projecting as elite full-season performers.

    Args:
        season_weights:      {year: weight}  — relative importance per season.
        full_season_matches: Expected total matches in a complete season (default 74
                             for IPL — 10 teams × 14 home/away + 4 playoffs).
        incomplete_seasons:  Set of years to treat as incomplete (normalised).
                             If None, a season is auto-detected as incomplete if
                             max team match count < 0.8 × full_season_matches.
        shrinkage_k:         Bayesian shrinkage constant for incomplete seasons.
                             Higher = more regression to mean for sparse players.
        min_incomplete_matches: Players with fewer matches than this in an incomplete
                             season are excluded from that season's projection and
                             fall back to prior seasons only.
    """

    def __init__(
        self,
        season_weights: dict[str, float],
        full_season_matches: int = 74,
        incomplete_seasons: set[str] | None = None,
        shrinkage_k: int = 4,
        min_incomplete_matches: int = 2,
    ):
        self.season_weights = season_weights
        self.full_season_matches = full_season_matches
        self.incomplete_seasons = incomplete_seasons
        self.shrinkage_k = shrinkage_k
        self.min_incomplete_matches = min_incomplete_matches

    def project(self, scored_df: pl.DataFrame) -> pl.DataFrame:
        seasons = list(self.season_weights.keys())
        df = scored_df.filter(pl.col("ipl_year").is_in(seasons))

        # Compute max team match count per season (proxy: max individual match count)
        season_max_matches = (
            df.group_by(["ipl_year", "player_name"])
            .agg(pl.col("match_id").n_unique().alias("player_matches"))
            .group_by("ipl_year")
            .agg(pl.col("player_matches").max().alias("max_matches"))
        )
        max_matches_map: dict[str, int] = {
            row["ipl_year"]: row["max_matches"]
            for row in season_max_matches.to_dicts()
        }

        # Compute role membership per player for shrinkage (role from full history)
        role_df = _classify_roles_df(
            scored_df.group_by("player_name")
            .agg([
                pl.col("runs_scored").sum().alias("total_runs"),
                pl.col("wickets").sum().alias("total_wkts"),
                pl.col("balls_bowled").sum().alias("total_balls_bowled"),
                pl.col("catches_caught").sum().alias("total_catches"),
            ])
            .with_columns([
                pl.lit(0.0).alias("projected_points"),
                pl.lit(0.0).alias("std_dev"),
                pl.lit(0).alias("matches_played"),
            ])
        ).select(["player_name", "role"])
        role_map: dict[str, str] = {r["player_name"]: r["role"] for r in role_df.to_dicts()}

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

        # Compute role-mean avg_pts per season (used as shrinkage prior)
        # Map roles onto the season_agg rows
        season_agg = season_agg.with_columns(
            pl.col("player_name")
            .map_elements(lambda n: role_map.get(n, "BAT"), return_dtype=pl.String)
            .alias("role")
        )
        role_season_means: dict[tuple[str, str], float] = {}
        for row in (
            season_agg.group_by(["ipl_year", "role"])
            .agg(pl.col("avg_pts").mean().alias("role_mean"))
            .to_dicts()
        ):
            role_season_means[(row["ipl_year"], row["role"])] = row["role_mean"]

        def _is_incomplete(yr: str) -> bool:
            if self.incomplete_seasons is not None:
                return yr in self.incomplete_seasons
            return max_matches_map.get(yr, self.full_season_matches) < self.full_season_matches * 0.8

        def _normalise_avg(row: dict) -> float:
            yr = row["ipl_year"]
            avg = row["avg_pts"]
            n = row["matches_in_season"]
            role = row.get("role", "BAT")

            if not _is_incomplete(yr):
                return avg

            # Drop players below minimum match threshold for this incomplete season
            if n < self.min_incomplete_matches:
                return float("nan")  # will be filtered out below

            # Bayesian shrinkage toward role mean
            role_mean = role_season_means.get((yr, role), avg)
            k = self.shrinkage_k
            shrunk = (n * avg + k * role_mean) / (n + k)

            # Scale to full-season equivalent (cap at 2×)
            max_m = max_matches_map.get(yr, self.full_season_matches)
            if max_m > 0:
                scale = min(self.full_season_matches / max_m, 2.0)
                return shrunk * scale
            return shrunk

        season_agg = season_agg.with_columns(
            pl.struct(["ipl_year", "avg_pts", "matches_in_season", "role"])
            .map_elements(_normalise_avg, return_dtype=pl.Float64)
            .alias("avg_pts_norm")
        )

        # Drop rows where normalisation returned NaN (below min_incomplete_matches)
        season_agg = season_agg.filter(pl.col("avg_pts_norm").is_not_nan())

        weighted = (
            season_agg.group_by("player_name")
            .agg(
                [
                    (
                        (pl.col("avg_pts_norm") * pl.col("weight")).sum()
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


# ── Cross-season consistency standard deviation ───────────────────────────────
def _compute_consistency_std(
    scored_df: pl.DataFrame,
    agg: pl.DataFrame,
    *,
    history_seasons: list[str] | None = None,
) -> pl.DataFrame:
    """
    Refine std_dev using cross-season consistency.

    Veterans (2+ historical seasons):
      std_dev = max(within_season_std, cross_season_std)
      A boom-bust player who varies greatly year-to-year gets higher std_dev,
      reflecting genuine projection uncertainty.

    New players (1 season only):
      std_dev = within_season_std unchanged — no penalty for being new.
      Playing only 1-2 seasons is itself a signal the selector trusts them;
      we don't artificially discount that.
    """
    if history_seasons is None:
        history_seasons = ["2022", "2023", "2024", "2025"]

    season_avgs = (
        scored_df
        .filter(pl.col("ipl_year").is_in(history_seasons))
        .group_by(["player_name", "ipl_year"])
        .agg(pl.col("match_fantasy_points").mean().alias("season_avg"))
    )
    cross = (
        season_avgs
        .group_by("player_name")
        .agg([
            pl.col("season_avg").std(ddof=1).alias("cross_std"),
            pl.col("season_avg").len().alias("n_seasons"),
        ])
        .with_columns(pl.col("cross_std").fill_null(0.0))
    )
    agg = (
        agg.join(cross, on="player_name", how="left")
        .with_columns([
            pl.col("cross_std").fill_null(0.0),
            pl.col("n_seasons").fill_null(1).cast(pl.Int32),
        ])
    )
    # Veterans: raise std_dev if cross-season variability exceeds within-season noise
    agg = agg.with_columns(
        pl.when(
            (pl.col("n_seasons") >= 2) & (pl.col("cross_std") > pl.col("std_dev"))
        )
        .then(pl.col("cross_std"))
        .otherwise(pl.col("std_dev"))
        .alias("std_dev")
    ).drop(["cross_std", "n_seasons"])
    return agg


# ── Role-aware percentile tier assignment ─────────────────────────────────────
def _assign_tiers_by_role(
    df: pl.DataFrame,
    *,
    t1_frac: float = 0.20,
    t2_frac: float = 0.40,
) -> pl.DataFrame:
    """
    Assign tiers by percentile rank *within each role*.

    Why role-aware instead of absolute thresholds?
    Bowlers structurally score fewer fantasy points than batsmen/ARs.
    An absolute pts threshold (e.g. 80) would classify elite bowlers like
    Bumrah as T2 purely because bowling earns fewer fantasy pts — not because
    they are lower quality. Role-aware percentiles ensure each position has
    proportional elite representation.

    t1_frac : top fraction per role  → T1 elite    (default 20%)
    t2_frac : next fraction per role → T2 quality  (default 40%)
    T3      : bottom 40% per role   → depth
    """
    rows = df.to_dicts()
    by_role: dict[str, list[dict]] = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)

    for group in by_role.values():
        group.sort(key=lambda x: x["projected_points"], reverse=True)
        n = len(group)
        n_t1 = max(1, round(n * t1_frac))
        n_t2 = max(n_t1 + 1, round(n * (t1_frac + t2_frac)))
        for i, p in enumerate(group):
            p["tier"] = 1 if i < n_t1 else (2 if i < n_t2 else 3)

    flattened = [p for grp in by_role.values() for p in grp]
    orig_cols = df.columns
    return pl.DataFrame(flattened).select([*orig_cols, "tier"])


# ── Main entry point ─────────────────────────────────────────────────────────
def build_pool(
    csv_path: str | Path | None = None,
    model: ProjectionModel | None = None,
    *,
    extra_csv_paths: list[str | Path] | None = None,
    min_matches: int = 5,
    pool_size: int = 200,
    num_teams: int = 6,
    role_requirements: dict[str, int] | None = None,
    vorp_method: str = "quantile",   # "quantile" | "depth"
    rules_path: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> list[dict]:
    """
    Full pipeline: CSV -> score -> project -> classify roles -> VORP -> tier -> pool.
    Returns list of player dicts sorted by projected_points descending.

    extra_csv_paths: optional list of additional season CSV files (e.g. 2026 partial).
                     All files must share the same schema as the primary csv_path.
                     Use NormalisedSeasonModel to handle partial seasons correctly.
    """
    csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV
    rules_path = Path(rules_path) if rules_path else _DEFAULT_RULES
    role_requirements = role_requirements or {"BAT": 5, "BOWL": 4, "AR": 2, "WK": 2}
    extra_paths = [Path(p) for p in (extra_csv_paths or [])]

    if model is None:
        # 2025 only for projected_points — avoids diluting current form with stale data.
        # Cross-season std (step 2a) still uses multi-year history for consistency.
        model = WeightedSeasonModel({"2025": 1.0})

    # 1. Load + score match log (all years — needed for cross-season consistency)
    scored = _load_and_score(csv_path, rules_path, extra_paths)

    # 2. Project → aggregated player stats (2025 data only by default)
    agg = model.project(scored)

    # 2a. Refine std_dev with cross-season consistency from full history.
    #     New players (1 season) are NOT penalised — their within-season std is kept.
    agg = _compute_consistency_std(scored, agg)

    # 3. Filter minimum matches (based on the projection season only)
    agg = agg.filter(pl.col("matches_played") >= min_matches)

    # 4. Classify roles
    agg = _classify_roles_df(agg)

    # 5. Round projected_points and std_dev; cap std_dev at projected_points
    #    Per-match std cannot meaningfully exceed average per-match score —
    #    a cap prevents extreme cross-season volatility from making BarbellStrategist
    #    boycott elite players like Bumrah (raw std=103, pts=104 → adj_pts≈1).
    agg = agg.with_columns(
        [
            pl.col("projected_points").round(1),
            pl.min_horizontal("std_dev", "projected_points").round(1).alias("std_dev"),
        ]
    )

    # 6. VORP
    agg = _calculate_vorp(agg, vorp_method, num_teams, role_requirements)
    agg = agg.with_columns(pl.col("vorp").round(1))

    # 7. Tier assignment — role-aware percentile (top 20% per role = T1, next 40% = T2)
    #    This prevents the structural bias where bowlers are always T2/T3 because
    #    they earn fewer raw fantasy points than batsmen and all-rounders.
    agg = _assign_tiers_by_role(agg)

    # 7b. Base price by tier (T1=₹5Cr, T2=₹3Cr, T3=₹1Cr)
    #     Capped at 5 so strategies have room to bid competitively without
    #     the floor alone draining most of their purse on marquee players.
    agg = agg.with_columns(
        pl.when(pl.col("tier") == 1).then(5.0)
        .when(pl.col("tier") == 2).then(3.0)
        .otherwise(1.0)
        .alias("base_price")
    )

    # 7c. Marquee flag — top MARQUEE_SIZE players by projected_points.
    #     These open the auction regardless of role, ensuring superstars
    #     are contested when all franchises still have full wallets.
    agg = agg.sort("projected_points", descending=True)
    marquee_names = set(agg["player_name"].head(MARQUEE_SIZE).to_list())
    agg = agg.with_columns(
        pl.col("player_name")
        .map_elements(lambda n: n in marquee_names, return_dtype=pl.Boolean)
        .alias("is_marquee")
    )

    # 8. Take top N by projected_points
    agg = agg.sort("projected_points", descending=True).head(pool_size)

    # 9. Convert to list of dicts
    pool = agg.select(
        ["player_name", "role", "projected_points", "std_dev",
         "vorp", "tier", "base_price", "is_marquee", "matches_played"]
    ).rename({"matches_played": "matches"}).to_dicts()

    # 10. Enrich with ipl_team and auction_set from player_master.json;
    #     filter out any player not present in the master list.
    master_path = _DEFAULT_MASTER_PATH
    if master_path.exists():
        with open(master_path) as f:
            master_data = json.load(f)
        # Two lookups: csv_name -> meta, and auction_name -> meta
        # Only consider players marked as in_auction (this year's auction).
        auction_master = [p for p in master_data if p.get("in_auction", True)]
        csv_to_meta = {p["csv_name"]: p for p in auction_master if p.get("csv_name")}
        name_to_meta = {p["name"]: p for p in auction_master}

        filtered_pool = []
        removed_players = []
        for player in pool:
            meta = csv_to_meta.get(player["player_name"]) or name_to_meta.get(player["player_name"])
            if meta is None:
                removed_players.append({"player_name": player["player_name"], "role": player["role"],
                                        "projected_points": player["projected_points"]})
            else:
                player["ipl_team"] = meta["ipl_team"]
                player["auction_set"] = meta["auction_set"]
                # Apply role from master — it's manually curated and more reliable
                # than the stats-based classifier for borderline/new players.
                player["role"] = meta["role"]
                filtered_pool.append(player)
        pool = filtered_pool

        # Save removed players for debugging
        _removed_path = Path(__file__).parent.parent / "removed_from_pool.json"
        with open(_removed_path, "w") as f:
            json.dump(removed_players, f, indent=2)
        if removed_players:
            import warnings
            warnings.warn(
                f"{len(removed_players)} players removed (not in player_master.json). "
                f"See removed_from_pool.json",
                stacklevel=2,
            )

        # Add floor-value players from master that have no CSV data.
        # Covers two cases:
        #   1. csv_name is None  — player has never appeared in the stat CSV
        #   2. csv_name is set but not in pool — player exists in CSV historically
        #      but played no matches in the active projection seasons (injured,
        #      overseas cap, etc.) and was filtered out by min_matches.
        stat_csv_names = {p["player_name"] for p in pool}
        for pm in auction_master:
            csv = pm.get("csv_name")
            in_pool = (csv and csv in stat_csv_names) or (pm["name"] in stat_csv_names)
            if not in_pool:
                role = pm["role"]  # already normalized (BWL→BOWL in player_master)
                pts = _FLOOR_PTS.get(role, 38.0)
                pool.append({
                    "player_name": pm["name"],
                    "role": role,
                    "projected_points": pts,
                    "std_dev": 15.0,
                    "vorp": 0.0,
                    "tier": 3,
                    "base_price": 1.0,
                    "is_marquee": False,
                    "matches": 0,
                    "ipl_team": pm["ipl_team"],
                    "auction_set": pm["auction_set"],
                })
    else:
        for player in pool:
            player.setdefault("ipl_team", "UNK")
            player.setdefault("auction_set", 12)

    # 11. Apply manual overrides
    pool = apply_overrides(pool, overrides_path)

    return pool


# ── IPL-realistic auction ordering ───────────────────────────────────────────
# Real IPL auction structure:
#   1. Role SETS processed in order: BAT → AR → WK → BOWL
#   2. Within each role: capped players (T1+T2) before uncapped (T3)
#   3. Within each capped/uncapped group: players divided into base-price
#      brackets of ~BRACKET_SIZE players (highest projected_points first).
#   4. Within each bracket: randomised — teams know the approximate price
#      band of the next player but not the exact pick order.
#   Round 2+ (unsold re-entry) is an accelerated auction with no set order.

MARQUEE_SIZE = 10  # kept for backward compatibility


def order_pool(
    pool: list[dict],
    *,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Order the pool by auction_set (1–12), randomised within each set.

    auction_set mirrors the real IPL auction structure where players are
    grouped into sets by role and tier, and the auctioneer draws cards
    randomly within each set.  Set numbers come from player_master.json.
    """
    if rng is None:
        rng = random.Random()

    from collections import defaultdict
    by_set: dict[int, list[dict]] = defaultdict(list)
    for p in pool:
        by_set[p.get("auction_set", 12)].append(p)

    result: list[dict] = []
    for aset in sorted(by_set.keys()):
        group = by_set[aset]
        rng.shuffle(group)
        result.extend(group)
    return result
