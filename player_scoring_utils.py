import polars as pl
import json


# Map the JSON keys to our DataFrame column names
STAT_COL_MAP = {
    "RUNS": "runs_scored",
    "FOURS": "four_count",
    "SIXES": "six_count",
    "BALLS_FACED": "balls_faced",
    "STRIKE_RATE": "strike_rate",
    "WICKETS": "wickets_taken",
    "BOWLED_LBW": "bowled_lbw_wickets",
    "MAIDENS": "maiden_count",
    "ECONOMY": "economy",
    "BALLS_BOWLED": "balls_bowled",
    "CATCHES": "catches_caught",
    "RUNOUTS": "runouts",
}


def load_data_and_rules(csv_path: str, json_path: str):
    """Loads the player stats CSV into a Polars DataFrame and reads the JSON rules."""
    df = pl.read_csv(csv_path, infer_schema_length=1000)
    with open(json_path, "r") as f:
        rules = json.load(f)
    return df, rules


def calculate_match_points(df: pl.DataFrame, rules: list) -> pl.DataFrame:
    """Calculates fantasy points for every match using vectorized Polars expressions."""

    # 1. Add derived stats safely (handling division by zero natively)
    df = df.with_columns(
        [
            pl.when(pl.col("balls_faced") > 0)
            .then((pl.col("runs_scored") / pl.col("balls_faced")) * 100)
            .otherwise(0.0)
            .alias("strike_rate"),
            (pl.col("balls_bowled") / 6.0).alias("overs_bowled"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("overs_bowled") > 0)
            .then(pl.col("runs_given") / pl.col("overs_bowled"))
            .otherwise(0.0)
            .alias("economy")
        ]
    )

    # 2. Compile JSON rules into a list of Polars expressions
    point_exprs = []

    for rule in rules:
        stat_key = rule["stat"]
        col_name = STAT_COL_MAP.get(stat_key)
        if not col_name:
            continue

        # Start with a base condition of True
        base_cond = pl.lit(True)

        # Apply secondary conditions (e.g., Min Balls Faced)
        if "condition" in rule:
            cond = rule["condition"]
            cond_col = STAT_COL_MAP.get(cond["stat"])
            if "min" in cond and cond["min"] is not None:
                base_cond = base_cond & (pl.col(cond_col) >= cond["min"])
            if "max" in cond and cond["max"] is not None:
                base_cond = base_cond & (pl.col(cond_col) < cond["max"])

        rule_type = rule["type"]
        rule_pts = rule["points"]

        # Build the specific mathematical expression for this rule
        if rule_type == "PER_UNIT":
            expr = pl.when(base_cond).then(pl.col(col_name) * rule_pts).otherwise(0)
            point_exprs.append(expr)

        elif rule_type == "RANGE":
            min_val = rule.get("min")
            max_val = rule.get("max")

            range_cond = base_cond
            if min_val is not None:
                range_cond = range_cond & (pl.col(col_name) >= min_val)
            if max_val is not None:
                range_cond = range_cond & (pl.col(col_name) < max_val)

            expr = pl.when(range_cond).then(rule_pts).otherwise(0)
            point_exprs.append(expr)

        elif rule_type == "MILESTONE":
            threshold = rule.get("threshold")
            if rule.get("exact"):
                ms_cond = base_cond & (pl.col(col_name) == threshold)

                # Duck safety check (-10 points only if they actually faced a ball or got out)
                if stat_key == "RUNS" and threshold == 0:
                    duck_safe = (pl.col("balls_faced") > 0) | (pl.col("is_out") == 1)
                    ms_cond = ms_cond & duck_safe

                expr = pl.when(ms_cond).then(rule_pts).otherwise(0)
                point_exprs.append(expr)

    # 3. Execute all expressions simultaneously and sum them horizontally
    df = df.with_columns(pl.sum_horizontal(point_exprs).alias("match_fantasy_points"))

    return df


def aggregate_year_wise_leaderboard(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregates points by year and player, calculating Points Per Match (PPM)."""
    return (
        df.group_by(["ipl_year", "player_id", "player_name"])
        .agg(
            [
                pl.len().alias("matches_played"),
                pl.col("runs_scored").sum().alias("total_runs"),
                pl.col("wickets_taken").sum().alias("total_wickets"),
                pl.col("balls_bowled")
                .sum()
                .alias("balls_bowled"),  # <--- Added this line!
                pl.col("match_fantasy_points").sum().alias("total_points"),
                pl.col("match_fantasy_points").std().alias("points_stddev"),
            ]
        )
        .with_columns(
            (pl.col("total_points") / pl.col("matches_played"))
            .round(2)
            .alias("points_per_match"),
        )
        .sort(["ipl_year", "total_points"], descending=[False, True])
    )


def aggregate_all_time_leaderboard(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregates all-time points and calculates career Points Per Match."""
    return (
        df.group_by(["player_id", "player_name"])
        .agg(
            [
                pl.len().alias("total_matches"),
                pl.col("match_fantasy_points").sum().alias("all_time_points"),
                pl.col("match_fantasy_points").std().alias("points_stddev"),
            ]
        )
        .with_columns(
            (pl.col("all_time_points") / pl.col("total_matches"))
            .round(2)
            .alias("all_time_ppm")
        )
        .sort("all_time_points", descending=True)
    )
