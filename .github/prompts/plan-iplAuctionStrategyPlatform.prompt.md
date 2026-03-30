# Plan: IPL Auction Strategy Platform

## Context
- 6-team league
- Python: Polars (drop Pandas)
- Evo algo: GA + CMA-ES both
- HTML data flow: local FastAPI HTTP server

---

## Codebase Summary

| File | What it does |
|---|---|
| `scoring_rules.json` | Fantasy scoring rules in clean JSON (PER_UNIT/RANGE/MILESTONE) — already great |
| `player_scoring_utils.py` | Polars-based scorer that reads the JSON rules + aggregation — solid foundation |
| `baseline.ipynb` | Pool generation + 4 strategy classes + auction engine (Polars) |
| `full_simulation_claude.py` | 9 strategies + Monte Carlo (Pandas, but **hardcodes** its own `_calc_fp()` ignoring the JSON rules) |
| `ipl_live_advisor.html` | Beautiful dark-mode advisor with **static 2024 player array hardcoded** in JS, 6 strategy simulations |
| `main.py` | Empty placeholder |

**Critical gaps:**
1. Three disconnected implementations — Polars notebook, Pandas simulation, JavaScript HTML each re-implement scoring and strategies independently
2. Scoring is duplicated and fragile — `_calc_fp()` in `full_simulation_claude.py` re-implements the same rules in row-wise Pandas instead of using the vectorised Polars pipe in `player_scoring_utils.py`. The numeric values happen to match `scoring_rules.json` today, but divergence is inevitable.
3. Strategy params are buried as class constants — `ELITE_CAP = 0.35`, `T1_CAP = 0.45`, `WC = 0.90`, `TARGET = {BAT:5, BOWL:4, AR:2, WK:2}` etc. — no way to tune or evolve them without editing source
4. VORP uses 25th-percentile replacement level per role (quantile(0.25) over avg_pts) — the baseline notebook switched to a depth-index approach; both need consolidating
5. No evolutionary learning, no persistence, HTML pool is static 2024 data, no manual overrides
6. The `season_score()` function scores **top-11 projected_points** (not all 15) — this must be the unified fitness metric everywhere

---

## Reference Constants (canonical source: `full_simulation_claude.py`)

These values are the ground truth for the engine. Only change them intentionally via config.

```python
# Pool generation
POOL_SIZE       = 80          # top N players by avg_pts
MIN_MATCHES     = 5           # minimum matches to qualify
TIER1_FLOOR     = 80          # avg pts/match → Elite (T1)
TIER2_FLOOR     = 58          # avg pts/match → Solid (T2), below = Depth (T3)

# Auction engine
BID_INCREMENT   = 0.25        # ₹Cr per bid step
NOISE_RANGE     = (-0.25, 0.5)  # uniform noise added to each WTP calculation
DEFAULT_PURSE   = 120.0      # ₹Cr per manager
MIN_ROSTER      = 13
MAX_ROSTER      = 15

# Scoring season (what load_player_pool() filters on by default)
SEASON          = "2024"      # ← replaced by ProjectionModel in Phase 1

# VORP replacement level: quantile(0.25) of avg_pts per role over the filtered season
# (baseline.ipynb uses num_teams × role_requirement depth index instead — Phase 1 unifies these)
```

**Role classification rules (from `_classify_role`):**
```
catches ≥ 8  AND runs < 150 AND wkts == 0         → WK
runs ≥ 250   AND wkts ≥ 8                         → AR
balls_bowled ≥ 120 AND runs < 200                 → BOWL
balls_bowled ≥ 60  AND wkts ≥ 6 AND runs ≥ 100   → AR
everything else                                    → BAT
```
*Note: baseline.ipynb uses a simpler heuristic (balls_bowled == 0 → BAT). Adopt the full_simulation version above as it handles all-rounders correctly.*

**Auction state dict contract** (what `build_state()` produces — every strategy receives this):
```python
state = {
    "players_remaining": int,
    "cr_per_vorp":       float,   # available_cash / remaining_vorp
    "pool_by_role":      {"BAT": [...], "BOWL": [...], "AR": [...], "WK": [...]},  # sorted by projected_points desc
    "agent_states":      {name: {"purse": float, "slots": int, "mandatory": int, "name": str}}
}
```

**Scoring — `_calc_fp()` vs `scoring_rules.json`:** Values are numerically identical today. The risk is future drift. Phase 1 deletes `_calc_fp()` and routes everything through `player_scoring_utils.calculate_match_points()` (Polars, JSON-driven).

---

## Phase 1 — Unified Engine Package
*Foundation. Everything else depends on this.*

1. Create `engine/` package with `__init__.py`
2. Move and clean `player_scoring_utils.py` → `engine/scoring.py` (Polars, JSON-rules-driven, becomes the only scorer in the entire codebase)
3. Create `engine/pool.py` — implements a `ProjectionModel` ABC with three concrete models:
   - `WeightedSeasonModel(season_weights: dict)` — e.g. `{"2024": 1.0, "2023": 0.6, "2022": 0.3}`. Aggregation: weighted `avg_pts` per player per season, then weighted average across seasons. Mirrors `load_player_pool()` in `full_simulation_claude.py` (which does a single-season `avg_pts` per match) but generalises it.
   - `RecentFormModel(last_n_matches: int)` — filter the match log to the most recent N matches per player before aggregating avg_pts. Good for players with momentum (e.g. late-season form).
   - `CustomFunctionModel(fn: callable)` — user passes a function `fn(df: pl.DataFrame) -> pl.DataFrame` that adds/overwrites a `projected_points` column. Full escape hatch.
   
   Pool generation steps (from `load_player_pool()` / `generate_auction_pool()`):
   1. Score matches via `engine/scoring.py` (JSON rules, Polars)
   2. Filter by `MIN_MATCHES = 5` (configurable)
   3. Aggregate: `avg_pts`, `std_pts` (fill null → 30), `total_runs`, `total_wkts`, `total_catches`, `total_balls_bowled`
   4. Classify roles using `_classify_role()` rules (see Reference Constants above)
   5. Calculate VORP: replacement level = `quantile(0.25)` of `avg_pts` per role (from `full_simulation_claude.py`). Alternative: depth-index (`num_teams × role_req`-th player) from `baseline.ipynb` — expose both as `vorp_method` parameter.
   6. Assign tiers: T1 if `avg_pts > 80`, T2 if `> 58`, else T3
   7. Take top `POOL_SIZE = 80` by `avg_pts`
   8. Apply overrides from `overrides.json`
   9. Output: list of dicts `{player_name, role, projected_points, std_dev, vorp, tier, matches}`
4. Create `engine/overrides.py` — loads `overrides.json` (new file) and applies manual point adjustments on top of any projection model. Schema: `{"Virat Kohli": {"projected_points": 130, "note": "gut feeling, IPL form superb"}}`

---

## Phase 2 — Parameterized Strategy Registry
*Parallel with Phase 1.*

5. Create `engine/strategies.py` — port all 9 strategies from `full_simulation_claude.py`. Each gets a companion `@dataclass XParams`. Exact default params:

   | Strategy | Params (defaults from full_simulation_claude.py) |
   |---|---|
   | `StarChaser` | `elite_cap=0.35, t1_solo_mult=2.4, t1_extra_mult=1.2, t2_mult=1.0, depth_mult=0.55` |
   | `ValueInvestor` | `urgency_min=0.85, urgency_max=1.35, fair_base=0.5` |
   | `DynamicMaximizer` | `base_exp=1.4, exp_per_agent=0.07, exp_cap=2.0` (exponent = min(exp_cap, base_exp + (n_agents−4)×exp_per_agent)) |
   | `ApexPredator` | `wc_discount=0.90, efficiency_exp=1.5, adj_min=0.8, adj_max=2.0` (winner's-curse 10% discount, Milgrom/Thaler) |
   | `BarbellStrategist` | `hi_cost_threshold=0.15, lo_cost_threshold=0.05, hi_penalty_mult=1.0, lo_reward_mult=1.0, mid_penalty_mult=0.25, base_exp=1.6` |
   | `VampireMaximizer` | `shade_margin=0.5, bluff_floor_ratio=0.40, bluff_cap_ratio=0.60, purse_healthy_pct=0.40, min_slots_to_bluff=4, elite_pts_threshold=90, elite_mult=1.4, normal_mult=0.9` |
   | `TierSniper` | `t1_cap=0.45, t1_solo_mult=2.8, t1_price_enforce_mult=1.05, t1_sit_back_mult=0.5, t1_enforce_when_remaining=3, t2_mult=1.1, depth_mult=0.65` |
   | `NominationGambler` | `phase_a_threshold=8, phase_b_threshold=3, phase_a_t1_mult=1.5, phase_a_other_mult=0.25, phase_a_exp=None, phase_b_t1_exp=1.5, phase_b_t2_mult=0.9, phase_b_depth_mult=0.40, phase_c_exp=2.0` |
   | `PositionalArbitrageur` | `target={BAT:5,BOWL:4,AR:2,WK:2}, scarcity_cap=2.5, filled_discount=0.7, scarcity_exp=0.8, premium_exp=1.4` |

   Shared base helpers (all strategies inherit from `Manager`):
   - `_desperation(wtp, state, rnd)`: scarcity > 0.15 → `wtp *= 1 + (scarcity*3)²`; rnd > 1 and mandatory > 0 → floor at `purse/slots`
   - `_alt_mean(player, state)`: mean projected_points of top-N alternatives in same role (N = max(1, 4 − role_count))
   - `_premium_wtp(player, state, rnd, exponent=1.6)`: `cash_per_slot × (pts/alt_mean)^exponent`, then desperation-clamped

6. Create `strategy_params.json` — stores default params for all 9 strategies. Also the target file for evolved params. Strategies read from this at runtime.
7. Create `STRATEGY_REGISTRY` dict mapping name → `(StrategyClass, DefaultParams)`. Adding a new strategy = add one class + one dataclass and register it. Nothing else changes.
8. Create `engine/auction.py` — clean ascending-bid auction engine (Polars, replaces both baseline and full_simulation engines). Key behaviours to preserve from `full_simulation_claude.py`:
   - State built via `build_state()` before each lot (exact contract above)
   - Noise: `uniform(-0.25, +0.5)` added to each WTP (only if raw WTP > 0.5)
   - Bid loop: randomised agent order each pass; increment = ₹0.25; loop restarts on each new bid
   - Fire-sale: triggered after each sale; returns player to re-entry pool at 75% price recovery
   - Unsold pool re-entered in subsequent rounds; stops when zero players bought in a round

---

## Phase 3 — Monte Carlo + Evolutionary Optimizer
*Depends on Phases 1 & 2.*

9. Create `engine/simulation.py` — Monte Carlo runner: takes strategy list + player pool, runs N auctions, returns `{strategy_name: {win_rate, avg_score, std_score, wins}}`. `season_score()` = sum of top-11 `projected_points` — same metric as `full_simulation_claude.py`.
10. Implement **Genetic Algorithm** in `engine/simulation.py`:
    - Chromosome = flattened param vector for one strategy (pulled from its `XParams` dataclass)
    - Fitness = `win_rate * 0.7 + (avg_score / theoretical_max) * 0.3` where `theoretical_max = sum of top-11 projected_points in pool`
    - Operators: tournament selection (k=3), blend crossover BLX-α (α=0.5), Gaussian mutation σ=0.05×param_range
    - Param bounds: each field in `XParams` carries `(lo, hi)` metadata — e.g. `elite_cap ∈ [0.20, 0.60]`, `t1_solo_mult ∈ [1.5, 4.0]`
    - Evolves per-strategy against a fixed opponent field (the other 8 strategies at their current best params)
    - Population size: 40; elitism: top-2 survive unchanged
11. Implement **CMA-ES** alongside — add `cma` to `pyproject.toml`. Same fitness function and bounds. Compare: GA tends to explore wider; CMA-ES converges faster near optima. Run both, keep the winner per strategy.
12. Evolution results save to `evolved_params.json` with fitness history per generation (for plotting)
13. CLI command: `python main.py evolve --algo ga --generations 50 --mc-runs 200 --strategy TierSniper`

---

## Phase 4 — Export + Local HTTP Server
*Depends on Phase 3. Enables Phase 5.*

14. Create `engine/export.py` — generates `player_pool.json` containing the full auction pool with tier assignments, VORP, std_dev, and per-strategy Monte Carlo insights
15. Create `server.py` — minimal **FastAPI** server (add `fastapi` + `uvicorn` to `pyproject.toml`) with endpoints:
    - `GET /api/pool` — current player pool (respects overrides)
    - `GET /api/strategies` — registry + current params + MC win rates
    - `POST /api/override` — update a player's `projected_points` + save to `overrides.json`
    - `GET /api/playbook?strategy=TierSniper` — pre-computed tier-by-tier bid ranges
    - `POST /api/simulate` — trigger a fresh MC run (async, streamed)
16. Wire `main.py` as the unified CLI: `build-pool`, `simulate`, `evolve`, `serve`

---

## Phase 5 — HTML Advisor Enhancement
*Depends on Phase 4. Can be done incrementally.*

17. Replace the hardcoded `const POOL = [...]` array with a `fetch('/api/pool')` on page load
18. Add **"I am playing as:"** strategy dropdown — switches the recommendation engine to use that strategy's evolved params from `/api/strategies`
19. Add **player value override** inline edit in the auction panel — clicking projected_points opens an editable field, submits `POST /api/override`, and recomputes live
20. Add **"Simulation Insights"** panel — horizontal bar chart of MC win rates per strategy (rendered with Canvas/SVG, no external libs)
21. Add **"Playbook"** tab — pre-fetched tier-by-tier advice for your selected strategy (e.g., "Phase A: lock one T1 at up to ₹54Cr; don't touch T2 above ₹X")
22. **Opponent strategy labeling** — label each opponent as a suspected strategy to get more accurate threat estimates from the server

---

## Resulting File Structure

```
auction-strategies/
  engine/
    __init__.py
    scoring.py        ← single source of truth for fantasy points
    pool.py           ← ProjectionModel ABC + 3 implementations
    overrides.py      ← manual player value adjustments
    strategies.py     ← 9 strategies + @dataclass params + REGISTRY
    auction.py        ← clean ascending-bid engine
    simulation.py     ← Monte Carlo + GA + CMA-ES
    export.py         ← generates pool/insights JSON
  main.py             ← unified CLI
  server.py           ← FastAPI HTTP server
  ipl_live_advisor.html  ← updated to fetch from server
  scoring_rules.json
  overrides.json      ← NEW (manual value overrides)
  strategy_params.json   ← NEW (default + evolved params)
  player_pool.json    ← generated output
  evolved_params.json ← generated output
  ipl_player_stats.csv
  pyproject.toml      ← add: cma, fastapi, uvicorn, httpx
  baseline.ipynb      ← keep for exploration
```

---

## Verification Checklist

1. `python main.py build-pool --model weighted --seasons 2022:0.3,2023:0.6,2024:1.0` → `player_pool.json` with correct VORP
2. `python main.py simulate --mc 500` → prints win-rate table matching rough expectations
3. `python main.py evolve --algo ga --strategy TierSniper` → `evolved_params.json` updated, fitness curve plotted
4. `python main.py serve` → HTML opens, loads players dynamically, override a player → recommendation updates instantly
5. Add a new dummy strategy with 5 params → `REGISTRY` entry → appears in HTML dropdown with MC stats, no other changes needed

---

## Additional Ideas

**Nomination strategy simulator** — In real IPL auctions, you control which player goes up for bid. A "nomination picker" module that chooses *who* to nominate based on current pool state (put up a player your rival desperately needs to drain their purse) would be a strong edge.

**Opponent modeling from past auctions** — Log prior real-world auction outcomes and fit each friend's behavior to the nearest strategy. The threat assessment in the HTML would then use their historical patterns, not just their current purse/slots.

**Match simulation for post-auction score projection** — Monte Carlo over actual season score distributions (you have `std_dev` already) to show "you have a 73% chance of winning the season" based on your current squad — a great morale/decision tool during a live auction.
