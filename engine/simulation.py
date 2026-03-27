"""
engine/simulation.py
=====================
Monte Carlo runner + evolutionary optimizers (GA and CMA-ES).

Public API:
  run_monte_carlo(pool, strategy_names, n, *, evolved, params_path)
      → dict[str, MCResult]

  evolve_strategy(pool, target_strategy, opponent_strategies, *, algo,
                  generations, mc_runs, population, params_path)
      → EvoResult

Fitness = win_rate * 0.7 + (avg_score / theoretical_max) * 0.3
theoretical_max = sum of top-11 projected_points in pool
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .strategies import (
    STRATEGY_REGISTRY,
    Manager,
    StrategyParams,
    instantiate,
    load_params,
    save_params,
)
from .auction import run_auction

# ── Result types ─────────────────────────────────────────────────────────────
@dataclass
@dataclass
@dataclass
class MCResult:
    strategy: str
    wins: int
    total_auctions: int
    participations: int
    avg_score: float
    std_score: float
    avg_purse_remaining: float = 0.0

    @property
    def win_share(self) -> float:
        """Fraction of ALL N auctions this strategy won.
        sum(win_share) == 1.0 exactly across all strategies."""
        return self.wins / max(1, self.total_auctions)

    @property
    def win_rate(self) -> float:
        """Alias for win_share."""
        return self.win_share

    @property
    def conditional_win_rate(self) -> float:
        """wins / participations: per-game skill metric."""
        return self.wins / max(1, self.participations)

    @property
    def budget_utilization(self) -> float:
        """Fraction of purse spent."""
        return max(0.0, 1.0 - self.avg_purse_remaining / 120.0)

@dataclass
class EvoResult:
    strategy: str
    algo: str
    best_params: StrategyParams
    best_fitness: float
    fitness_history: list[float]    # per generation
    duration_s: float


# ── Monte Carlo ───────────────────────────────────────────────────────────────
def run_monte_carlo(
    pool: list[dict],
    strategy_names: list[str] | None = None,
    n: int = 500,
    *,
    field_size: int = 6,
    evolved: bool = False,
    params_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, MCResult]:
    """
    Run exactly N independent auctions, each with field_size randomly
    sampled strategies (without replacement per auction).

    Two complementary metrics result:
      win_share          = wins / N  (sums to 100% — one winner per auction)
      conditional_win_rate = wins / participations  (per-game skill, fair
                             because every draw is independent)

    This avoids the shuffle-and-batch problem where the last batch has
    fewer than field_size participants and the denominator inflates with
    auctions a strategy was never part of.
    """
    names = strategy_names or list(STRATEGY_REGISTRY.keys())
    field_size = min(field_size, len(names))

    wins: dict[str, int] = {nm: 0 for nm in names}
    participations: dict[str, int] = {nm: 0 for nm in names}
    scores: dict[str, list[float]] = {nm: [] for nm in names}
    purses: dict[str, list[float]] = {nm: [] for nm in names}

    if verbose:
        print(
            f"\nMonte Carlo: {n} independent auctions × {field_size} managers each"
            f" ({len(names)} strategies in pool) …",
            end="", flush=True,
        )

    for i in range(n):
        # Draw exactly field_size distinct strategies at random per auction
        batch = random.sample(names, field_size)
        agents = [
            instantiate(nm, evolved=evolved, params_path=params_path)
            for nm in batch
        ]
        run_auction([p.copy() for p in pool], agents, verbose=False)
        sim = {a.name: a.season_score() for a in agents}
        winner = max(sim, key=sim.__getitem__)
        for a in agents:
            nm = a.name
            participations[nm] += 1
            scores[nm].append(sim[nm])
            purses[nm].append(a.purse)
        wins[winner] += 1   # exactly one winner per auction

        if verbose and (i + 1) % 100 == 0:
            print(f" {i+1}…", end="", flush=True)

    if verbose:
        print(" done!\n")

    results: dict[str, MCResult] = {}
    for nm in names:
        sc = scores[nm]
        avg = sum(sc) / len(sc) if sc else 0.0
        std = math.sqrt(sum((s - avg) ** 2 for s in sc) / len(sc)) if sc else 0.0
        avg_purse = sum(purses[nm]) / len(purses[nm]) if purses[nm] else 0.0
        results[nm] = MCResult(
            strategy=nm,
            wins=wins[nm],
            total_auctions=n,
            participations=participations[nm],
            avg_score=avg,
            std_score=std,
            avg_purse_remaining=avg_purse,
        )

    return results


def print_mc_results(results: dict[str, MCResult]) -> None:
    sample = next(iter(results.values()))
    n = sample.total_auctions
    field_size = round(sum(r.participations for r in results.values()) / n)
    sep = "=" * 100
    print(sep)
    print(f"  MONTE CARLO  {n} auctions x {field_size} managers  | {len(results)} strategies")
    print(f"  WinShare   = wins / {n} total auctions -> sums to 100%")
    print(f"  WinRate/gm = wins / games-played  (equal baseline = {100/field_size:.1f}%)")
    print(sep)
    print(f"  {'Strategy':<28} {'WinShare':>8}  {'WinRate/gm':>10}  {'Games':>6}  {'AvgPts':>8}  {'BudgetUse':>9}  Chart")
    for r in sorted(results.values(), key=lambda r: r.win_share, reverse=True):
        bar = "#" * max(1, int(r.win_share * 100 / 1.5))
        print(
            f"  {r.strategy:<28} {r.win_share*100:>7.2f}%  "
            f"{r.conditional_win_rate*100:>9.2f}%  {r.participations:>6}  "
            f"{r.avg_score:>8.1f}  {r.budget_utilization*100:>8.1f}%  {bar}"
        )
    total_share = sum(r.win_share for r in results.values()) * 100
    print(sep)
    print(f"  {'TOTAL':<28} {total_share:>7.2f}%")
    best = max(results.values(), key=lambda r: r.win_share)
    worst = min(results.values(), key=lambda r: r.budget_utilization)
    print(f"\n  DOMINANT: {best.strategy} ({best.win_share*100:.2f}% win share, {best.conditional_win_rate*100:.2f}% per game)")
    print(f"  MOST WASTEFUL: {worst.strategy} ({(1-worst.budget_utilization)*100:.1f}% purse unspent)\n")


# ── Strategic insights: player pick tracking ──────────────────────────────────
def analyze_strategy_picks(
    pool: list[dict],
    strategy_names: list[str] | None = None,
    n: int = 150,
    *,
    field_size: int = 6,
    evolved: bool = False,
    params_path: str | Path | None = None,
) -> dict[str, dict]:
    """
    Run n independent auctions tracking which players each strategy buys.

    Returns a dict keyed by strategy name, each value containing:
      total_games  : int    — auctions the strategy participated in
      picks        : Counter — player_name → number of times bought
      prices       : dict   — player_name → list[float] of prices paid
      pts          : dict   — player_name → projected_points
    """
    from collections import Counter
    from .strategies import STRATEGY_REGISTRY, load_params, instantiate
    from .auction import run_auction

    names = strategy_names or list(STRATEGY_REGISTRY.keys())
    stats: dict[str, dict] = {
        name: {"auctions_in": 0, "picks": Counter(), "prices": {}, "pts": {}}
        for name in names
    }
    stats["_meta"] = {"n_auctions": n}

    for _ in range(n):
        field = random.sample(names, min(field_size, len(names)))
        agents = [
            STRATEGY_REGISTRY[nm][0](
                name=nm,
                params=load_params(nm, params_path, evolved=evolved),
            )
            for nm in field
        ]
        log: list[dict] = []
        run_auction([p.copy() for p in pool], agents, verbose=False, purchase_log=log)
        for nm in field:
            stats[nm]["auctions_in"] += 1
        for entry in log:
            s = stats[entry["strategy"]]
            s["picks"][entry["player"]] += 1
            s["prices"].setdefault(entry["player"], []).append(entry["price"])
            s["pts"][entry["player"]] = entry["projected_points"]

    return stats


def print_strategy_insights(
    pick_stats: dict[str, dict],
    mc_results: dict[str, "MCResult"],
    pool: list[dict],
    *,
    top_strategies: int = 5,
    top_players: int = 8,
) -> None:
    """
    Print three insight panels:
      1. For each top strategy — most-targeted players, avg price vs base price
      2. Value buys — high-quality players consistently bought below their VORP-implied price
      3. Overlooked gems — high projected_points but low pick rate across all strategies
    """
    pool_map = {p["player_name"]: p for p in pool}
    n_auctions: int = pick_stats.get("_meta", {}).get("n_auctions", 150)  # type: ignore[arg-type]

    # Panel 1: Per-strategy player preferences
    sorted_strats = sorted(
        [s for s in pick_stats if s != "_meta" and s in mc_results],
        key=lambda s: mc_results[s].win_share,
        reverse=True,
    )[:top_strategies]

    print("\n" + "=" * 90)
    print("  STRATEGIC INSIGHTS — How top strategies win")
    print("=" * 90)

    for name in sorted_strats:
        st = pick_stats[name]
        mc = mc_results[name]
        auctions_in = max(1, st["auctions_in"])
        if auctions_in == 0:
            continue
        print(f"\n  \u25b6 {name}  (WinShare={mc.win_share*100:.1f}%  WinRate/gm={mc.conditional_win_rate*100:.1f}%)")
        print(f"    {'Player':<26} {'Role':<5} {'T':>2} {'Pick%':>6} {'AvgPrice':>9} {'Base':>6} {'Premium':>8}")
        print("    " + "\u2500" * 67)
        for pname, cnt in st["picks"].most_common(top_players):
            pct = 100 * cnt / auctions_in
            prices = st["prices"].get(pname, [])
            avg_p = sum(prices) / len(prices) if prices else 0.0
            p = pool_map.get(pname, {})
            base = p.get("base_price", 1.0)
            premium = 100 * (avg_p - base) / max(0.01, base)
            star = " \u2605" if p.get("is_marquee") else ""
            print(
                f"    {pname + star:<26} {p.get('role','?'):<5} {p.get('tier','?'):>2}"
                f" {pct:>5.1f}%  \u20b9{avg_p:>6.2f}Cr  \u20b9{base:>4.0f}Cr  {premium:>+7.0f}%"
            )

    # Global pick rates: denominator = n_auctions (player can be bought once per auction)
    all_picks: dict[str, int] = {}
    all_prices: dict[str, list[float]] = {}
    for nm, st in pick_stats.items():
        if nm == "_meta":
            continue
        for pname, cnt in st["picks"].items():
            all_picks[pname] = all_picks.get(pname, 0) + cnt
            all_prices.setdefault(pname, []).extend(st["prices"].get(pname, []))
    # Cap per-player pick count at n_auctions (bought by exactly one team per auction)
    all_picks = {k: min(v, n_auctions) for k, v in all_picks.items()}

    # Panel 2: Value buys (T1/T2 players bought at low premium)
    print("\n\n  \U0001f48e VALUE BUYS \u2014 quality players bought close to base price")
    print(f"  {'Player':<26} {'Role':<5} {'T':>2} {'Pts':>6} {'BoughtIn%':>10} {'AvgPrice':>9} {'Base':>6}")
    print("  " + "\u2500" * 69)
    value_candidates = [
        (pname, p) for pname, p in pool_map.items()
        if p.get("tier", 3) <= 2
        and all_picks.get(pname, 0) > 0
    ]
    value_candidates.sort(
        key=lambda x: (
            sum(all_prices.get(x[0], [x[1].get("base_price", 1)])) /
            max(1, len(all_prices.get(x[0], [1])))
        ) / max(0.01, x[1].get("base_price", 1))
    )
    for pname, p in value_candidates[:8]:
        prices = all_prices.get(pname, [p.get("base_price", 1.0)])
        avg_p = sum(prices) / len(prices)
        bought_pct = 100 * all_picks.get(pname, 0) / n_auctions
        star = " \u2605" if p.get("is_marquee") else ""
        print(
            f"  {pname + star:<26} {p.get('role','?'):<5} {p.get('tier','?'):>2}"
            f" {p['projected_points']:>6.1f} {bought_pct:>9.1f}%  \u20b9{avg_p:>6.2f}Cr  \u20b9{p.get('base_price',1):>4.0f}Cr"
        )

    # Panel 3: Overlooked gems (high pts, low BoughtIn%)
    print("\n\n  \U0001f50d OVERLOOKED GEMS \u2014 high quality, low competition")
    print(f"  {'Player':<26} {'Role':<5} {'T':>2} {'Pts':>6} {'BoughtIn%':>10} {'AvgPrice':>9}")
    print("  " + "\u2500" * 62)
    overlooked = [
        (pname, p) for pname, p in pool_map.items()
        if p.get("tier", 3) <= 2
    ]
    overlooked.sort(
        key=lambda x: (
            all_picks.get(x[0], 0) / n_auctions - x[1]["projected_points"] / 200
        )
    )
    for pname, p in overlooked[:8]:
        prices = all_prices.get(pname, [])
        avg_p = sum(prices) / len(prices) if prices else p.get("base_price", 1.0)
        bought_pct = 100 * all_picks.get(pname, 0) / n_auctions
        star = " \u2605" if p.get("is_marquee") else ""
        print(
            f"  {pname + star:<26} {p.get('role','?'):<5} {p.get('tier','?'):>2}"
            f" {p['projected_points']:>6.1f} {bought_pct:>9.1f}%  \u20b9{avg_p:>6.2f}Cr"
        )
    print()

def _theoretical_max(pool: list[dict]) -> float:
    """Sum of top-11 projected_points — same as season_score() ceiling."""
    return sum(sorted((p["projected_points"] for p in pool), reverse=True)[:11])


def _evaluate_fitness(
    pool: list[dict],
    target_name: str,
    params: StrategyParams,
    opponent_names: list[str],
    mc_runs: int,
    evolved: bool,
    params_path: Path | None,
) -> float:
    """
    Run mc_runs auctions and compute composite fitness:
      0.55 × win_rate  +  0.25 × (avg_score/theoretical_max)  +  0.20 × budget_utilization

    The budget_utilization term directly penalises leftover purse, closing
    the gap between strategies that win games and those that spend efficiently.
    """
    TargetCls, _ = STRATEGY_REGISTRY[target_name]
    all_opp_agents = [
        instantiate(nm, evolved=evolved, params_path=params_path)
        for nm in opponent_names
    ]
    # Match real game: target + 5 opponents = 6 managers per auction
    _FIELD = 6
    n_opps = min(_FIELD - 1, len(all_opp_agents))

    wins = 0
    scores: list[float] = []
    purses: list[float] = []
    for _ in range(mc_runs):
        sampled_opps = random.sample(all_opp_agents, n_opps)
        target_agent = TargetCls(name=target_name, params=copy.deepcopy(params))
        agents = [target_agent] + [copy.deepcopy(a) for a in sampled_opps]
        run_auction([p.copy() for p in pool], agents, verbose=False)
        sim = {a.name: a.season_score() for a in agents}
        top = max(sim.values())
        sc = sim[target_name]
        scores.append(sc)
        purses.append(target_agent.purse)
        if sc >= top:
            wins += 1

    win_rate = wins / mc_runs
    avg_score = sum(scores) / len(scores)
    avg_purse = sum(purses) / len(purses)
    budget_util = max(0.0, 1.0 - avg_purse / 120.0)
    t_max = _theoretical_max(pool)
    return win_rate * 0.55 + (avg_score / t_max) * 0.25 + budget_util * 0.20


# ── Genetic Algorithm ─────────────────────────────────────────────────────────
# Improvements over vanilla GA:
#   • Self-adaptive mutation: each individual carries per-dimension σ values
#     that evolve via log-normal update (Schwefel 1995).
#   • Stagnation restart: when best fitness has not improved by ≥ 0.0005
#     for STAGNATION_PATIENCE consecutive generations, the bottom 35% of
#     the population is replaced with fresh random individuals, breaking
#     out of local optima without discarding the elite.
#   • Fitness averaging: each candidate is evaluated twice and averaged to
#     reduce stochastic noise from the Monte Carlo inner loop.

_STAGNATION_PATIENCE = 8       # gens with < ε improvement before restart
_STAGNATION_EPS = 5e-4         # minimum improvement threshold
_STAGNATION_RESET_FRAC = 0.35  # fraction of pop to reseed on stagnation
_TAU_GLOBAL = None             # computed from ndim at runtime
_TAU_LOCAL = None


def _ga_evolve(
    pool: list[dict],
    target_name: str,
    opponent_names: list[str],
    *,
    generations: int,
    mc_runs: int,
    population: int,
    evolved: bool,
    params_path: Path | None,
) -> EvoResult:
    """
    Self-adaptive GA with BLX-α crossover, log-normal σ evolution,
    stagnation restarts, and 2× fitness averaging.
    """
    import dataclasses as _dc

    _, ParamsCls = STRATEGY_REGISTRY[target_name]
    default_params = load_params(target_name, params_path, evolved=evolved)
    bounds = default_params.get_bounds()
    scalar_fields = [
        f for f in _dc.fields(default_params)
        if isinstance(getattr(default_params, f.name), (int, float))
    ]
    ndim = len(bounds)

    # Self-adaptive learning rates (Schwefel 1995)
    tau_global = 1.0 / math.sqrt(2.0 * ndim)
    tau_local  = 1.0 / math.sqrt(2.0 * math.sqrt(ndim))
    sigma_min  = 1e-4
    # Initial σ per dimension = 20% of param range
    sigma_init = [0.20 * (hi - lo) for lo, hi in bounds]

    def clamp(vec: list[float]) -> list[float]:
        return [min(hi, max(lo, v)) for v, (lo, hi) in zip(vec, bounds)]

    def random_individual() -> tuple[list[float], list[float]]:
        x = [random.uniform(lo, hi) for lo, hi in bounds]
        s = list(sigma_init)
        return x, s

    def params_from_vec(vec: list[float]) -> StrategyParams:
        kwargs = {
            f.name: type(getattr(default_params, f.name))(v)
            for f, v in zip(scalar_fields, vec)
        }
        return ParamsCls(**kwargs)

    def evaluate(vec: list[float]) -> float:
        """Average of 2 evaluations to suppress MC noise."""
        f1 = _evaluate_fitness(pool, target_name, params_from_vec(vec),
                               opponent_names, mc_runs, evolved, params_path)
        f2 = _evaluate_fitness(pool, target_name, params_from_vec(vec),
                               opponent_names, mc_runs, evolved, params_path)
        return (f1 + f2) / 2.0

    # Initialise population: list of (x, sigma, fitness)
    pop: list[tuple[list[float], list[float]]] = [random_individual() for _ in range(population)]
    # Seed first individual with defaults
    pop[0] = ([getattr(default_params, f.name) for f in scalar_fields], list(sigma_init))

    fitness = [evaluate(x) for x, _ in pop]

    fitness_history: list[float] = [max(fitness)]
    stagnation_counter = 0
    prev_best = fitness_history[0]
    start = time.time()

    for gen in range(generations):
        ranked = sorted(range(population), key=lambda i: fitness[i], reverse=True)
        new_pop: list[tuple[list[float], list[float]]] = []

        # Elitism: carry top-2 unchanged
        new_pop.extend([pop[i] for i in ranked[:2]])

        while len(new_pop) < population:
            # Tournament selection (k=3)
            def tournament() -> tuple[list[float], list[float]]:
                cands = random.sample(range(population), min(3, population))
                return pop[max(cands, key=lambda i: fitness[i])]

            (x1, s1), (x2, s2) = tournament(), tournament()

            # BLX-α crossover on x and σ (α=0.5)
            alpha = 0.5
            cx, cs = [], []
            for v1, v2, sv1, sv2, (lo, hi) in zip(x1, x2, s1, s2, bounds):
                spread = abs(v2 - v1)
                lo_c = min(v1, v2) - alpha * spread
                hi_c = max(v1, v2) + alpha * spread
                cx.append(random.uniform(max(lo, lo_c), min(hi, hi_c)))
                cs.append((sv1 + sv2) / 2.0)  # inherit average σ

            # Self-adaptive σ update (log-normal)
            global_step = random.gauss(0, tau_global)
            child_s = [
                max(sigma_min, s * math.exp(global_step + random.gauss(0, tau_local)))
                for s in cs
            ]

            # Mutate x using updated σ
            child_x = clamp([
                v + child_s[i] * random.gauss(0, 1.0)
                for i, v in enumerate(cx)
            ])
            new_pop.append((child_x, child_s))

        pop = new_pop
        fitness = [evaluate(x) for x, _ in pop]

        best_f = max(fitness)
        fitness_history.append(best_f)

        # Stagnation detection + restart
        if best_f - prev_best >= _STAGNATION_EPS:
            prev_best = best_f
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= _STAGNATION_PATIENCE:
            n_reset = max(1, int(population * _STAGNATION_RESET_FRAC))
            worst_idx = sorted(range(population), key=lambda i: fitness[i])[:n_reset]
            for idx in worst_idx:
                pop[idx] = random_individual()
                fitness[idx] = evaluate(pop[idx][0])
            stagnation_counter = 0
            prev_best = max(fitness)
            print(f"  ↺  Stagnation restart at gen {gen+1} — reseeded {n_reset} individuals")

        print(f"  GA gen {gen+1:>3}/{generations}  best={best_f:.4f}  "
              f"σ_mean={sum(pop[ranked[0]][1])/ndim:.4f}")

    best_idx = max(range(population), key=lambda i: fitness[i])
    best_params = params_from_vec(pop[best_idx][0])
    return EvoResult(
        strategy=target_name,
        algo="ga",
        best_params=best_params,
        best_fitness=fitness[best_idx],
        fitness_history=fitness_history,
        duration_s=time.time() - start,
    )


# ── CMA-ES ────────────────────────────────────────────────────────────────────
def _cmaes_evolve(
    pool: list[dict],
    target_name: str,
    opponent_names: list[str],
    *,
    generations: int,
    mc_runs: int,
    evolved: bool,
    params_path: Path | None,
) -> EvoResult:
    """CMA-ES via the `cma` library."""
    try:
        import cma  # type: ignore
    except ImportError:
        raise ImportError("Install `cma`: uv add cma  (or pip install cma)")

    _, ParamsCls = STRATEGY_REGISTRY[target_name]
    default_params = load_params(target_name, params_path, evolved=evolved)
    bounds = default_params.get_bounds()
    scalar_fields = [
        f for f in __import__("dataclasses").fields(default_params)
        if isinstance(getattr(default_params, f.name), (int, float))
    ]

    x0 = [getattr(default_params, f.name) for f in scalar_fields]
    # Normalise to [0,1] domain for CMA-ES, then un-normalise inside fitness
    lo_arr = [b[0] for b in bounds]
    hi_arr = [b[1] for b in bounds]

    def normalise(vec):
        return [(v - lo) / max(hi - lo, 1e-9) for v, lo, hi in zip(vec, lo_arr, hi_arr)]

    def unnormalise(vec):
        return [lo + v * (hi - lo) for v, lo, hi in zip(vec, lo_arr, hi_arr)]

    def params_from_vec(vec: list[float]) -> StrategyParams:
        raw = unnormalise(vec)
        raw = [min(hi, max(lo, v)) for v, lo, hi in zip(raw, lo_arr, hi_arr)]
        kwargs = {f.name: type(getattr(default_params, f.name))(v)
                  for f, v in zip(scalar_fields, raw)}
        return ParamsCls(**kwargs)

    fitness_history: list[float] = []
    start = time.time()

    # Clip x0 to strictly inside param bounds before normalising.
    # If evolved params drifted outside declared bounds, or a param sits
    # exactly on a boundary, CMA-ES's internal boundary handler raises
    # ValueError("argument of inverse must be within the given bounds").
    _EPS = 1e-6
    x0_clipped = [
        min(hi - _EPS, max(lo + _EPS, v))
        for v, lo, hi in zip(x0, lo_arr, hi_arr)
    ]
    x0_norm = normalise(x0_clipped)
    # Clamp to (0, 1) in normalised space as a second safety net
    x0_norm = [min(1.0 - _EPS, max(_EPS, v)) for v in x0_norm]

    # Use a smaller initial sigma for narrow domains (< 3 params)
    ndim = len(x0_norm)
    sigma0 = 0.15 if ndim <= 3 else 0.25

    es = cma.CMAEvolutionStrategy(
        x0_norm,
        sigma0,
        {
            "maxiter": generations,
            "bounds": [[0.0] * ndim, [1.0] * ndim],
            "verbose": -9,
        },
    )

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            p = params_from_vec(sol)
            f = _evaluate_fitness(
                pool, target_name, p, opponent_names, mc_runs, evolved, params_path
            )
            fitnesses.append(-f)  # CMA-ES minimises
        es.tell(solutions, fitnesses)
        best_f = -min(fitnesses)
        fitness_history.append(best_f)
        print(f"  CMA-ES iter {es.countiter:>3}  best_fitness={best_f:.4f}")

    best_params = params_from_vec(es.result.xbest)
    return EvoResult(
        strategy=target_name,
        algo="cmaes",
        best_params=best_params,
        best_fitness=-es.result.fbest,
        fitness_history=fitness_history,
        duration_s=time.time() - start,
    )


# ── Public entry point ────────────────────────────────────────────────────────
_DEFAULT_EVOLVED_PATH = Path(__file__).parent.parent / "evolved_params.json"


def evolve_strategy(
    pool: list[dict],
    target_strategy: str,
    opponent_strategies: list[str] | None = None,
    *,
    algo: str = "ga",               # "ga" | "cmaes" | "both"
    generations: int = 50,
    mc_runs: int = 20,              # auctions per fitness evaluation (kept low for speed)
    population: int = 40,           # GA only
    evolved: bool = False,          # whether opponents use their evolved params
    params_path: str | Path | None = None,
    save_path: str | Path | None = None,
    plot: bool = True,
) -> dict[str, EvoResult]:
    """
    Evolve parameters for target_strategy against a fixed opponent field.

    Args:
        pool:               player pool from build_pool()
        target_strategy:    name to evolve (must be in STRATEGY_REGISTRY)
        opponent_strategies: opponents; defaults to all other strategies
        algo:               "ga", "cmaes", or "both"
        generations:        number of generations / CMA-ES iterations
        mc_runs:            auctions per fitness call (speed vs. accuracy trade-off)
        population:         GA population size
        evolved:            if True, opponents use their previously evolved params
        params_path:        path to strategy_params.json
        save_path:          where to write evolved_params.json (default workspace root)
        plot:               show a matplotlib fitness curve when done

    Returns:
        dict mapping algo name → EvoResult
    """
    if target_strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {target_strategy!r}")

    opponent_names = opponent_strategies or [
        nm for nm in STRATEGY_REGISTRY if nm != target_strategy
    ]
    p_path = Path(params_path) if params_path else None
    s_path = Path(save_path) if save_path else _DEFAULT_EVOLVED_PATH

    results: dict[str, EvoResult] = {}

    if algo in ("ga", "both"):
        print(f"\n[GA] Evolving {target_strategy} …")
        r = _ga_evolve(
            pool, target_strategy, opponent_names,
            generations=generations, mc_runs=mc_runs,
            population=population, evolved=evolved, params_path=p_path,
        )
        results["ga"] = r

    if algo in ("cmaes", "both"):
        print(f"\n[CMA-ES] Evolving {target_strategy} …")
        r = _cmaes_evolve(
            pool, target_strategy, opponent_names,
            generations=generations, mc_runs=mc_runs,
            evolved=evolved, params_path=p_path,
        )
        results["cmaes"] = r

    # Pick the best across algos and save as "evolved"
    best_result = max(results.values(), key=lambda r: r.best_fitness)
    save_params(target_strategy, best_result.best_params, key="evolved", path=p_path)

    # Persist full evo history to evolved_params.json
    s_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if s_path.exists():
        with open(s_path) as f:
            existing = json.load(f)
    existing[target_strategy] = {
        algo_name: {
            "best_fitness": r.best_fitness,
            "fitness_history": r.fitness_history,
            "duration_s": r.duration_s,
            "best_params": r.best_params.to_dict(),
        }
        for algo_name, r in results.items()
    }
    with open(s_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n✅ Best fitness: {best_result.best_fitness:.4f} ({best_result.algo.upper()})")
    print(f"   Saved to {s_path}")

    if plot:
        _plot_fitness(results, target_strategy)

    return results


def _plot_fitness(results: dict[str, EvoResult], strategy_name: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("  (matplotlib not available — skipping fitness plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for algo_name, r in results.items():
        ax.plot(r.fitness_history, label=algo_name.upper(), linewidth=2)
    ax.set_xlabel("Generation / Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Evolution — {strategy_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fitness_{strategy_name}.png", dpi=150)
    print(f"  Fitness curve saved to fitness_{strategy_name}.png")
    plt.show()
