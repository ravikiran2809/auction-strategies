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
class MCResult:
    strategy: str
    wins: int
    runs: int
    avg_score: float
    std_score: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.runs if self.runs > 0 else 0.0


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
    evolved: bool = False,
    params_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, MCResult]:
    """
    Run N full auctions with the given strategies.
    Returns per-strategy MCResult containing win_rate, avg_score, std_score.
    """
    names = strategy_names or list(STRATEGY_REGISTRY.keys())

    wins: dict[str, int] = {nm: 0 for nm in names}
    scores: dict[str, list[float]] = {nm: [] for nm in names}

    if verbose:
        print(f"\nMonte Carlo: {n} runs × {len(names)} strategies …", end="", flush=True)

    for i in range(n):
        agents = [
            instantiate(nm, evolved=evolved, params_path=params_path)
            for nm in names
        ]
        run_auction([p.copy() for p in pool], agents, verbose=False)
        sim = {a.name: a.season_score() for a in agents}
        top_score = max(sim.values())
        for nm, sc in sim.items():
            scores[nm].append(sc)
            if sc >= top_score:
                wins[nm] += 1
        if verbose and (i + 1) % 100 == 0:
            print(f" {i+1}…", end="", flush=True)

    if verbose:
        print(" done!\n")

    results: dict[str, MCResult] = {}
    for nm in names:
        sc = scores[nm]
        avg = sum(sc) / len(sc) if sc else 0.0
        std = math.sqrt(sum((s - avg) ** 2 for s in sc) / len(sc)) if sc else 0.0
        results[nm] = MCResult(strategy=nm, wins=wins[nm], runs=n, avg_score=avg, std_score=std)

    return results


def print_mc_results(results: dict[str, MCResult]) -> None:
    n = next(iter(results.values())).runs
    print("═" * 72)
    print(f"  📊  MONTE CARLO  ({n} runs, {len(results)} strategies)")
    print("═" * 72)
    print(f"  {'Strategy':<30} {'Win%':>6}  {'AvgPts':>8}  {'StdDev':>7}  Chart")
    for r in sorted(results.values(), key=lambda r: r.win_rate, reverse=True):
        bar = "█" * max(1, int(r.win_rate * 100 / 3))
        print(
            f"  {r.strategy:<30} {r.win_rate*100:>5.1f}%  {r.avg_score:>8.1f}"
            f"  {r.std_score:>7.1f}  {bar}"
        )
    print("═" * 72)
    best = max(results.values(), key=lambda r: r.win_rate)
    print(f"\n  🏆 DOMINANT: {best.strategy}  ({best.win_rate*100:.1f}% wins)\n")


# ── Fitness function ──────────────────────────────────────────────────────────
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
    """Run mc_runs auctions and compute fitness for one param vector."""
    TargetCls, _ = STRATEGY_REGISTRY[target_name]
    opp_agents = [
        instantiate(nm, evolved=evolved, params_path=params_path)
        for nm in opponent_names
    ]

    wins = 0
    scores: list[float] = []
    for _ in range(mc_runs):
        target_agent = TargetCls(name=target_name, params=copy.deepcopy(params))
        agents = [target_agent] + [copy.deepcopy(a) for a in opp_agents]
        run_auction([p.copy() for p in pool], agents, verbose=False)
        sim = {a.name: a.season_score() for a in agents}
        top = max(sim.values())
        sc = sim[target_name]
        scores.append(sc)
        if sc >= top:
            wins += 1

    win_rate = wins / mc_runs
    avg_score = sum(scores) / len(scores)
    t_max = _theoretical_max(pool)
    return win_rate * 0.7 + (avg_score / t_max) * 0.3


# ── Genetic Algorithm ─────────────────────────────────────────────────────────
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
    """BLX-α crossover + Gaussian mutation GA."""
    _, ParamsCls = STRATEGY_REGISTRY[target_name]
    default_params = load_params(target_name, params_path, evolved=evolved)
    bounds = default_params.get_bounds()
    scalar_fields = [
        f for f in __import__("dataclasses").fields(default_params)
        if isinstance(getattr(default_params, f.name), (int, float))
    ]

    def clamp(vec: list[float]) -> list[float]:
        return [min(hi, max(lo, v)) for v, (lo, hi) in zip(vec, bounds)]

    def random_individual() -> list[float]:
        return [random.uniform(lo, hi) for lo, hi in bounds]

    def params_from_vec(vec: list[float]) -> StrategyParams:
        kwargs = {f.name: type(getattr(default_params, f.name))(v)
                  for f, v in zip(scalar_fields, vec)}
        return ParamsCls(**kwargs)

    # Initialise population
    pop = [random_individual() for _ in range(population)]
    # Seed first individual with defaults
    pop[0] = [getattr(default_params, f.name) for f in scalar_fields]

    fitness = [0.0] * population
    for i, ind in enumerate(pop):
        fitness[i] = _evaluate_fitness(
            pool, target_name, params_from_vec(ind), opponent_names,
            mc_runs, evolved, params_path,
        )

    fitness_history: list[float] = [max(fitness)]
    start = time.time()

    for gen in range(generations):
        new_pop: list[list[float]] = []

        # Elitism: carry top-2 unchanged
        ranked = sorted(range(population), key=lambda i: fitness[i], reverse=True)
        new_pop.extend([list(pop[i]) for i in ranked[:2]])

        while len(new_pop) < population:
            # Tournament selection (k=3)
            def tournament() -> list[float]:
                candidates = random.sample(range(population), min(3, population))
                return list(pop[max(candidates, key=lambda i: fitness[i])])

            p1, p2 = tournament(), tournament()
            # BLX-α crossover (α=0.5)
            alpha = 0.5
            child = []
            for v1, v2, (lo, hi) in zip(p1, p2, bounds):
                lo_c = min(v1, v2) - alpha * abs(v2 - v1)
                hi_c = max(v1, v2) + alpha * abs(v2 - v1)
                child.append(random.uniform(max(lo, lo_c), min(hi, hi_c)))
            # Gaussian mutation σ = 0.05 × range
            child = [
                v + random.gauss(0, 0.05 * (hi - lo))
                for v, (lo, hi) in zip(child, bounds)
            ]
            new_pop.append(clamp(child))

        pop = new_pop
        fitness = [0.0] * population
        for i, ind in enumerate(pop):
            fitness[i] = _evaluate_fitness(
                pool, target_name, params_from_vec(ind), opponent_names,
                mc_runs, evolved, params_path,
            )

        best_f = max(fitness)
        fitness_history.append(best_f)
        print(f"  GA gen {gen+1:>3}/{generations}  best_fitness={best_f:.4f}")

    best_idx = max(range(population), key=lambda i: fitness[i])
    best_params = params_from_vec(pop[best_idx])
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

    es = cma.CMAEvolutionStrategy(
        normalise(x0),
        0.3,
        {
            "maxiter": generations,
            "bounds": [[0.0] * len(x0), [1.0] * len(x0)],
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
