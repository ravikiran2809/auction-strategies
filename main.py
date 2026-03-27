"""
main.py — Unified CLI for the IPL Auction Strategy Platform
============================================================
Commands:
  build-pool   Score data, generate player pool, save to player_pool.json
  simulate     Run Monte Carlo across all (or selected) strategies
  evolve       Run GA / CMA-ES to optimise strategy params
  serve        Start the FastAPI advisor server

Examples:
  python main.py build-pool
  python main.py build-pool --model weighted --seasons 2022:0.3,2023:0.6,2024:1.0
  python main.py build-pool --model recent --last-n 10
  python main.py simulate --mc 500
  python main.py simulate --mc 200 --strategies TierSniper NominationGambler
  python main.py evolve --strategy TierSniper --algo ga --generations 50
  python main.py evolve --strategy TierSniper --algo both --mc-runs 30
  python main.py serve --port 8080
  python main.py run --quiet           # single live auction (verbose by default)
"""

import argparse
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_season_weights(raw: str) -> dict[str, float]:
    """Parse '2022:0.3,2023:0.6,2024:1.0' into {year: weight}."""
    result: dict[str, float] = {}
    for part in raw.split(","):
        year, _, weight = part.partition(":")
        result[year.strip()] = float(weight.strip())
    return result


def _build_pool(args) -> list[dict]:
    from engine.pool import build_pool, WeightedSeasonModel, RecentFormModel

    if args.model == "recent":
        model = RecentFormModel(last_n_matches=args.last_n)
    else:
        weights = _parse_season_weights(args.seasons) if args.seasons else {"2024": 1.0}
        model = WeightedSeasonModel(weights)

    print(f"Building pool (model={args.model}, vorp={args.vorp_method}) …")
    pool = build_pool(
        model=model,
        min_matches=args.min_matches,
        pool_size=args.pool_size,
        num_teams=args.teams,
        vorp_method=args.vorp_method,
    )
    t = {1: 0, 2: 0, 3: 0}
    for p in pool:
        t[p["tier"]] += 1
    print(
        f"  {len(pool)} players | T1:{t[1]}  T2:{t[2]}  T3:{t[3]}"
        f" | top: {pool[0]['player_name']} ({pool[0]['projected_points']:.0f}pts)"
    )
    return pool


# ── Subcommand handlers ───────────────────────────────────────────────────────
def cmd_build_pool(args):
    from engine.export import export_pool

    pool = _build_pool(args)
    out = export_pool(pool)
    print(f"  ✅ Saved → {out}")


def cmd_simulate(args):
    from engine.export import load_pool, export_pool, export_insights
    from engine.simulation import run_monte_carlo, print_mc_results

    if Path("player_pool.json").exists() and not args.rebuild:
        from engine.export import load_pool
        pool = load_pool()
        print(f"Loaded existing pool ({len(pool)} players). Use --rebuild to regenerate.")
    else:
        pool = _build_pool(args)
        export_pool(pool)

    names = args.strategies or None
    results = run_monte_carlo(pool, names, args.mc, evolved=args.evolved)
    print_mc_results(results)
    out = export_insights(pool, results)
    print(f"  ✅ Insights saved → {out}")


def cmd_evolve(args):
    from engine.export import load_pool
    from engine.simulation import evolve_strategy

    if not Path("player_pool.json").exists():
        print("No player_pool.json found. Run `python main.py build-pool` first.")
        sys.exit(1)
    pool = load_pool()

    opps = args.opponents or None
    results = evolve_strategy(
        pool,
        args.strategy,
        opps,
        algo=args.algo,
        generations=args.generations,
        mc_runs=args.mc_runs,
        population=args.population,
        evolved=args.evolved,
        plot=not args.no_plot,
    )
    for algo_name, r in results.items():
        print(f"\n[{algo_name.upper()}] Best params for {args.strategy}:")
        for k, v in r.best_params.to_dict().items():
            print(f"  {k}: {v}")


def cmd_run(args):
    from engine.export import load_pool
    from engine.strategies import STRATEGY_REGISTRY, instantiate
    from engine.auction import run_auction, print_results

    if not Path("player_pool.json").exists():
        print("No player_pool.json found. Run `python main.py build-pool` first.")
        sys.exit(1)
    pool = load_pool()

    names = args.strategies or list(STRATEGY_REGISTRY.keys())
    agents = [instantiate(nm, evolved=args.evolved) for nm in names]
    verbose = not args.quiet
    if verbose:
        print("🏏 LIVE ASCENDING-BID AUCTION")
        print(f"   Purse ₹120Cr | Roster 13–15 | Pool {len(pool)} | {len(agents)} managers")
        print("═" * 72)
    unsold = run_auction([p.copy() for p in pool], agents, verbose=verbose)
    print_results(agents, unsold)


def cmd_serve(args):
    import uvicorn  # type: ignore
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


# ── CLI definition ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="IPL Auction Strategy Platform",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build-pool ────────────────────────────────────────────────────────────
    bp = sub.add_parser("build-pool", help="Generate player pool from CSV data")
    bp.add_argument("--model", choices=["weighted", "recent"], default="weighted")
    bp.add_argument("--seasons", type=str, default=None,
                    help="Season weights e.g. '2022:0.3,2023:0.6,2024:1.0'")
    bp.add_argument("--last-n", type=int, default=10,
                    help="Matches for RecentFormModel")
    bp.add_argument("--min-matches", type=int, default=5)
    bp.add_argument("--pool-size", type=int, default=80)
    bp.add_argument("--teams", type=int, default=6)
    bp.add_argument("--vorp-method", choices=["quantile", "depth"], default="quantile")
    bp.set_defaults(func=cmd_build_pool)

    # ── simulate ──────────────────────────────────────────────────────────────
    sim = sub.add_parser("simulate", help="Run Monte Carlo across strategies")
    sim.add_argument("--mc", type=int, default=500, help="Number of auction simulations")
    sim.add_argument("--strategies", nargs="+", default=None, help="Strategy names (default: all)")
    sim.add_argument("--evolved", action="store_true", help="Use evolved params")
    sim.add_argument("--rebuild", action="store_true", help="Rebuild pool before simulating")
    # Pool args (used when --rebuild)
    sim.add_argument("--model", choices=["weighted", "recent"], default="weighted")
    sim.add_argument("--seasons", type=str, default=None)
    sim.add_argument("--last-n", type=int, default=10)
    sim.add_argument("--min-matches", type=int, default=5)
    sim.add_argument("--pool-size", type=int, default=80)
    sim.add_argument("--teams", type=int, default=6)
    sim.add_argument("--vorp-method", choices=["quantile", "depth"], default="quantile")
    sim.set_defaults(func=cmd_simulate)

    # ── evolve ────────────────────────────────────────────────────────────────
    evo = sub.add_parser("evolve", help="Optimise strategy params via GA / CMA-ES")
    evo.add_argument("--strategy", required=True, help="Strategy name to evolve")
    evo.add_argument("--algo", choices=["ga", "cmaes", "both"], default="ga")
    evo.add_argument("--generations", type=int, default=50)
    evo.add_argument("--mc-runs", type=int, default=20,
                     help="Auctions per fitness evaluation")
    evo.add_argument("--population", type=int, default=40, help="GA population size")
    evo.add_argument("--opponents", nargs="+", default=None,
                     help="Opponent strategy names (default: all others)")
    evo.add_argument("--evolved", action="store_true",
                     help="Opponents use their evolved params")
    evo.add_argument("--no-plot", action="store_true", help="Skip fitness curve plot")
    evo.set_defaults(func=cmd_evolve)

    # ── run ───────────────────────────────────────────────────────────────────
    run = sub.add_parser("run", help="Run a single live auction (pretty-printed)")
    run.add_argument("--strategies", nargs="+", default=None)
    run.add_argument("--evolved", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

    # ── serve ─────────────────────────────────────────────────────────────────
    srv = sub.add_parser("serve", help="Start FastAPI advisor server")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8080)
    srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

