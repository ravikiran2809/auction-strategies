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

# Register PTA (dynamic) strategies into the global registry at startup
from engine.pta_strategies import register_pta_strategies
register_pta_strategies()

# Load player intel (injury/form) — makes intel_mult() active for all CLI runs
from engine.intel import load_intel as _load_intel
_load_intel()

# Register persona strategies if personas.json exists
from pathlib import Path as _Path
if _Path("personas.json").exists():
    from engine.personas import register_persona_strategies
    register_persona_strategies()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_season_weights(raw: str) -> dict[str, float]:
    """Parse '2022:0.3,2023:0.6,2024:1.0' into {year: weight}."""
    result: dict[str, float] = {}
    for part in raw.split(","):
        year, _, weight = part.partition(":")
        result[year.strip()] = float(weight.strip())
    return result


def _parse_extra_seasons(raw: str | None) -> tuple[dict[str, float], list[str]]:
    """Parse '2025:path/to/file.csv:1.0,2026:path/to/2026.csv:0.8' into
    (season_weights, extra_csv_paths).
    Format per entry: year:csv_path:weight  (weight optional, defaults to 1.0)
    The first CSV path is always the primary ipl_player_stats.csv; extra paths
    come from entries that are NOT the default CSV.
    """
    if not raw:
        return {"2025": 1.0}, []
    weights: dict[str, float] = {}
    extra_paths: list[str] = []
    for part in raw.split("|"):
        parts = part.strip().split(":")
        if len(parts) < 2:
            continue
        year = parts[0].strip()
        csv_p = parts[1].strip()
        weight = float(parts[2]) if len(parts) > 2 else 1.0
        weights[year] = weight
        if csv_p and csv_p != "default":
            extra_paths.append(csv_p)
    return weights, extra_paths


def _build_pool(args) -> list[dict]:
    from engine.pool import build_pool, WeightedSeasonModel, RecentFormModel, NormalisedSeasonModel

    extra_paths: list[str] = []

    if args.model == "recent":
        model = RecentFormModel(last_n_matches=args.last_n)
    elif getattr(args, "extra_seasons", None):
        weights, extra_paths = _parse_extra_seasons(args.extra_seasons)
        incomplete = set(getattr(args, "incomplete_seasons", None) or [])
        model = NormalisedSeasonModel(
            weights,
            incomplete_seasons=incomplete if incomplete else None,
        )
    else:
        weights = _parse_season_weights(args.seasons) if args.seasons else {"2025": 1.0}
        model = WeightedSeasonModel(weights)

    print(f"Building pool (model={args.model}, vorp={args.vorp_method}) …")
    pool = build_pool(
        model=model,
        extra_csv_paths=extra_paths if extra_paths else None,
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
    from engine.simulation import run_monte_carlo, print_mc_results, analyze_strategy_picks, print_strategy_insights

    if Path("player_pool.json").exists() and not args.rebuild:
        from engine.export import load_pool
        pool = load_pool()
        print(f"Loaded existing pool ({len(pool)} players). Use --rebuild to regenerate.")
    else:
        pool = _build_pool(args)
        export_pool(pool)

    # --marquee-pressure boosts T1/T2 bids on all STATIC strategies in the field,
    # simulating a live human auction where opponents overbid on star players.
    # Used for grid-search calibration of marquee_recovery_mult on SCB/ARM.
    marquee_pressure = getattr(args, "marquee_pressure", 1.0) or 1.0

    names = args.strategies or None
    results = run_monte_carlo(
        pool, names, args.mc,
        field_size=args.field_size,
        evolved=args.evolved,
        marquee_pressure=marquee_pressure,
    )
    print_mc_results(results)
    out = export_insights(pool, results)
    print(f"  ✅ Insights saved → {out}")

    if not args.no_insights:
        print("\nRunning pick analysis (150 auctions) …")
        pick_stats = analyze_strategy_picks(
            pool, names, n=150, field_size=args.field_size, evolved=args.evolved
        )
        print_strategy_insights(pick_stats, results, pool)


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
        field_size=args.field_size,
        marquee_pressure=args.marquee_pressure,
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


def cmd_persona_sim(args):
    """
    Run a persona-vs-strategies simulation:
      - All 6 human personas as opponents (from personas.json)
      - Our target strategies (default: SquadCompletionBidder + AdaptiveRecoveryManager + PersonaCounter)
      - Field = all 6 personas + our strategies, capped at field_size per auction
    Prints a counter-playbook report after the MC run.
    """
    from engine.export import load_pool
    from engine.simulation import run_monte_carlo, print_mc_results
    from engine.strategies import STRATEGY_REGISTRY
    from engine.personas import load_personas

    if not Path("player_pool.json").exists():
        print("No player_pool.json found. Run `python main.py build-pool` first.")
        sys.exit(1)
    if not Path("personas.json").exists():
        print("No personas.json found. Run `python build_personas.py` first.")
        sys.exit(1)

    pool = load_pool()
    personas = load_personas()
    persona_names = [f"Persona_{t.replace(' ', '_')}" for t in personas]

    our_strategies = args.strategies or ["SquadCompletionBidder", "AdaptiveRecoveryManager", "PersonaCounter"]
    all_names = persona_names + [s for s in our_strategies if s not in persona_names]

    # Validate
    unknown = [n for n in all_names if n not in STRATEGY_REGISTRY]
    if unknown:
        print(f"Unknown strategies: {unknown}")
        sys.exit(1)

    print(f"\n🎭 PERSONA SIMULATION")
    print(f"   Personas ({len(persona_names)}): {', '.join(persona_names)}")
    print(f"   Our strategies ({len(our_strategies)}): {', '.join(our_strategies)}")
    print(f"   Field size: {args.field_size} | Runs: {args.mc}\n")

    results = run_monte_carlo(pool, all_names, args.mc, field_size=args.field_size, evolved=args.evolved)
    print_mc_results(results)

    # ── Counter-playbook report ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  COUNTER-PLAYBOOK REPORT")
    print("=" * 80)

    personas_data = load_personas()
    print("\n  OPPONENT PROFILES:")
    for team, persona in sorted(personas_data.items()):
        pr = persona["profile"]
        pname = f"Persona_{team.replace(' ', '_')}"
        r = results.get(pname)
        win_str = f"{r.win_share*100:.1f}% WS, {r.conditional_win_rate*100:.1f}% WR/gm" if r else "not in results"
        print(f"\n  {team} [{persona['archetype']}] — {win_str}")
        print(f"    Overall ratio : {pr['overall_ratio']:.2f}x fair value")
        print(f"    T1 / T2 / T3  : {pr['t1_ratio']:.2f}x / {pr['t2_ratio']:.2f}x / {pr['t3_ratio']:.2f}x")
        print(f"    Early / Late  : {pr['early_ratio']:.2f}x / {pr['late_ratio']:.2f}x")
        print(f"    Budget use    : {pr['budget_pct']*100:.0f}%")

        # Generate tactical advice
        tactics = []
        if pr['t1_ratio'] < 0.65:
            tactics.append("→ Underpays T1 — STEAL marquee players before they engage (bid 1.5-1.8x fair value early)")
        if pr['t3_ratio'] > 2.0:
            tactics.append(f"→ Massively overbids T3 ({pr['t3_ratio']:.1f}x) — floor-bid T3 to drain their purse; don't compete")
        if pr['late_ratio'] > 1.5 and pr['early_ratio'] < 0.8:
            tactics.append("→ LateSweeper — deploy your budget in first half; they're quiet early")
        if pr['budget_pct'] > 0.95:
            tactics.append("→ Deploys nearly all budget — they'll be aggressive when short on mandatory slots")
        if not tactics:
            tactics.append("→ Balanced bidder — no strong exploitable pattern")
        for t in tactics:
            print(f"    {t}")

    print("\n" + "-" * 80)
    print("  OUR STRATEGY PERFORMANCE vs PERSONA FIELD:")
    for name in our_strategies:
        r = results.get(name)
        if r:
            print(f"  {name:<30} WinShare={r.win_share*100:.1f}%  WR/gm={r.conditional_win_rate*100:.1f}%  "
                  f"AvgPts={r.avg_score:.0f}  BudgetUse={r.budget_utilization*100:.0f}%")

    print("\n  GLOBAL COUNTER ADVICE:")
    all_t1 = [personas_data[t]["profile"]["t1_ratio"] for t in personas_data]
    all_late = [personas_data[t]["profile"]["late_ratio"] for t in personas_data]
    avg_t1 = sum(all_t1) / len(all_t1) if all_t1 else 1.0
    avg_late = sum(all_late) / len(all_late) if all_late else 1.0
    print(f"  Mean T1 pay ratio across all opponents: {avg_t1:.2f}x")
    print(f"  Mean late-auction aggression:           {avg_late:.2f}x")
    if avg_t1 < 0.7:
        print("  ✅ T1 WINDOW: Opponents consistently undervalue marquee players.")
        print("     Bid up to 1.6× fair value on T1 in first 30 lots — you will win them cheaply.")
    if avg_late > 1.4:
        print("  ⚠️  LATE-ROUND INFLATION: Opponents spike bids in final third.")
        print("     Prioritise early budget deployment. Avoid T3 bidding wars.")
    print("=" * 80)


def cmd_coevolve(args):
    """
    Sequentially evolve every strategy (or a named subset) against all others.
    On each round, each strategy faces the current best params of its opponents,
    then saves its improved params so the next strategy faces a tougher field.
    Runs --rounds times to allow strategies to adapt to each other's evolution.
    """
    from engine.export import load_pool
    from engine.simulation import evolve_strategy
    from engine.strategies import STRATEGY_REGISTRY

    if not Path("player_pool.json").exists():
        print("No player_pool.json found. Run `python main.py build-pool` first.")
        sys.exit(1)
    pool = load_pool()

    names = args.strategies or list(STRATEGY_REGISTRY.keys())
    print(f"\n🧬 Co-evolution: {len(names)} strategies × {args.rounds} round(s)")
    print(f"   algo={args.algo}  generations={args.generations}  mc_runs={args.mc_runs}\n")

    for rnd in range(1, args.rounds + 1):
        print(f"{'─'*60}")
        print(f"  Round {rnd}/{args.rounds}")
        print(f"{'─'*60}")
        for nm in names:
            opps = [o for o in names if o != nm]
            print(f"\n  → Evolving {nm} vs {len(opps)} opponents …")
            evolve_strategy(
                pool, nm, opps,
                algo=args.algo,
                generations=args.generations,
                mc_runs=args.mc_runs,
                population=args.population,
                evolved=True,
                plot=False,
            )
    print("\n✅ Co-evolution complete. Run `python main.py simulate --evolved` to see results.")


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
    bp.add_argument("--extra-seasons", type=str, default=None,
                    help="Multi-season with extra CSVs (pipe-separated): "
                         "'2025:default:1.0|2026:ipl_2026.csv:0.8'. "
                         "Use 'default' for the primary ipl_player_stats.csv. "
                         "Activates NormalisedSeasonModel with auto-detection of incomplete seasons.")
    bp.add_argument("--incomplete-seasons", nargs="+", default=None,
                    help="Years to treat as incomplete for normalisation e.g. '2026'. "
                         "If omitted, auto-detected from match counts.")
    bp.add_argument("--last-n", type=int, default=10,
                    help="Matches for RecentFormModel")
    bp.add_argument("--min-matches", type=int, default=5)
    bp.add_argument("--pool-size", type=int, default=130)
    bp.add_argument("--teams", type=int, default=6)
    bp.add_argument("--vorp-method", choices=["quantile", "depth"], default="quantile")
    bp.set_defaults(func=cmd_build_pool)

    # ── simulate ──────────────────────────────────────────────────────────────
    sim = sub.add_parser("simulate", help="Run Monte Carlo across strategies")
    sim.add_argument("--mc", type=int, default=500, help="Number of auction rounds")
    sim.add_argument("--field-size", type=int, default=6,
                     help="Managers per auction (default: 6 to match real game)")
    sim.add_argument("--strategies", nargs="+", default=None, help="Strategy names (default: all)")
    sim.add_argument("--evolved", action="store_true", help="Use evolved params")
    sim.add_argument("--rebuild", action="store_true", help="Rebuild pool before simulating")
    sim.add_argument("--no-insights", action="store_true", help="Skip player pick analysis")
    # Pool args (used when --rebuild)
    sim.add_argument("--model", choices=["weighted", "recent"], default="weighted")
    sim.add_argument("--seasons", type=str, default=None)
    sim.add_argument("--last-n", type=int, default=10)
    sim.add_argument("--min-matches", type=int, default=5)
    sim.add_argument("--pool-size", type=int, default=130)
    sim.add_argument("--teams", type=int, default=6)
    sim.add_argument("--vorp-method", choices=["quantile", "depth"], default="quantile")
    sim.add_argument("--marquee-pressure", type=float, default=1.0,
                     help="T1/T2 bid multiplier applied to all STATIC strategies (default 1.0). "
                          "Values > 1 simulate human overbidding on marquee players. "
                          "Use with grid search to calibrate marquee_recovery_mult on SCB/ARM. "
                          "Example: --marquee-pressure 1.75")
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
    evo.add_argument("--field-size", type=int, default=6,
                     help="Number of teams per auction during evolution (default: 6; use 8 for real game)")
    evo.add_argument("--marquee-pressure", type=float, default=1.0,
                     help="T1/T2 bid multiplier applied to opponents in first 30 lots (1.0 = off; 1.4 = realistic human overbid)")
    evo.set_defaults(func=cmd_evolve)

    # ── coevolve ──────────────────────────────────────────────────────────────
    cevo = sub.add_parser("coevolve", help="Co-evolve all strategies against each other")
    cevo.add_argument("--rounds", type=int, default=2,
                      help="Number of full co-evolution rounds (default: 2)")
    cevo.add_argument("--strategies", nargs="+", default=None,
                      help="Subset of strategies to co-evolve (default: all)")
    cevo.add_argument("--algo", choices=["ga", "cmaes", "both"], default="ga")
    cevo.add_argument("--generations", type=int, default=30)
    cevo.add_argument("--mc-runs", type=int, default=20)
    cevo.add_argument("--population", type=int, default=30)
    cevo.set_defaults(func=cmd_coevolve)

    # ── persona-sim ───────────────────────────────────────────────────────────
    psim = sub.add_parser("persona-sim", help="Simulate vs human persona opponents + counter-playbook")
    psim.add_argument("--mc", type=int, default=200, help="Number of auction rounds")
    psim.add_argument("--field-size", type=int, default=6)
    psim.add_argument("--strategies", nargs="+", default=None,
                      help="Our strategies to test (default: SCB + ARM + PersonaCounter)")
    psim.add_argument("--evolved", action="store_true")
    psim.set_defaults(func=cmd_persona_sim)

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

