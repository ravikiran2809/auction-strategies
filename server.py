"""
server.py — FastAPI advisor server
=====================================
Start with:  python main.py serve  (or uvicorn server:app --reload)

Endpoints:
  GET    /api/health                liveness check
  GET    /api/pool                  current player pool (with overrides applied)
  GET    /api/strategies            registry + params + MC win rates
  POST   /api/override              set manual projected_points for a player
  DELETE /api/override/{name}       remove an override
  GET    /api/playbook              tier-by-tier advice for a strategy
  POST   /api/simulate              trigger fresh MC run (blocking)
  GET    /api/state                 load persisted live auction state
  POST   /api/state                 persist live auction state
  POST   /api/advice                get bid ceiling from a strategy
  POST   /api/advice/sold           notify a completed sale (feeds MarketAnalyzer)
  DELETE /api/session/{tourney_id}  clear MarketAnalyzer history for a tourney
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
# Advisor bot decisions are written to advisor.log (and echoed to stdout).
# View live:  tail -f advisor.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("advisor.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("advisor")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.export import load_pool, export_pool, export_insights
from engine.overrides import load_overrides, set_override, remove_override
from engine.strategies import STRATEGY_REGISTRY, load_params
from engine.pool import build_pool, WeightedSeasonModel

_POOL_PATH            = Path("player_pool.json")
_INSIGHTS_PATH        = Path("insights.json")
_EVOLVED_PATH         = Path("evolved_params.json")
_MASTER_PATH          = Path("player_master.json")
_AUCTION_STATE_PATH   = Path("auction_state.json")


def _build_master_map() -> dict[str, dict]:
    """Build csv_name → master entry lookup from player_master.json."""
    if not _MASTER_PATH.exists():
        return {}
    with open(_MASTER_PATH) as f:
        master = json.load(f)
    return {p["csv_name"]: p for p in master if p.get("csv_name")}


_MASTER_MAP: dict[str, dict] = _build_master_map()

# ── Session-scoped MarketAnalyzers ────────────────────────────────────────────
# One MarketAnalyzer per live tourney, keyed by tourney_id.
# Initialised on the first POST /api/advice/sold for a given tourney.
# Accumulates real sale history so market_mode reflects actual auction dynamics.
# Without this, bids called via /api/advice always see market_mode='normal'.
from engine.market import MarketAnalyzer as _MarketAnalyzer
_sessions: dict[str, _MarketAnalyzer] = {}


def _get_session(tourney_id: str) -> _MarketAnalyzer:
    """Return the MarketAnalyzer for a tourney, creating it if needed."""
    if tourney_id not in _sessions:
        _sessions[tourney_id] = _MarketAnalyzer()
    return _sessions[tourney_id]

app = FastAPI(title="IPL Auction Advisor", version="1.0")

# Allow the HTML file (opened locally) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML advisor at /
_HTML_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_HTML_DIR), html=True), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(_HTML_DIR / "ipl_live_advisor.html")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_pool() -> list[dict]:
    if not _POOL_PATH.exists():
        # Auto-generate with defaults on first request
        pool = build_pool(model=WeightedSeasonModel({"2024": 1.0}))
        export_pool(pool)
    pool = load_pool()
    # Enrich each player with their auction display name from player_master.json
    for p in pool:
        m = _MASTER_MAP.get(p["player_name"])
        if m:
            p["auction_name"] = m["name"]
        else:
            p["auction_name"] = p["player_name"]  # fallback: use csv_name as-is
    return pool


def _load_insights() -> dict:
    if not _INSIGHTS_PATH.exists():
        return {}
    with open(_INSIGHTS_PATH) as f:
        return json.load(f)


def _load_evolved() -> dict:
    if not _EVOLVED_PATH.exists():
        return {}
    with open(_EVOLVED_PATH) as f:
        return json.load(f)


# ── Models ────────────────────────────────────────────────────────────────────
class OverrideRequest(BaseModel):
    player_name: str
    projected_points: float
    note: Optional[str] = ""


class SimulateRequest(BaseModel):
    n: int = 200
    strategies: Optional[list[str]] = None
    evolved: bool = False


class AuctionStateModel(BaseModel):
    me: dict          # {purse: float, roster: [{player_name, role, price, ...}]}
    opponents: list   # [{name, purse, roster: [player_name, ...]}, ...]


class AdviceRequest(BaseModel):
    """
    Generalised bid-advice request — designed to be called from any frontend
    or API consumer that can supply current auction state.

    player_name: csv_name or auction_name of the player currently on the block.
    my_purse:    current remaining purse (₹ Cr).
    my_roster:   list of player identifiers (csv_name or auction_name) in my squad.
    opponents:   list of {name, purse, roster: [player identifiers]}.
    strategy:    which registered strategy to ask for advice.
    evolved:     whether to use CMA-ES evolved params (True) or defaults.
    tourney_id:  if provided, the strategy's MarketAnalyzer is seeded with real
                 sale history accumulated via POST /api/advice/sold for this tourney.
                 Omit for the HTML advisor (it manages its own state); include for
                 the Spring Boot bot integration.
    """
    player_name: str
    my_purse: float = 120.0
    my_roster: list[str] = []
    opponents: list[dict] = []
    strategy: str = "SquadCompletionBidder"
    evolved: bool = True
    tourney_id: Optional[str] = None


class SoldEvent(BaseModel):
    """
    Notifies the advisor that a player was sold at auction.
    Called by AuctionServiceImpl.closeAuction() after every hammer.

    Feeds the session MarketAnalyzer so market_mode ('overbid'/'buyer'/'normal')
    reflects real price history. Without these events, market dampening and boosting
    signals in the strategies are always neutral.

    player_name: display name or csv_name — both accepted.
    price:       final sale price in ₹ Crore (platform sends paisa ÷ 10_000_000).
    buyer:       winning team's display name.
    tourney_id:  FantasyTourney.id — keys the session MarketAnalyzer.
    """
    player_name: str
    price: float
    buyer: str
    tourney_id: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/pool")
def get_pool():
    """Return the full player pool with overrides applied."""
    return {"pool": _get_pool()}


@app.get("/api/strategies")
def get_strategies():
    """Return registry info, current params, and MC win rates if available."""
    insights = _load_insights()
    evolved  = _load_evolved()
    strategy_data = []
    for name in STRATEGY_REGISTRY:
        default_p = load_params(name)
        evolved_p_dict = None
        try:
            ep = load_params(name, evolved=True)
            if ep.to_dict() != default_p.to_dict():
                evolved_p_dict = ep.to_dict()
        except Exception:
            pass

        mc = insights.get("strategies", {}).get(name, {})
        evo_history = evolved.get(name, {})

        strategy_data.append({
            "name":                  name,
            "default_params":        default_p.to_dict(),
            "evolved_params":        evolved_p_dict,
            # insights.json stores win_share; expose as win_rate so UI reads it correctly
            "win_rate":              mc.get("win_share"),
            "win_share":             mc.get("win_share"),
            "conditional_win_rate": mc.get("conditional_win_rate"),
            "avg_score":             mc.get("avg_score"),
            "std_score":             mc.get("std_score"),
            "wins":                  mc.get("wins"),
            "participations":        mc.get("participations"),
            "total_auctions":        mc.get("total_auctions"),
            "runs":                  mc.get("runs"),
            "has_evolved":           evolved_p_dict is not None,
            "evo_history":    {
                algo: d.get("fitness_history", [])
                for algo, d in evo_history.items()
            },
        })
    return {"strategies": strategy_data}


@app.post("/api/override")
def post_override(req: OverrideRequest):
    """Set a manual projected_points override for a player."""
    overrides = set_override(req.player_name, req.projected_points, req.note or "")
    # Regenerate pool with new overrides applied (pool already on disk)
    from engine.overrides import apply_overrides
    pool = apply_overrides(_get_pool(), overrides=overrides)
    export_pool(pool)
    return {
        "ok": True,
        "player": req.player_name,
        "projected_points": req.projected_points,
        "overrides_count": len(overrides),
    }


@app.delete("/api/override/{player_name}")
def delete_override(player_name: str):
    """Remove an override for a player."""
    decoded = player_name.replace("%20", " ")
    overrides = remove_override(decoded)
    from engine.overrides import apply_overrides
    base_pool = load_pool()
    pool = apply_overrides(base_pool, overrides=overrides)
    export_pool(pool)
    return {"ok": True, "player": decoded, "overrides_count": len(overrides)}


@app.get("/api/overrides")
def get_overrides():
    return {"overrides": load_overrides()}


@app.get("/api/playbook")
def get_playbook(strategy: str = "TierSniper", evolved: bool = False):
    """
    Pre-compute tier-by-tier bid advice for the given strategy.
    Returns rough bid multipliers and phase thresholds for real-time use.
    """
    if strategy not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found")
    pool = _get_pool()
    params = load_params(strategy, evolved=evolved)
    p_dict = params.to_dict()

    t1_count = sum(1 for pl in pool if pl.get("tier") == 1)
    t2_count = sum(1 for pl in pool if pl.get("tier") == 2)
    t3_count = sum(1 for pl in pool if pl.get("tier") == 3)
    avg_pts  = sum(pl["projected_points"] for pl in pool) / len(pool) if pool else 80

    # Rough illustrative bid ceilings at default purse/slots (120/16 = ₹7.5/slot)
    cash_per_slot = 120.0 / 16

    advice: dict = {
        "strategy": strategy,
        "params": p_dict,
        "pool_summary": {"t1": t1_count, "t2": t2_count, "t3": t3_count},
        "phases": [],
        "tips": [],
    }

    # Build human-readable phase advice per strategy type
    if "phase_a_threshold" in p_dict:
        advice["phases"] = [
            {
                "name": "Phase A — Conserve",
                "condition": f"{p_dict['phase_a_threshold']}+ elite players remaining",
                "t1_action": f"bid up to ₹{cash_per_slot * p_dict.get('phase_a_t1_mult', 1.5):.1f}Cr",
                "other_action": f"underbid at ₹{cash_per_slot * p_dict.get('phase_a_other_mult', 0.25):.1f}Cr to conserve cash",
            },
            {
                "name": "Phase B — Selective",
                "condition": f"{p_dict['phase_b_threshold']}–{p_dict['phase_a_threshold']-1} elite players remaining",
                "t1_action": "compete on T1 with premium multiplier",
                "other_action": f"T2 at ₹{cash_per_slot * p_dict.get('phase_b_t2_mult', 0.9):.1f}Cr",
            },
            {
                "name": "Phase C — Liquidate",
                "condition": f"< {p_dict['phase_b_threshold']} elite players remaining",
                "t1_action": "full aggression — spend all remaining cash",
                "other_action": "same — liquidate",
            },
        ]
        advice["tips"] = [
            "Track the elite count carefully — phase transitions are your buying signals.",
            "During Phase A, nominate your rivals' target players to drain their purses.",
        ]
    elif "t1_cap" in p_dict:
        cap_cr = 120 * p_dict["t1_cap"]
        advice["phases"] = [
            {
                "name": "Lock One Elite",
                "condition": "Start of auction, no T1 owned",
                "t1_action": f"bid up to ₹{cap_cr:.0f}Cr for first T1",
                "other_action": "ignore T2/T3 until elite is locked",
            },
            {
                "name": "Sweep Depth",
                "condition": "After locking first T1",
                "t1_action": f"sit back unless ≤{p_dict.get('t1_enforce_when_remaining', 3)} T1s remain",
                "other_action": f"T2 at ≤ ₹{cash_per_slot * p_dict.get('t2_mult', 1.1):.1f}Cr",
            },
        ]
        advice["tips"] = [
            "Never be the last bidder on a tier-1 player — buy the 2nd or 3rd in queue.",
            f"Your T1 ceiling is ₹{cap_cr:.0f}Cr. Walk away if the room exceeds it.",
        ]
    else:
        advice["tips"] = [
            "Bid proportionally to player VORP — higher VORP = higher ceiling.",
            "Watch your cash-per-slot ratio; stay above ₹6Cr per remaining slot.",
        ]

    return advice


@app.post("/api/simulate")
def post_simulate(req: SimulateRequest):
    """Trigger a fresh Monte Carlo run (blocking)."""
    from engine.simulation import run_monte_carlo
    pool = _get_pool()
    results = run_monte_carlo(pool, req.strategies, req.n, evolved=req.evolved, verbose=False)
    out = export_insights(pool, results)
    return {
        "ok": True,
        "runs": req.n,
        "strategies": {
            nm: {
                "win_rate": round(r.win_rate, 4),
                "avg_score": round(r.avg_score, 2),
                "std_score": round(r.std_score, 2),
            }
            for nm, r in results.items()
        },
    }


# ── State persistence ────────────────────────────────────────────────────────
@app.get("/api/state")
def get_auction_state():
    """
    Return the last-saved live auction state (my roster + opponent rosters).
    State is written by the HTML advisor after every purchase.
    """
    if not _AUCTION_STATE_PATH.exists():
        return {"me": {"purse": 120.0, "roster": []}, "opponents": []}
    with open(_AUCTION_STATE_PATH) as f:
        return json.load(f)


@app.post("/api/state")
def post_auction_state(req: AuctionStateModel):
    """Persist live auction state to disk so it survives page refreshes."""
    state = {"me": req.me, "opponents": req.opponents}
    with open(_AUCTION_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    return {"ok": True}


# ── Real strategy advice ──────────────────────────────────────────────────────
@app.post("/api/advice")
def post_advice(req: AdviceRequest):
    """
    Run a registered strategy against the supplied auction state and return
    its recommended bid ceiling for the player on the block.

    This is the generalised entry-point: any UI or downstream system can call
    it with state derived from real-time inputs rather than a simulated auction.
    """
    pool = _get_pool()

    # Build a dual-key lookup so callers can use either csv_name or auction_name
    name_to_player: dict[str, dict] = {}
    for p in pool:
        name_to_player[p["player_name"]] = p
        if p.get("auction_name") and p["auction_name"] != p["player_name"]:
            name_to_player[p["auction_name"]] = p

    player = name_to_player.get(req.player_name)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player '{req.player_name}' not found in pool")

    def resolve_names(names: list[str]) -> list[dict]:
        return [name_to_player[n] for n in names if n in name_to_player]

    my_roster_players = resolve_names(req.my_roster)

    # Collect all sold player csv_names
    all_rostered: set[str] = set()
    for n in req.my_roster:
        p = name_to_player.get(n)
        if p:
            all_rostered.add(p["player_name"])
    for opp in req.opponents:
        for n in opp.get("roster", []):
            p = name_to_player.get(n)
            if p:
                all_rostered.add(p["player_name"])

    remaining = [
        p for p in pool
        if p["player_name"] not in all_rostered
        and p["player_name"] != player["player_name"]
    ]

    pool_by_role: dict[str, list] = {"BAT": [], "BOWL": [], "AR": [], "WK": []}
    for p in remaining:
        role = p.get("role", "BAT")
        if role in pool_by_role:
            pool_by_role[role].append(p)
    for role in pool_by_role:
        pool_by_role[role].sort(key=lambda x: x["projected_points"], reverse=True)

    # Build opposing agent states
    agent_states: dict[str, dict] = {}
    for opp in req.opponents:
        opp_players = resolve_names(opp.get("roster", []))
        agent_states[opp["name"]] = {
            "purse":         float(opp["purse"]),
            "slots":         max(0, 16 - len(opp_players)),
            "mandatory":     max(0, 13 - len(opp_players)),
            "name":          opp["name"],
            "roster_players": opp_players,
        }

    my_slots    = max(0, 16 - len(my_roster_players))
    my_mandatory = max(0, 13 - len(my_roster_players))
    my_avail    = max(0.0, req.my_purse - 1.0 * max(0, my_mandatory - 1))
    opp_avail   = sum(
        max(0.0, float(opp["purse"]) - 1.0 * max(0, agent_states[opp["name"]]["mandatory"] - 1))
        for opp in req.opponents if opp["name"] in agent_states
    )
    rem_vorp = sum(p.get("vorp", 0.0) for p in remaining)

    # Include my own entry in agent_states so PTA helpers (_opp_wtp_max etc.) work
    agent_states["__me__"] = {
        "purse":         req.my_purse,
        "slots":         my_slots,
        "mandatory":     my_mandatory,
        "name":          "__me__",
        "roster_players": my_roster_players,
    }

    state = {
        "players_remaining": len(remaining),
        "cr_per_vorp":       (my_avail + opp_avail) / rem_vorp if rem_vorp > 0 else 0.0,
        "pool_by_role":      pool_by_role,
        "agent_states":      agent_states,
    }

    if req.strategy not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{req.strategy}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )

    cls, _ = STRATEGY_REGISTRY[req.strategy]
    params  = load_params(req.strategy, evolved=req.evolved)
    agent   = cls(name="__me__", params=params)

    # If a tourney_id is supplied, replace the fresh MarketAnalyzer with the
    # session one that has real sale history from /api/advice/sold events.
    if req.tourney_id:
        agent._market = _get_session(req.tourney_id)

    # Manually inject auction state onto the agent (override what reset() set)
    agent.purse      = req.my_purse
    agent.roster     = [{"player": p, "price": 0.0} for p in my_roster_players]
    agent.role_counts = {"BAT": 0, "BOWL": 0, "AR": 0, "WK": 0}
    for p in my_roster_players:
        role = p.get("role", "BAT")
        if role in agent.role_counts:
            agent.role_counts[role] += 1

    bid = round(float(agent.bid(player, state, rnd=1)), 2)

    # ── Decision log — shows exactly why the strategy returned this ceiling ──
    priority = round(float(agent._target_priority(player, state)), 3)
    market_mode = agent._market.market_mode
    cash_per_slot = round(agent.cash_per_slot, 2)
    log.info(
        "ADVICE  tourney=%-20s  player=%-22s  strategy=%-26s  "
        "bid=%5.2f Cr  priority=%.2f  market=%-7s  "
        "purse=%6.1f  slots=%d  roster_size=%d  opponents=%d",
        req.tourney_id or "(none)",
        player.get("auction_name", player["player_name"]),
        req.strategy,
        bid, priority, market_mode,
        req.my_purse, my_slots, len(my_roster_players), len(req.opponents),
    )

    return {
        "strategy":        req.strategy,
        "player_name":     player["player_name"],
        "auction_name":    player.get("auction_name", player["player_name"]),
        "recommended_bid": bid,
        "my_slots":        my_slots,
        "my_max_bid":      round(float(my_avail), 2),
    }


# ── Sale events (from Spring Boot bot integration) ───────────────────────────
@app.post("/api/advice/sold")
def post_sold(event: SoldEvent):
    """
    Called by AuctionServiceImpl.closeAuction() after every hammer.
    Feeds the session MarketAnalyzer so market_mode reflects real price history.

    Spring Boot sends:
      player_name: AuctionDto.playerName
      price:       auction.getBidAmount() / 10_000_000.0
      buyer:       auction.getCurrentWinningTeamName()
      tourney_id:  auction.getFantasyTourneyId()
    """
    pool = _get_pool()
    name_to_player: dict[str, dict] = {}
    for p in pool:
        name_to_player[p["player_name"]] = p
        if p.get("auction_name") and p["auction_name"] != p["player_name"]:
            name_to_player[p["auction_name"]] = p

    player = name_to_player.get(event.player_name)
    if not player:
        # Unknown player (e.g. a booster lot) — don't fail, just skip
        return {"ok": True, "known": False, "market_mode": None}

    market = _get_session(event.tourney_id)

    # Derive cr_per_vorp: how much each VORP point is worth at this sale price.
    # Falls back to 1.0 if vorp is missing to avoid division by zero.
    vorp = player.get("vorp") or 1.0
    cr_per_vorp = event.price / vorp

    market.record_sale(player, event.price, event.buyer, cr_per_vorp)

    log.info(
        "SOLD    tourney=%-20s  player=%-22s  price=%5.2f Cr  buyer=%-20s  "
        "market_mode=%-7s  sales_so_far=%d",
        event.tourney_id,
        player.get("auction_name", player["player_name"]),
        event.price,
        event.buyer,
        market.market_mode,
        len(market._sales),
    )

    return {
        "ok":          True,
        "known":       True,
        "player_name": player["player_name"],
        "auction_name":player.get("auction_name", player["player_name"]),
        "price":       event.price,
        "market_mode": market.market_mode,
        "sales_count": len(market._sales),
    }


@app.delete("/api/session/{tourney_id}")
def delete_session(tourney_id: str):
    """
    Clear the MarketAnalyzer session for a tourney.
    Call this when the auction ends to free memory.
    """
    existed = tourney_id in _sessions
    _sessions.pop(tourney_id, None)
    return {"ok": True, "existed": existed}


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
