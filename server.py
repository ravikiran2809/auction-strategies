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
import sqlite3
from contextlib import contextmanager
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
from engine.intel import load_intel
from engine.overrides import load_overrides, set_override, remove_override
from engine.pta_strategies import register_pta_strategies as _register_pta
from engine.strategies import STRATEGY_REGISTRY, load_params

# Register PTA strategies (SCB, ARM, PersonaCounter) into the shared registry.
# Must run before any route handler that calls STRATEGY_REGISTRY.
_register_pta()
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

# ── SQLite persistence ────────────────────────────────────────────────────────
# advisor.db stores:
#   sale_events   — one row per completed sale; hydrates MarketAnalyzer on restart
#   param_overrides — mid-auction strategy param tweaks per tourney
_DB_PATH = Path("advisor.db")


def _init_db() -> None:
    """Create tables if they don't exist.  Safe to call on every startup."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sale_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                tourney_id   TEXT    NOT NULL,
                player_name  TEXT    NOT NULL,
                price        REAL    NOT NULL,
                buyer        TEXT    NOT NULL,
                vorp         REAL    NOT NULL DEFAULT 1.0,
                cr_per_vorp  REAL    NOT NULL DEFAULT 1.0,
                role         TEXT,
                tier         INTEGER,
                projected_points REAL,
                ts           TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS param_overrides (
                tourney_id     TEXT NOT NULL,
                strategy       TEXT NOT NULL,
                overrides_json TEXT NOT NULL,
                updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (tourney_id, strategy)
            )
        """)
        conn.commit()


_init_db()

# ── Player market intelligence ────────────────────────────────────────────────
# Load player_intel.json if present.  The registry is a module-level singleton
# in engine.intel; strategies call intel_mult() which returns 1.0 when no file
# is loaded, so this is a safe no-op when the file is absent or empty.
_intel = load_intel()
if len(_intel) > 0:
    log.info("Player intel loaded: %d entries from player_intel.json", len(_intel))
else:
    log.info("player_intel.json not found or empty — intel adjustments disabled")


@contextmanager
def _db():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Session-scoped MarketAnalyzers ────────────────────────────────────────────
# In-memory cache: rehydrated from SQLite on first use after a restart.
from engine.market import MarketAnalyzer as _MarketAnalyzer, SaleRecord as _SaleRecord
_sessions: dict[str, _MarketAnalyzer] = {}
_session_cr_per_vorp: dict[str, float] = {}  # last market rate computed for each session
_last_query: dict[str, dict] = {}  # last /api/advice request per tourney_id (for caddy live polling)


def _hydrate_session_from_db(tourney_id: str) -> _MarketAnalyzer:
    """Rebuild a MarketAnalyzer from persisted sale_events rows."""
    ma = _MarketAnalyzer()
    with _db() as conn:
        rows = conn.execute(
            "SELECT player_name, price, buyer, vorp, cr_per_vorp, role, tier, projected_points "
            "FROM sale_events WHERE tourney_id = ? ORDER BY id",
            (tourney_id,),
        ).fetchall()
    for row in rows:
        player = {
            "player_name":       row["player_name"],
            "role":              row["role"] or "BAT",
            "tier":              row["tier"] or 3,
            "projected_points":  row["projected_points"] or 0.0,
            "vorp":              row["vorp"],
        }
        ma.record_sale(player, row["price"], row["buyer"], row["cr_per_vorp"])
    return ma


def _get_session(tourney_id: str) -> _MarketAnalyzer:
    """Return the MarketAnalyzer for a tourney; rehydrate from DB if not in cache."""
    if tourney_id not in _sessions:
        _sessions[tourney_id] = _hydrate_session_from_db(tourney_id)
    return _sessions[tourney_id]


def _clear_session(tourney_id: str) -> bool:
    existed = tourney_id in _sessions
    _sessions.pop(tourney_id, None)
    _session_cr_per_vorp.pop(tourney_id, None)  # also clear the stored rate
    return existed

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
    return FileResponse(_HTML_DIR / "caddy.html")


@app.get("/caddy", include_in_schema=False)
def caddy():
    """Lightweight live caddy for the human player — polls /api/session/{id}/live."""
    return FileResponse(_HTML_DIR / "caddy.html")


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
    current_bid: current bid on the floor (₹ Cr) — used to compute remaining headroom.
                 If omitted, defaults to 0. The recommended_bid in the response is
                 always the strategy's ceiling; callers should bid recommended_bid
                 only if current_bid < recommended_bid.
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
    current_bid: float = 0.0
    my_purse: float = 120.0
    my_roster: list[str] = []
    opponents: list[dict] = []
    strategy: str = "SquadCompletionBidder"
    evolved: bool = True
    tourney_id: Optional[str] = None
    field_size: int = 8   # total number of teams in the auction (used to calibrate cr_per_vorp)


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
            # Use intel-adjusted projected_points so role_cliff / role_depth
            # correctly treat injured/unavailable players as low-value.
            # We shallow-copy the player dict to avoid mutating the shared pool.
            from engine.intel import intel_mult as _imult
            im = _imult(p["player_name"])
            if im < 1.0:
                p = {**p, "projected_points": p["projected_points"] * im}
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

    # Scale cr_per_vorp to the full expected field size.
    # When Spring Boot sends fewer opponents than the actual field (or opponents=[]),
    # the naive total available cash underrepresents market liquidity, causing sale prices
    # to look like extreme "overbid" and breaking market_mode signals.  Extrapolate to
    # req.field_size (default 8) using the known teams' per-team average.
    n_teams_known = len(req.opponents) + 1  # +1 for our team
    if n_teams_known < req.field_size and n_teams_known > 0:
        avg_avail_per_team = (my_avail + opp_avail) / n_teams_known
        scaled_avail = avg_avail_per_team * req.field_size
    else:
        scaled_avail = my_avail + opp_avail

    state = {
        "players_remaining": len(remaining),
        "cr_per_vorp":       scaled_avail / rem_vorp if rem_vorp > 0 else 0.0,
        "pool_by_role":      pool_by_role,
        "agent_states":      agent_states,
    }

    # Store this session's market rate so /api/advice/sold can use it.
    # This fixes market_mode: sold events use the true market cr_per_vorp
    # (total purse / total VORP) rather than back-computing it from the sale price,
    # which would always produce a ratio ≈ 1.0 (always "fair").
    if req.tourney_id:
        _session_cr_per_vorp[req.tourney_id] = state["cr_per_vorp"]

    if req.strategy not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{req.strategy}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )

    cls, _ = STRATEGY_REGISTRY[req.strategy]
    params  = load_params(req.strategy, evolved=req.evolved)

    # Merge any active mid-auction param overrides for this tourney.
    if req.tourney_id:
        active_overrides = _get_active_overrides(req.tourney_id, req.strategy)
        if active_overrides:
            base_dict = params.to_dict()
            base_dict.update(active_overrides)
            _, params_cls = STRATEGY_REGISTRY[req.strategy]
            params = params_cls.from_dict(base_dict)

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

    # Cap at 20% of remaining purse to prevent single-player over-commitment.
    # The evolved multipliers (must_bid × cliff ≈ 2.3×) were trained bot-vs-bot
    # and push elite outliers (Narine, Sai Sudharsan) to ₹30+ Cr — above real IPL
    # auction highs (Starc ₹24.75 Cr, Kohli ₹21 Cr).  20% keeps the ceiling at
    # ₹24 Cr for a ₹120 Cr purse, matching the top of real competitive bidding.
    purse_cap = round(req.my_purse * 0.20, 2)
    bid = min(bid, purse_cap)

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

    resp = {
        "strategy":        req.strategy,
        "player_name":     player["player_name"],
        "auction_name":    player.get("auction_name", player["player_name"]),
        "recommended_bid": bid,
        "current_bid":     round(req.current_bid, 2),
        "should_bid":      bid > req.current_bid,   # True if we'd raise the current bid
        "my_slots":        my_slots,
        "my_max_bid":      round(float(my_avail), 2),
    }

    # Store for caddy live polling — keyed by tourney_id so the human advisor
    # can detect when a new player is nominated and fetch SCB advice automatically.
    if req.tourney_id:
        _last_query[req.tourney_id] = {
            "player_name":     req.player_name,
            "player":          player,
            "current_bid":     round(req.current_bid, 2),
            "my_purse":        req.my_purse,
            "my_roster":       req.my_roster,
            "opponents":       req.opponents,
            "evolved":         req.evolved,
            "field_size":      req.field_size,
            "strategy":        req.strategy,
            "recommended_bid": bid,
            "priority":        priority,
            "market_mode":     market_mode,
        }

    return resp


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

    # Use the market cr_per_vorp stored from the last /api/advice call for this session.
    # This is the correct rate: (total remaining purse) / (total remaining VORP).
    # Falling back to price/vorp is self-referential and makes market_mode always "fair".
    vorp = player.get("vorp") or 1.0
    cr_per_vorp = _session_cr_per_vorp.get(event.tourney_id, event.price / vorp)

    market.record_sale(player, event.price, event.buyer, cr_per_vorp)

    # Persist to SQLite for restart recovery.
    with _db() as conn:
        conn.execute(
            "INSERT INTO sale_events "
            "(tourney_id, player_name, price, buyer, vorp, cr_per_vorp, role, tier, projected_points) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.tourney_id,
                player["player_name"],
                event.price,
                event.buyer,
                vorp,
                cr_per_vorp,
                player.get("role"),
                player.get("tier"),
                player.get("projected_points"),
            ),
        )

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
    Clear the MarketAnalyzer session for a tourney (in-memory + DB rows).
    Call this when the auction ends.
    """
    existed = _clear_session(tourney_id)
    _last_query.pop(tourney_id, None)
    with _db() as conn:
        conn.execute("DELETE FROM sale_events WHERE tourney_id = ?", (tourney_id,))
    return {"ok": True, "existed": existed}


@app.get("/api/sessions")
def list_sessions():
    """Debug endpoint — returns all active tourney IDs and the last player seen for each."""
    return {
        "active_sessions": [
            {"tourney_id": k, "last_player": v.get("player_name"), "strategy": v.get("strategy")}
            for k, v in _last_query.items()
        ]
    }


@app.get("/api/session/{tourney_id}/live")
def get_session_live(tourney_id: str):
    """
    Real-time snapshot for the human caddy advisor (caddy.html).

    Polled every ~2 s by the human's browser. Returns the last player the ARM bot
    was queried about, its recommended bid, and recent sale history so the caddy can
    auto-fetch SCB advice for the same player using the human's own state.

    Returns {has_update: false} when no advice has been requested yet for this tourney.
    """
    query = _last_query.get(tourney_id)
    if not query:
        return {"has_update": False}

    with _db() as conn:
        rows = conn.execute(
            "SELECT player_name, price, buyer, role, tier, projected_points "
            "FROM sale_events WHERE tourney_id = ? ORDER BY id DESC LIMIT 10",
            (tourney_id,),
        ).fetchall()

    recent_sales = [
        {"player": r["player_name"], "price": r["price"], "buyer": r["buyer"],
         "role": r["role"], "tier": r["tier"], "projected_points": r["projected_points"]}
        for r in rows
    ]

    return {
        "has_update":       True,
        "current_player":   query["player_name"],
        "player":           query["player"],
        "current_bid":      query["current_bid"],
        "arm_bid":          query["recommended_bid"],
        "arm_should_bid":   query["recommended_bid"] > query["current_bid"],
        "priority":         query["priority"],
        "market_mode":      query["market_mode"],
        "bot_purse":        query["my_purse"],
        "bot_roster":       query["my_roster"],
        "opponents":        query["opponents"],
        "evolved":          query["evolved"],
        "field_size":       query["field_size"],
        "recent_sales":     recent_sales,
    }


# ── Mid-auction param overrides ───────────────────────────────────────────────
class ParamOverrideRequest(BaseModel):
    """
    Set temporary param overrides for a strategy within a specific tourney.
    Overrides are shallow-merged onto loaded params before strategy instantiation.
    Unknown field names are rejected to prevent silent misconfigurations.

    Example:
        POST /api/params/override
        {
          "tourney_id": "GroupA_IPL2026",
          "strategy":   "SquadCompletionBidder",
          "overrides":  { "must_bid_mult": 2.5 }
        }
    """
    tourney_id: str
    strategy: str
    overrides: dict


def _get_active_overrides(tourney_id: str, strategy: str) -> dict:
    """Load active param overrides for a tourney+strategy from SQLite."""
    with _db() as conn:
        row = conn.execute(
            "SELECT overrides_json FROM param_overrides WHERE tourney_id = ? AND strategy = ?",
            (tourney_id, strategy),
        ).fetchone()
    return json.loads(row["overrides_json"]) if row else {}


@app.post("/api/params/override")
def post_param_override(req: ParamOverrideRequest):
    """Set temporary param overrides for a strategy within a tourney."""
    if req.strategy not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{req.strategy}'")

    # Validate field names against the strategy's param dataclass
    _, params_cls = STRATEGY_REGISTRY[req.strategy]
    default_params = load_params(req.strategy)
    valid_fields = set(default_params.to_dict().keys())
    invalid = [k for k in req.overrides if k not in valid_fields]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown param fields for {req.strategy}: {invalid}. "
                   f"Valid: {sorted(valid_fields)}",
        )

    with _db() as conn:
        conn.execute(
            "INSERT INTO param_overrides (tourney_id, strategy, overrides_json, updated_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(tourney_id, strategy) DO UPDATE SET "
            "overrides_json = excluded.overrides_json, updated_at = excluded.updated_at",
            (req.tourney_id, req.strategy, json.dumps(req.overrides)),
        )

    log.info(
        "PARAM_OVERRIDE  tourney=%-20s  strategy=%-26s  overrides=%s",
        req.tourney_id, req.strategy, req.overrides,
    )
    return {"ok": True, "tourney_id": req.tourney_id, "strategy": req.strategy,
            "overrides": req.overrides}


@app.get("/api/params/override/{tourney_id}")
def get_param_overrides(tourney_id: str):
    """Return all active param overrides for a tourney."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT strategy, overrides_json, updated_at FROM param_overrides "
            "WHERE tourney_id = ?",
            (tourney_id,),
        ).fetchall()
    result = {
        row["strategy"]: {
            "overrides":   json.loads(row["overrides_json"]),
            "updated_at":  row["updated_at"],
        }
        for row in rows
    }
    return {"tourney_id": tourney_id, "overrides": result}


@app.delete("/api/params/override/{tourney_id}")
def delete_param_overrides(tourney_id: str, strategy: Optional[str] = None):
    """Clear param overrides for a tourney (all strategies or a specific one)."""
    with _db() as conn:
        if strategy:
            conn.execute(
                "DELETE FROM param_overrides WHERE tourney_id = ? AND strategy = ?",
                (tourney_id, strategy),
            )
        else:
            conn.execute(
                "DELETE FROM param_overrides WHERE tourney_id = ?", (tourney_id,)
            )
    return {"ok": True, "tourney_id": tourney_id, "strategy": strategy}


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
