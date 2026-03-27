"""
server.py — FastAPI advisor server
=====================================
Start with:  python main.py serve  (or uvicorn server:app --reload)

Endpoints:
  GET  /api/pool                  current player pool (with overrides applied)
  GET  /api/strategies            registry + params + MC win rates
  POST /api/override              set manual projected_points for a player
  DELETE /api/override/{name}     remove an override
  GET  /api/playbook              tier-by-tier advice for a strategy
  POST /api/simulate              trigger fresh MC run (blocking, ~10s for 200 runs)
  GET  /api/health                liveness check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.export import load_pool, export_pool, export_insights
from engine.overrides import load_overrides, set_override, remove_override
from engine.strategies import STRATEGY_REGISTRY, load_params
from engine.pool import build_pool, WeightedSeasonModel

_POOL_PATH     = Path("player_pool.json")
_INSIGHTS_PATH = Path("insights.json")
_EVOLVED_PATH  = Path("evolved_params.json")

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
    return load_pool()


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
            "name":           name,
            "default_params": default_p.to_dict(),
            "evolved_params": evolved_p_dict,
            "win_rate":       mc.get("win_rate"),
            "avg_score":      mc.get("avg_score"),
            "std_score":      mc.get("std_score"),
            "wins":           mc.get("wins"),
            "runs":           mc.get("runs"),
            "has_evolved":    evolved_p_dict is not None,
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

    # Rough illustrative bid ceilings at default purse/slots (120/15 = ₹8/slot)
    cash_per_slot = 120.0 / 15

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


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
