"""
engine/intel.py
===============
Player market intelligence — injury status, availability, and current form.

Design
------
A lightweight opt-in layer that adjusts willingness-to-pay (WTP) based on
information that cannot be derived from historical scoring data:

  - injury / unavailability  → WTP drops to 0 (don't bid)
  - doubtful                 → WTP multiplied by ~0.40 (bid very cautiously)
  - fit, good form           → small positive nudge (≤ +10%)
  - fit, poor form           → small negative nudge (≤ -10%)

The module exposes a **module-level singleton registry** so callers don't need
to pass an object around:

    from engine.intel import load_intel, intel_mult

    load_intel("player_intel.json")   # once, on startup

    wtp *= intel_mult(player["player_name"])   # in any strategy WTP method

When the registry has **not** been loaded, ``intel_mult()`` returns 1.0 — so
simulation runs are completely unaffected unless intel is explicitly loaded.

Edit ``player_intel.json`` before each auction session to reflect the latest
news. Players not listed are assumed fit with neutral form (multiplier = 1.0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────

InjuryStatus = Literal["fit", "doubtful", "injured", "unavailable"]

_INJURY_MULT: dict[str, float] = {
    "fit": 1.0,
    "doubtful": 0.40,     # significant uncertainty — bid at most 40% of normal WTP
    "injured": 0.0,       # confirmed injury — don't bid
    "unavailable": 0.0,   # out for any other reason (trade, national duty, etc.)
}

# Form effect is deliberately small so that only injury status causes dramatic
# changes.  A form_rating of 1.5 (excellent) still only adds +10% to WTP.
_FORM_EFFECT = 0.10


# ── Core dataclass ────────────────────────────────────────────────────────────

@dataclass
class PlayerIntel:
    """
    Intel record for a single player.

    Fields
    ------
    injury_status : "fit" | "doubtful" | "injured" | "unavailable"
    form_rating   : float in [0.5, 1.5]; 1.0 = neutral (default)
    availability_note : free-text reason (shown in advisor UI, not used in math)
    last_updated  : YYYY-MM-DD string for staleness tracking
    """
    injury_status: InjuryStatus = "fit"
    availability_note: str = ""
    form_rating: float = 1.0
    last_updated: str = ""

    def utility_mult(self) -> float:
        """
        WTP multiplier combining injury status and form:

          injured / unavailable  →  0.0
          doubtful               →  0.36 – 0.44  (form-adjusted)
          fit                    →  0.90 – 1.10  (form-adjusted)
        """
        base = _INJURY_MULT.get(self.injury_status, 1.0)
        if base == 0.0:
            return 0.0
        if self.injury_status == "doubtful":
            # form_rating encodes player quality tier for doubtful players.
            # Maps form_rating [0.5, 1.5] → WTP multiplier ≈ [0.20, 0.65]
            #   elite   (fr=1.4): ~0.60 — worth bidding cautiously
            #   average (fr=1.0): ~0.43 — bid only if gap is large
            #   mediocre(fr=0.6): ~0.25 — barely worth the risk
            quality = max(0.0, min(1.0, self.form_rating - 0.5))
            return round(0.20 + 0.45 * quality, 3)
        # fit: small ±10% form adjustment
        form_adj = 1.0 + _FORM_EFFECT * max(-1.0, min(1.0, self.form_rating - 1.0))
        return base * form_adj


# ── Registry ──────────────────────────────────────────────────────────────────

class IntelRegistry:
    """
    Holds per-player intel records loaded from player_intel.json.

    Unknown players (not in the file) default to fit + neutral form → mult 1.0.
    Keys starting with ``_`` are treated as comments/schema docs and skipped.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._data: dict[str, PlayerIntel] = {}
        if path and path.exists():
            raw: dict = json.loads(path.read_text(encoding="utf-8"))
            _fields = set(PlayerIntel.__dataclass_fields__)
            for name, rec in raw.items():
                if name.startswith("_"):
                    continue
                if not isinstance(rec, dict):
                    continue
                try:
                    self._data[name] = PlayerIntel(
                        **{k: v for k, v in rec.items() if k in _fields}
                    )
                except Exception:
                    pass  # malformed entry — silently skip

    def get(self, player_name: str) -> PlayerIntel:
        """Return intel for player, defaulting to fit+neutral if not found."""
        return self._data.get(player_name, PlayerIntel())

    def utility_mult(self, player_name: str) -> float:
        return self.get(player_name).utility_mult()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"IntelRegistry({len(self._data)} players)"


# ── Module-level singleton ─────────────────────────────────────────────────────

_registry: IntelRegistry | None = None


def load_intel(path: str | Path | None = None) -> IntelRegistry:
    """
    Load player intel from *path* into the module-level singleton.

    Default path: ``player_intel.json`` in the current working directory.
    Call this once on application startup (server, CLI run) before any
    strategies are instantiated.

    Returns the registry so callers can inspect it if needed.
    """
    global _registry
    resolved = Path(path) if path else Path("player_intel.json")
    _registry = IntelRegistry(resolved)
    return _registry


def get_registry() -> IntelRegistry | None:
    """Return the active registry, or None if not yet loaded."""
    return _registry


def intel_mult(player_name: str) -> float:
    """
    Return the WTP multiplier for *player_name*.

    Returns 1.0 (no effect) when the registry has not been loaded — so
    simulation runs are completely unaffected by default.

    Typical values:
      1.0       — fit, neutral form (or player not in intel file)
      0.90–1.10 — fit, form-adjusted
      ~0.40     — doubtful
      0.0       — injured or unavailable
    """
    if _registry is None:
        return 1.0
    return _registry.utility_mult(player_name)
