"""
engine/overrides.py
====================
Manual player value overrides — applied on top of any ProjectionModel.
Persists to overrides.json in the workspace root.

Schema (overrides.json):
  {
    "Virat Kohli": {"projected_points": 130, "note": "gut feeling"},
    "JJ Bumrah":   {"projected_points": 125}
  }
"""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent / "overrides.json"


def load_overrides(path: str | Path | None = None) -> dict[str, dict]:
    """Load overrides from JSON. Returns empty dict if file doesn't exist."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def save_overrides(overrides: dict[str, dict], path: str | Path | None = None) -> None:
    """Persist overrides to JSON, creating the file if needed."""
    p = Path(path) if path else _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(overrides, f, indent=2)


def set_override(
    player_name: str,
    projected_points: float,
    note: str = "",
    path: str | Path | None = None,
) -> dict[str, dict]:
    """Set or update one player's override and save. Returns the full overrides dict."""
    overrides = load_overrides(path)
    entry: dict = {"projected_points": projected_points}
    if note:
        entry["note"] = note
    overrides[player_name] = entry
    save_overrides(overrides, path)
    return overrides


def remove_override(
    player_name: str,
    path: str | Path | None = None,
) -> dict[str, dict]:
    """Remove a player's override and save. No-op if player not in overrides."""
    overrides = load_overrides(path)
    overrides.pop(player_name, None)
    save_overrides(overrides, path)
    return overrides


def apply_overrides(
    pool: list[dict],
    path: str | Path | None = None,
    overrides: dict[str, dict] | None = None,
) -> list[dict]:
    """
    Apply manual projected_points overrides to the pool list.
    Overrides win over any model-computed value.
    Also recalculates tier based on the new projected_points.
    """
    if overrides is None:
        overrides = load_overrides(path)
    if not overrides:
        return pool

    result = []
    for player in pool:
        name = player["player_name"]
        if name in overrides:
            player = dict(player)  # don't mutate the original
            player["projected_points"] = overrides[name]["projected_points"]
            player["override_note"] = overrides[name].get("note", "")
            # Recalculate tier
            pts = player["projected_points"]
            player["tier"] = 1 if pts > 80 else (2 if pts > 58 else 3)
        result.append(player)
    return result
