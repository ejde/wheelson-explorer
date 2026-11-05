"""House map loading and validation (Phase 1: vacuum map import).

The map artifact lives at middleware/map/house_map.json, produced by
map/import_vacuum_map.py from the vacuum robot's rendered map. The grid
is row-major strings of single-digit codes: 0 = blocked or unmapped,
1..7 = room ids (see the rooms list). The pose is null until Phase 2
localization exists; it is never guessed.
"""

import json
import os

MAP_PATH = os.path.join(os.path.dirname(__file__), "map", "house_map.json")


def load_map(path=MAP_PATH):
    """Load and validate the house map. Raises on missing/invalid data."""
    with open(path) as f:
        data = json.load(f)
    grid = data["grid"]
    cols, rows = grid["cols"], grid["rows"]
    cells = grid["cells"]
    if len(cells) != rows:
        raise ValueError(f"grid has {len(cells)} rows, expected {rows}")
    valid = {r["id"] for r in data["rooms"]} | {0}
    for i, row in enumerate(cells):
        if len(row) != cols:
            raise ValueError(f"grid row {i} has {len(row)} cols, "
                             f"expected {cols}")
        for ch in row:
            if not ch.isdigit() or int(ch) not in valid:
                raise ValueError(f"grid row {i} has invalid code {ch!r}")
    if data.get("pose") is not None and len(data["pose"]) != 3:
        raise ValueError("pose must be [col, row, heading_deg]")
    return data


def map_data():
    """Map payload for the /map endpoint; an error dict if unavailable."""
    try:
        return load_map()
    except FileNotFoundError:
        return {"error": "house map not found"}
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return {"error": f"house map invalid: {exc}"}
