#!/usr/bin/env python3
"""Import a vacuum robot map image into the Wheelson house map.

Reads a rendered vacuum map (colored room fills on a black background),
segments it by room color, splits same-color regions into rooms via
connected components, and writes middleware/map/house_map.json:

  {
    "version": 1,
    "source": ...,
    "grid": {"cols": N, "rows": M, "cell_meters": null, "cells": ["0102..."]},
    "rooms": [{"id": 1, "name": "Kitchen", "color": [r, g, b],
               "centroid": [col, row], "cells": count}],
    "adjacency": [{"a": 1, "b": 2, "shared_border": n}],
    "pose": null
  }

Grid codes: 0 = blocked/unknown, 1..7 = room ids. Rows are strings so a
single digit per cell suffices (ids stay below 10).

Scale is NOT guessed: cell_meters stays null until one real-world
measurement (a wall or doorway) is supplied, then it is set by hand.

The source image itself is never committed to the repo: it is a picture
of the user's home. Only the derived grid ships.

Usage:
    python import_vacuum_map.py --input /tmp/vacuum-map.jpg \
        --output house_map.json --preview /tmp/house_map_preview.png
"""

import argparse
import colorsys
import json
import sys
from collections import deque

# Room-fill prototypes sampled from the vacuum image (median block color).
# Kitchen and Leo's Room share yellow; Dining Area and Simon's Room share
# blue; Living Room and Room share orange; Main Bath is teal.
PROTOTYPES = {
    "yellow": (229, 190, 89),
    "blue": (115, 170, 234),
    "teal": (37, 208, 199),
    "orange": (218, 132, 107),
}

# Room ids and the positional rule that labels each connected component.
# Verified against the rendered preview before the map is accepted.
ROOM_IDS = {
    "Kitchen": 1,
    "Leo's Room": 2,
    "Dining Area": 3,
    "Main Bath": 4,
    "Simon's Room": 5,
    "Living Room": 6,
    "Room": 7,
}


def main():
    ap = argparse.ArgumentParser(description="Import a vacuum map image.")
    ap.add_argument("--input", required=True, help="Vacuum map image (JPG/PNG).")
    ap.add_argument("--output", required=True, help="house_map.json to write.")
    ap.add_argument("--preview", default=None, help="Preview PNG to render.")
    ap.add_argument("--cols", type=int, default=200, help="Grid columns.")
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    im = Image.open(args.input).convert("RGB")
    w, h = im.size
    px = im.load()

    # Crop to the house: bounding box of non-near-black pixels.
    min_x, min_y, max_x, max_y = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if r > 40 or g > 40 or b > 40:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    if max_x < 0:
        print("no non-black content found", file=sys.stderr)
        return 1
    m = 4
    min_x = max(0, min_x - m)
    min_y = max(0, min_y - m)
    max_x = min(w - 1, max_x + m)
    max_y = min(h - 1, max_y + m)
    crop_w = max_x - min_x + 1
    crop_h = max_y - min_y + 1
    cols = args.cols
    cell = crop_w / cols
    rows = max(1, round(crop_h / cell))
    print(f"source {w}x{h}, crop {crop_w}x{crop_h} at ({min_x},{min_y}), "
          f"grid {cols}x{rows}, cell ~{cell:.1f}px")

    # Classify each cell by its block mean color, in HUE space.
    # Whitening (vacuum paths, text) preserves hue but shifts RGB distance,
    # so RGB distance mislabels whitened orange as yellow. Hue survives it.
    # Codes: 0 blocked, -1 unknown, else a color key.
    proto_hues = {}
    for name, (pr, pg, pb) in PROTOTYPES.items():
        proto_hues[name] = colorsys.rgb_to_hsv(pr / 255, pg / 255,
                                               pb / 255)[0] * 360
    grid = []
    for r in range(rows):
        row = []
        y0 = min_y + int(r * cell)
        y1 = min_y + min(crop_h, int((r + 1) * cell))
        for c in range(cols):
            x0 = min_x + int(c * cell)
            x1 = min_x + min(crop_w, int((c + 1) * cell))
            sr = sg = sb = n = 0
            for y in range(y0, max(y0 + 1, y1)):
                for x in range(x0, max(x0 + 1, x1)):
                    pr, pg, pb = px[x, y]
                    sr += pr
                    sg += pg
                    sb += pb
                    n += 1
            mean = (sr / n, sg / n, sb / n)
            if max(mean) < 70:
                row.append(0)
                continue
            h, s, _v = colorsys.rgb_to_hsv(mean[0] / 255, mean[1] / 255,
                                           mean[2] / 255)
            if s < 0.15:
                row.append(-1)  # near-white/gray: path, text, icon
                continue
            hue = h * 360
            best, best_d = None, 1e9
            for name, ph in proto_hues.items():
                d = min(abs(hue - ph), 360 - abs(hue - ph))
                if d < best_d:
                    best, best_d = name, d
            row.append(best if best_d <= 25 else -1)
        grid.append(row)

    # Hole-fill unknowns surrounded by one room color (paths, text, icons).
    for _ in range(3):
        changed = False
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != -1:
                    continue
                votes = {}
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        v = grid[nr][nc]
                        if v not in (0, -1):
                            votes[v] = votes.get(v, 0) + 1
                if votes:
                    top, top_n = max(votes.items(), key=lambda kv: kv[1])
                    if top_n >= 3:
                        grid[r][c] = top
                        changed = True
        if not changed:
            break
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == -1:
                grid[r][c] = 0

    # Connected components per color (4-connectivity). Small specks drop out.
    comp_of = [[-1] * cols for _ in range(rows)]
    components = []  # (color, [(r, c), ...])
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color == 0 or comp_of[r][c] != -1:
                continue
            cells = []
            dq = deque([(r, c)])
            comp_of[r][c] = len(components)
            while dq:
                cr, cc = dq.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and grid[nr][nc] == color
                            and comp_of[nr][nc] == -1):
                        comp_of[nr][nc] = len(components)
                        dq.append((nr, nc))
            components.append((color, cells))

    # Drop specks below ~1 m^2 (real rooms are thousands of cells; a
    # 71-cell blue tip at the map edge is not a room).
    kept = []
    for color, cells in components:
        if len(cells) < 150:
            for (r, c) in cells:
                grid[r][c] = 0
            continue
        kept.append((color, cells))
    print(f"components kept: {len(kept)} "
          f"(dropped {len(components) - len(kept)} specks)")

    by_color = {}
    for color, cells in kept:
        by_color.setdefault(color, []).append(cells)

    # Label components by position. Abort on an unexpected shape so a
    # mislabeled map never ships.
    def expect(color, n, label):
        got = by_color.get(color, [])
        if len(got) != n:
            print(f"ABORT: {color} has {len(got)} components, expected {n}:",
                  file=sys.stderr)
            for cells in got:
                cr = sum(r for r, _ in cells) / len(cells)
                cc = sum(c for _, c in cells) / len(cells)
                print(f"  {label}: {len(cells)} cells at "
                      f"(col {cc:.0f}, row {cr:.0f})", file=sys.stderr)
            sys.exit(2)
        return got

    yellow = expect("yellow", 2, "yellow")
    blue = expect("blue", 2, "blue")
    teal = expect("teal", 1, "teal")
    orange = expect("orange", 2, "orange")

    yellow.sort(key=lambda cells: sum(r for r, _ in cells) / len(cells))
    blue.sort(key=lambda cells: sum(c for _, c in cells) / len(cells))
    orange.sort(key=lambda cells: sum(r for r, _ in cells) / len(cells))
    # Yellow top-to-bottom: Kitchen, Leo's Room.
    # Blue left-to-right: Simon's Room, Dining Area.
    # Orange top-to-bottom: Room, Living Room.
    labeled = [
        ("Kitchen", yellow[0]),
        ("Leo's Room", yellow[1]),
        ("Simon's Room", blue[0]),
        ("Dining Area", blue[1]),
        ("Main Bath", teal[0]),
        ("Room", orange[0]),
        ("Living Room", orange[1]),
    ]

    final = [[0] * cols for _ in range(rows)]
    rooms = []
    for name, cells in labeled:
        rid = ROOM_IDS[name]
        for (r, c) in cells:
            final[r][c] = rid
        cr = sum(r for r, _ in cells) / len(cells)
        cc = sum(c for _, c in cells) / len(cells)
        # Color for display: the prototype of this room's family.
        family = {"Kitchen": "yellow", "Leo's Room": "yellow",
                  "Simon's Room": "blue", "Dining Area": "blue",
                  "Main Bath": "teal", "Room": "orange",
                  "Living Room": "orange"}[name]
        rooms.append({"id": rid, "name": name,
                      "color": list(PROTOTYPES[family]),
                      "centroid": [round(cc, 1), round(cr, 1)],
                      "cells": len(cells)})
        print(f"room {rid} {name}: {len(cells)} cells, "
              f"centroid (col {cc:.0f}, row {cr:.0f})")

    adjacency = []
    ids = sorted(ROOM_IDS.values())
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = 0
            for r in range(rows):
                for c in range(cols):
                    if final[r][c] != a:
                        continue
                    for dr, dc in ((1, 0), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if (nr < rows and nc < cols
                                and final[nr][nc] == b):
                            shared += 1
            if shared:
                adjacency.append({"a": a, "b": b, "shared_border": shared})
    print(f"adjacencies: {len(adjacency)}")

    data = {
        "version": 1,
        "source": "vacuum robot map image, supplied 2026-11-01",
        "grid": {
            "cols": cols,
            "rows": rows,
            "cell_meters": None,
            "cells": ["".join(str(v) for v in row) for row in final],
        },
        "rooms": sorted(rooms, key=lambda r: r["id"]),
        "adjacency": adjacency,
        "pose": None,
        "notes": [
            "cell_meters is null until one real-world measurement is supplied.",
            "Doorway coordinates are not extracted yet; adjacency is a hint.",
            "Black cells are blocked or unmapped; the planner must not route through them.",
        ],
    }
    with open(args.output, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {args.output}")

    if args.preview:
        scale = 3
        prev = Image.new("RGB", (cols * scale, rows * scale), (0, 0, 0))
        draw = ImageDraw.Draw(prev)
        id_to_color = {r["id"]: tuple(r["color"]) for r in rooms}
        for r in range(rows):
            for c in range(cols):
                v = final[r][c]
                if v:
                    draw.rectangle([c * scale, r * scale,
                                    (c + 1) * scale - 1, (r + 1) * scale - 1],
                                   fill=id_to_color[v])
        for room in rooms:
            cc, cr = room["centroid"]
            draw.text((cc * scale + 2, cr * scale + 2), room["name"],
                      fill=(0, 0, 0))
        prev.save(args.preview)
        print(f"wrote preview {args.preview}")


if __name__ == "__main__":
    main()
