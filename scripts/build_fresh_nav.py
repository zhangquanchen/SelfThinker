#!/usr/bin/env python3
"""Synthesize the unseen Frozen-Lake navigation pool (1,500 train + 500 diagnostic items)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_selfthinker import ROOT

QUESTION = (
    "You are a maze solver. Your goal is to guide a player from the start "
    "to the goal on a grid map while avoiding holes. The player can move "
    "one square at a time in the directions left (L), right (R), up (U), "
    "or down (D). The frozen lake is not slippery; the player will always "
    "move in the intended direction. Moving off the edge or falling into a "
    "hole results in failure. Reaching the goal means success.\n\n"
    "Provide your solution as a sequence of moves wrapped in \\boxed{{}}, "
    "such as \\boxed{L,R,U,D}. The moves should be comma-separated."
)

MOVE = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}
MOVE_FROM_DELTA = {v: k for k, v in MOVE.items()}


def bfs(grid):
    n = len(grid)
    start = goal = None
    for r in range(n):
        for c in range(n):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        return None
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in MOVE.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if (nr, nc) in prev or grid[nr][nc] == "H":
                continue
            prev[(nr, nc)] = (r, c)
            q.append((nr, nc))
    if goal not in prev:
        return None
    cells = []
    cur = goal
    while cur is not None:
        cells.append(cur)
        cur = prev[cur]
    cells.reverse()
    moves = [MOVE_FROM_DELTA[(r1 - r0, c1 - c0)] for (r0, c0), (r1, c1) in zip(cells, cells[1:])]
    return cells, moves


def make_grid(rng: random.Random, n: int, hole_frac: float, min_len: int, max_len: int):
    for _ in range(200):
        cells = ["F"] * (n * n)
        n_holes = max(1, int(n * n * hole_frac))
        for i in rng.sample(range(n * n), n_holes):
            cells[i] = "H"
        free = [i for i, ch in enumerate(cells) if ch == "F"]
        if len(free) < 2:
            continue
        s, g = rng.sample(free, 2)
        cells[s] = "S"
        cells[g] = "G"
        grid = [cells[i * n : (i + 1) * n] for i in range(n)]
        sol = bfs(grid)
        if sol is None:
            continue
        path, moves = sol
        if min_len <= len(moves) <= max_len:
            return grid, path, moves
    return None


def render(grid, path=None, size=192) -> Image.Image:
    n = len(grid)
    img = Image.new("RGB", (size, size), (230, 242, 250))
    draw = ImageDraw.Draw(img)
    cell = size / n
    pal = {"F": (210, 230, 245), "H": (30, 90, 160), "S": (60, 170, 90), "G": (220, 170, 50)}
    for r in range(n):
        for c in range(n):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell, y0 + cell
            ch = grid[r][c]
            draw.rectangle([x0, y0, x1, y1], fill=pal[ch], outline=(180, 200, 215), width=1)
            if ch == "H":
                m = cell * 0.18
                draw.ellipse([x0 + m, y0 + m, x1 - m, y1 - m], fill=(20, 70, 140))
            elif ch == "S":
                m = cell * 0.28
                draw.ellipse([x0 + m, y0 + m, x1 - m, y1 - m], fill=(40, 140, 70))
            elif ch == "G":
                m = cell * 0.25
                draw.rectangle([x0 + m, y0 + m, x1 - m, y1 - m], fill=(200, 40, 50))
                draw.line(
                    [(x0 + cell / 2, y0 + m), (x0 + cell / 2, y1 - m)],
                    fill=(220, 180, 40),
                    width=max(2, int(cell / 12)),
                )
    if path and len(path) >= 2:
        pts = [(c * cell + cell / 2, r * cell + cell / 2) for r, c in path]
        draw.line(pts, fill=(220, 30, 30), width=max(3, int(cell / 6)))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            draw.line([(x0, y0), (x1, y1)], fill=(220, 30, 30), width=max(3, int(cell / 6)))
            ang = math.atan2(y1 - y0, x1 - x0)
            ah = cell * 0.22
            for da in (2.5, -2.5):
                ax = x1 - ah * math.cos(ang + da)
                ay = y1 - ah * math.sin(ang + da)
                draw.line([(x1, y1), (ax, ay)], fill=(220, 30, 30), width=max(2, int(cell / 8)))
    return img


def pil_to_blob(im: Image.Image) -> dict:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def grid_key(grid) -> str:
    return hashlib.sha1("".join("".join(row) for row in grid).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/fresh_pool/Spatial_Navigation")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-diag", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    want = args.n_train + args.n_diag
    seen = set()
    rows = []
    sizes = [4, 5, 6, 7, 8]
    while len(rows) < want:
        n = rng.choice(sizes)
        hole = rng.uniform(0.12, 0.28)
        min_len = 2 if n <= 4 else 3
        max_len = n * 2
        made = make_grid(rng, n, hole, min_len, max_len)
        if made is None:
            continue
        grid, path, moves = made
        key = grid_key(grid)
        if key in seen:
            continue
        seen.add(key)
        problem = render(grid, path=None, size=192)
        vt = render(grid, path=path, size=192)
        ans = ",".join(moves)
        split = "diag" if len(rows) >= args.n_train else "train"
        pid = f"fresh_nav_{args.seed}_{len(rows):05d}"
        rows.append(
            {
                "pid": pid,
                "question": QUESTION,
                "answer": ans,
                "problem_image_0": pil_to_blob(problem),
                "resoning_thought_0": "Follow the highlighted path.",
                "reasoning_image_0": pil_to_blob(vt),
                "resoning_thought_1": "",
                "task": "Spatial_Navigation",
                "full_text_only_thought": ans,
                "split": split,
                "grid_size": n,
                "path_len": len(moves),
                "grid_key": key,
            }
        )

    def to_table(subset):
        keys = list(subset[0].keys())
        return pa.table({k: [r[k] for r in subset] for k in keys})

    train = [r for r in rows if r["split"] == "train"]
    diag = [r for r in rows if r["split"] == "diag"]
    out_data = args.out / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(train), out_data / "train-00000-of-00001.parquet")
    pq.write_table(to_table(diag), out_data / "diag-00000-of-00001.parquet")
    meta = {"n_train": len(train), "n_diag": len(diag), "seed": args.seed, "sizes": sizes}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
