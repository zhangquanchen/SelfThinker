#!/usr/bin/env python3
"""Synthesize the unseen multi-object Visual Search pool (1,500 train + 500 diagnostic items)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_selfthinker import ROOT

COLORS = {
    "red": (200, 40, 40),
    "blue": (40, 90, 200),
    "green": (40, 160, 70),
    "yellow": (230, 200, 40),
    "purple": (150, 60, 180),
    "orange": (230, 130, 40),
    "cyan": (40, 180, 200),
    "brown": (130, 80, 40),
}
SHAPES = ["circle", "square", "triangle", "star", "diamond"]

Q_COLOR = "What color is the {shape} in this image? Answer with a single color word."
Q_SHAPE = "What shape is drawn in {color}? Answer with a single shape word (circle, square, triangle, star, or diamond)."


def pil_to_blob(im: Image.Image) -> dict:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def draw_star(d, cx, cy, r, fill):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.4
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(pts, fill=fill)


def draw_shape(d, shape, box, fill):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    elif shape == "triangle":
        d.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=fill)
    elif shape == "diamond":
        d.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=fill)
    else:
        draw_star(d, cx, cy, min(rx, ry), fill)


def place_nonoverlap(rng, n, size, margin, box_s):
    boxes = []
    for _ in range(400):
        if len(boxes) >= n:
            break
        x = rng.randint(margin, size - margin - box_s)
        y = rng.randint(margin, size - margin - box_s)
        cand = (x, y, x + box_s, y + box_s)
        if all(cand[2] < b[0] or cand[0] > b[2] or cand[3] < b[1] or cand[1] > b[3] for b in boxes):
            boxes.append(cand)
    return boxes


def render(rng: random.Random, size=384):
    n = rng.randint(5, 8)
    box_s = rng.randint(42, 70)
    boxes = place_nonoverlap(rng, n, size, 16, box_s)
    if len(boxes) < 4:
        return None
    color_names = list(COLORS)
    rng.shuffle(color_names)
    shapes = list(SHAPES)
    rng.shuffle(shapes)
    items = [
        {"shape": shapes[i % len(shapes)], "color": color_names[i % len(color_names)], "box": box}
        for i, box in enumerate(boxes)
    ]
    img = Image.new("RGB", (size, size), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size - 1, size - 1], outline=(200, 200, 195), width=2)
    for it in items:
        draw_shape(d, it["shape"], it["box"], COLORS[it["color"]])
    return img, items


def add_bbox(im: Image.Image, box, pad=6) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out)
    x0, y0, x1, y1 = box
    d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(220, 20, 20), width=6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/fresh_pool/Visual_Search")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-diag", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    want = args.n_train + args.n_diag
    rows = []
    seen = set()
    while len(rows) < want:
        made = render(rng)
        if made is None:
            continue
        img, items = made
        rng.shuffle(items)
        target = None
        for it in items:
            same_s = sum(1 for x in items if x["shape"] == it["shape"])
            same_c = sum(1 for x in items if x["color"] == it["color"])
            if same_s == 1 and same_c == 1:
                target = it
                break
        if target is None:
            continue
        key = hashlib.sha1(img.tobytes() + f"{target['shape']}{target['color']}".encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        if rng.random() < 0.5:
            q = Q_COLOR.format(shape=target["shape"])
            ans = target["color"]
        else:
            q = Q_SHAPE.format(color=target["color"])
            ans = target["shape"]
        vt = add_bbox(img, target["box"])
        split = "diag" if len(rows) >= args.n_train else "train"
        pid = f"fresh_vs_{args.seed}_{len(rows):05d}"
        rows.append(
            {
                "pid": pid,
                "question": q,
                "answer": ans,
                "problem_image_0": pil_to_blob(img),
                "resoning_thought_0": "Highlight the queried object.",
                "reasoning_image_0": pil_to_blob(vt),
                "resoning_thought_1": "",
                "task": "Visual_Search",
                "full_text_only_thought": ans,
                "split": split,
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
    meta = {"n_train": len(train), "n_diag": len(diag), "seed": args.seed}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
