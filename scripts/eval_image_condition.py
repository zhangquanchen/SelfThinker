#!/usr/bin/env python3
"""Image-condition evaluation.

The real image is compared with a swapped, blank, scrambled, mosaic, blurred,
or tile-permuted image. Final answers are extracted from <answer>...</answer>
or post-</think> output when present, falling back to the reasoning trace when
generation terminates at <image_start>. Numeric answers use the last number
within 5% relative error; multiple-choice answers use the last option letter.
"""
from __future__ import annotations

import base64
import csv
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

csv.field_size_limit(sys.maxsize)

ROOT = (
    Path(os.environ["SELFTHINKER_ROOT"])
    if "SELFTHINKER_ROOT" in os.environ
    else (
        Path(__file__).resolve().parents[1]
        if (
            (Path(__file__).resolve().parents[1] / "models").is_dir()
            or (Path(__file__).resolve().parents[1] / "data").is_dir()
        )
        else Path(__file__).resolve().parents[2]
    )
)
_DEFAULT_MODEL = (
    ROOT / "models"
    if (ROOT / "models" / "model.safetensors").is_file()
    else ROOT / "models/selfthinker"
)
MODEL = Path(os.environ.get("MUCG_INTV_MODEL", str(_DEFAULT_MODEL)))
TSV = Path(os.environ.get("MUCG_CHART_TSV", str(ROOT / "data/lmudata/ChartQA_h_bar.tsv")))


def _num(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _extract_answer_text(pred: str) -> str:
    """Extract the final answer segment, preferring explicit <answer> tags or post-</think> text."""
    ans_blocks = re.findall(r"<answer>(.*?)</answer>", pred, flags=re.DOTALL | re.IGNORECASE)
    for blk in reversed(ans_blocks):
        if blk.strip():
            return blk.strip()
    if "</think>" in pred:
        after = pred.split("</think>", 1)[1].replace("<image_start>", "").strip()
        if after:
            return after
    return pred.split("</think>")[0].split("<image_start>")[0]


def _hit(pred: str, answer: str) -> int:
    body = _extract_answer_text(pred)
    fallback = pred.split("</think>")[0].split("<image_start>")[0]
    gold = str(answer).strip()
    if gold.upper() in {"A", "B", "C", "D"} and len(gold) == 1:
        letters = re.findall(r"\b([ABCD])\b", body.upper())
        if not letters and body != fallback:
            letters = re.findall(r"\b([ABCD])\b", fallback.upper())
        return int(bool(letters) and letters[-1] == gold.upper())
    pred_n = _num(body)
    if pred_n is None and body != fallback:
        pred_n = _num(fallback)
    gold_n = _num(gold)
    if pred_n is None or gold_n is None:
        return int(gold.lower() in body.lower() or gold.lower() in fallback.lower())
    if abs(gold_n) < 1e-8:
        return int(abs(pred_n) <= 1e-2)
    return int(abs(pred_n - gold_n) / abs(gold_n) <= 0.05)


def _img(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _scramble(chart: Image.Image, key: str) -> Image.Image:
    """Same pixels in a random order. Seeded by the item id."""
    arr = np.array(chart)
    flat = arr.reshape(-1, arr.shape[-1])
    seed = np.uint32(0)
    for ch in key:
        seed = np.uint32(seed * np.uint32(131) + np.uint32(ord(ch)))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(flat)
    return Image.fromarray(flat.reshape(arr.shape))


def _tiles(chart: Image.Image, key: str, grid: int = 8) -> Image.Image:
    """Permute a grid of intact patches. Local pixels stay; their places do not."""
    g = max(2, int(grid))
    w, h = chart.size
    nw = max(g, (w // g) * g)
    nh = max(g, (h // g) * g)
    img = chart if (nw == w and nh == h) else chart.resize((nw, nh), Image.NEAREST)
    arr = np.array(img)
    th, tw = nh // g, nw // g
    tiles = (
        arr.reshape(g, th, g, tw, arr.shape[-1])
        .swapaxes(1, 2)
        .reshape(g * g, th, tw, arr.shape[-1])
    )
    seed = np.uint32(0)
    for ch in key:
        seed = np.uint32(seed * np.uint32(131) + np.uint32(ord(ch)))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(tiles)
    out = (
        tiles.reshape(g, g, th, tw, arr.shape[-1])
        .swapaxes(1, 2)
        .reshape(nh, nw, arr.shape[-1])
    )
    done = Image.fromarray(out)
    if nw != w or nh != h:
        done = done.resize((w, h), Image.NEAREST)
    return done


def _mosaic(chart: Image.Image, factor: int = 16) -> Image.Image:
    """Coarse blocks of the same image. Bar layout stays; printed digits do not."""
    w, h = chart.size
    nw = max(1, w // factor)
    nh = max(1, h // factor)
    small = chart.resize((nw, nh), Image.BOX)
    return small.resize((w, h), Image.NEAREST)


def _blur(chart: Image.Image, sigma: float) -> Image.Image:
    """Smooth Gaussian blur. Layout stays; fine marks do not."""
    return chart.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    mode = os.environ.get("MUCG_INTV", "keep")
    if mode not in {"keep", "swap", "blank", "scramble", "mosaic", "blur", "tiles"}:
        raise SystemExit(
            "MUCG_INTV must be keep, swap, blank, scramble, mosaic, blur, or tiles, "
            f"got {mode}"
        )
    out = Path(os.environ["MUCG_INTV_OUT"])
    out.mkdir(parents=True, exist_ok=True)

    from vlmeval.vlm.ThinkMorph import ThinkMorph

    think_n = int(os.environ.get("MUCG_MAX_THINK", "256") or "256")
    model = ThinkMorph(
        model_path=str(MODEL),
        think=True,
        understanding_output=True,
        temperature=0.0,
        do_sample=False,
        max_think_token_n=think_n,
        save_dir=str(out / "imgs"),
        use_torch_compile=False,
    )
    with TSV.open() as f:
        rows_in = list(csv.DictReader(f, delimiter="\t"))
    max_n = int(os.environ.get("MUCG_MAX_ITEMS", "0") or "0")
    skip_n = int(os.environ.get("MUCG_SKIP_ITEMS", "0") or "0")
    if skip_n > 0:
        rows_in = rows_in[skip_n:]
    if max_n > 0:
        rows_in = rows_in[:max_n]
    prev = Image.new("RGB", (64, 64), (127, 127, 127))
    rows = []
    for i, rec in enumerate(rows_in):
        chart = _img(rec["image"])
        if i % world != rank:
            prev = chart
            continue
        if mode == "keep":
            used = chart
        elif mode == "blank":
            used = Image.new("RGB", chart.size, (127, 127, 127))
        elif mode == "scramble":
            used = _scramble(chart, str(rec["index"]))
        elif mode == "tiles":
            used = _tiles(
                chart,
                str(rec["index"]),
                int(os.environ.get("MUCG_TILE_GRID", "8") or "8"),
            )
        elif mode == "mosaic":
            used = _mosaic(chart, int(os.environ.get("MUCG_MOSAIC_FACTOR", "16") or "16"))
        elif mode == "blur":
            used = _blur(chart, float(os.environ.get("MUCG_BLUR_SIGMA", "2") or "2"))
        else:
            used = prev
        question = rec["question"]
        if rec.get("A"):
            lines = [question]
            for letter in ("A", "B", "C", "D"):
                if rec.get(letter):
                    lines.append(f"{letter}. {rec[letter]}")
            lines.append("Answer with a single letter.")
            question = "\n".join(lines)
        message = [
            {"type": "image", "value": used},
            {"type": "text", "value": question},
        ]
        pred = model.generate_inner(message, dataset="ChartQA_image_swap")
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_s = str(pred)
        rows.append(
            {
                "index": rec["index"],
                "answer": rec["answer"],
                "pred": pred_s,
                "hit": _hit(pred_s, rec["answer"]),
                "mode": mode,
            }
        )
        prev = chart
        if len(rows) % 20 == 0:
            print(f"[chart] rank={rank} mode={mode} n={len(rows)}", flush=True)
    dest = out / f"{mode}_rank{rank}.jsonl"
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    hits = sum(r["hit"] for r in rows)
    print(f"[chart] rank={rank} mode={mode} acc={hits}/{len(rows)} -> {dest}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
