#!/usr/bin/env python3
"""Context interventions on the 1,092 visual-search items (Table 9 / Appendix E)."""
from __future__ import annotations

import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_selfthinker import ROOT

PARQUET = Path(
    os.environ.get(
        "MUCG_INTV_PARQUET",
        str(ROOT / "data/audit_pass_l0_vsearch_trainonly/Visual_Search/data/train-00000-of-00001.parquet"),
    )
)
_DEFAULT_MODEL = (
    ROOT / "models"
    if (ROOT / "models" / "model.safetensors").is_file()
    else ROOT / "models/selfthinker"
)
MODEL = Path(os.environ.get("MUCG_INTV_MODEL", str(_DEFAULT_MODEL)))


def _norm(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("answer:", " ")
    words = re.findall(r"[a-z]+", text)
    return words[-1] if words else ""


def _hit(pred: str, answer: str) -> int:
    return int(_norm(pred) == _norm(str(answer)))


def _neutral_clause(answer: str) -> str:
    n_words = len(f"The answer is {answer}."[:-1].split())
    banned = {answer.strip().lower()}
    pool = ["A", "note", "is", "attached", "here", "now", "please", "look", "again", "today", "mark"]
    words = [w for w in pool if w.lower() not in banned] or ["mark"]
    out = words[:n_words]
    if len(out) < n_words:
        out = (out + ["mark"] * n_words)[:n_words]
    return " ".join(out) + "."


def _img(cell) -> Image.Image:
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    return Image.open(BytesIO(raw)).convert("RGB")


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    mode = os.environ.get("MUCG_INTV", "vt")
    if mode not in {"vt", "swap", "none", "blank", "textswap", "textown", "textneutral", "vtonly"}:
        raise SystemExit(
            "MUCG_INTV must be vt, swap, none, blank, textswap, textown, textneutral, or vtonly, "
            f"got {mode}"
        )
    out = Path(os.environ.get("MUCG_INTV_OUT", str(ROOT / f"outputs/intv_{mode}")))
    out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    from vlmeval.vlm.ThinkMorph import ThinkMorph

    model = ThinkMorph(
        model_path=str(MODEL),
        think=True,
        understanding_output=True,
        temperature=0.0,
        do_sample=False,
        max_think_token_n=256,
        save_dir=str(out / "imgs"),
        use_torch_compile=False,
    )
    table = pq.read_table(
        PARQUET,
        columns=["pid", "question", "answer", "problem_image_0", "reasoning_image_0"],
    )
    n = table.num_rows
    prev_vt = Image.new("RGB", (64, 64), (127, 127, 127))
    prev_answer = "gray"
    rows = []
    for i in range(n):
        vt = _img(table.column("reasoning_image_0")[i].as_py())
        answer = table.column("answer")[i].as_py()
        if i % world != rank:
            prev_vt = vt
            prev_answer = answer
            continue
        problem = _img(table.column("problem_image_0")[i].as_py())
        question = table.column("question")[i].as_py()
        if mode == "none":
            message = [{"type": "image", "value": problem}, {"type": "text", "value": question}]
        elif mode == "blank":
            message = [
                {"type": "image", "value": problem},
                {"type": "image", "value": Image.new("RGB", problem.size, (127, 127, 127))},
                {"type": "text", "value": question},
            ]
        elif mode == "textswap":
            message = [
                {"type": "image", "value": problem},
                {"type": "text", "value": f"{question} A different example was {prev_answer}."},
            ]
        elif mode == "textown":
            message = [
                {"type": "image", "value": problem},
                {"type": "text", "value": f"{question} The answer is {answer}."},
            ]
        elif mode == "textneutral":
            message = [
                {"type": "image", "value": problem},
                {"type": "text", "value": f"{question} {_neutral_clause(str(answer))}"},
            ]
        elif mode == "vtonly":
            message = [{"type": "image", "value": vt}, {"type": "text", "value": question}]
        else:
            used = vt if mode == "vt" else prev_vt
            message = [
                {"type": "image", "value": problem},
                {"type": "image", "value": used},
                {"type": "text", "value": question},
            ]
        pred = model.generate_inner(message, dataset="Visual_Search_intv")
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_s = str(pred)
        rows.append(
            {
                "pid": table.column("pid")[i].as_py(),
                "answer": answer,
                "pred": pred_s,
                "hit": _hit(pred_s, str(answer)),
                "mode": mode,
            }
        )
        prev_vt = vt
        prev_answer = answer
    dest = out / f"{mode}_rank{rank}.jsonl"
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
