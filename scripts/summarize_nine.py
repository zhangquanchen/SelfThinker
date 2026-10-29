#!/usr/bin/env python3
"""Summarize VLMEvalKit outputs into the seven-benchmark average (Avg(7)).

ChartQA is the mean of the horizontal-bar and vertical-bar subsets (ChartQA_h_bar, ChartQA_v_bar).
CV-Bench is the mean of the 2D and 3D subsets (CV-Bench-2D, CV-Bench-3D).
SAT uses the circular evaluation split (SAT_circular).
VSP uses the main maze task split (VSP_maze_task_main_original).
Values in (0, 1.5] are fractions and are scaled to percent.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BENCHES = {
    "VSP": ["VSP_maze_task_main_original"],
    "ChartQA": ["ChartQA_h_bar", "ChartQA_v_bar"],
    "VStar": ["VStarBench"],
    "MMVP": ["MMVP"],
    "SAT": ["SAT_circular"],
    "BLINK": ["BLINK"],
    "CV-Bench": ["CV-Bench-2D", "CV-Bench-3D"],
}


def _to_pct(x: float) -> float:
    return 100.0 * x if 0.0 <= x <= 1.5 else x


def _read_overall(path: Path) -> float | None:
    try:
        rows = list(csv.reader(path.open()))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = [c.strip() for c in rows[0]]
    body = rows[-1]
    if "Overall" in header:
        idx = header.index("Overall")
        if idx < len(body) and body[idx]:
            return _to_pct(float(body[idx]))
    for cell in reversed(body):
        try:
            return _to_pct(float(cell))
        except ValueError:
            continue
    return None


def collect(root: Path, model_key: str) -> dict:
    model_dir = root / model_key if (root / model_key).is_dir() else root
    out = {"model": model_key, "root": str(root), "benches": {}, "missing": []}
    vals = []
    for name, stems in BENCHES.items():
        found = []
        for stem in stems:
            hits = sorted(model_dir.rglob(f"*{stem}*_acc.csv"))
            hits = [p for p in hits if "bak_" not in str(p)]
            if not hits:
                score_hits = sorted(model_dir.rglob(f"*{stem}*score.csv"))
                hits = [p for p in score_hits if "bak_" not in str(p)]
            if hits:
                v = _read_overall(hits[-1])
                if v is not None:
                    found.append(v)
        if found:
            out["benches"][name] = sum(found) / len(found)
            vals.append(out["benches"][name])
        else:
            out["missing"].append(name)
    out["n_present"] = len(vals)
    out["avg7"] = (sum(vals) / 7.0) if len(vals) == 7 else None
    out["avg_present"] = (sum(vals) / len(vals)) if vals else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path, help="eval work-dir(s)")
    ap.add_argument("--model-key", default="", help="subdir; default = only child dir")
    args = ap.parse_args()
    rows = []
    for root in args.roots:
        key = args.model_key
        if not key:
            kids = [p for p in root.iterdir() if p.is_dir() and p.name != "imgs"] if root.is_dir() else []
            key = kids[0].name if kids else root.name
        rows.append(collect(root, key))
    print(json.dumps(rows if len(rows) > 1 else rows[0], indent=2))


if __name__ == "__main__":
    main()
