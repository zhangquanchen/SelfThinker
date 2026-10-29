#!/usr/bin/env python3
"""Stage 2 (Verify): Filter by the 95th-percentile mismatched threshold Q_i and dump verified parquet."""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_selfthinker import ROOT


def pil_blob(path: Path) -> dict:
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "outputs/score_u_rollout/self.jsonl")
    ap.add_argument("--rollouts", type=Path, default=ROOT / "outputs/rollout_k6_vsearch/rollouts.jsonl")
    ap.add_argument(
        "--src-parquet",
        type=Path,
        default=ROOT / "data/fresh_pool/Visual_Search/data/train-00000-of-00001.parquet",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/audit_pass_l0_vsearch_trainonly/Visual_Search/data/train-00000-of-00001.parquet",
    )
    ap.add_argument("--meta-path", type=Path, default=None)
    ap.add_argument("--task", type=str, default="Visual_Search")
    args = ap.parse_args()

    score_files = sorted(args.scores.rglob("*.jsonl")) if args.scores.is_dir() else [args.scores]
    scores, seen_s = [], set()
    for fp in score_files:
        if not fp.is_file():
            continue
        for l in fp.read_text().splitlines():
            if "u_self" not in l:
                continue
            try:
                s = json.loads(l)
            except json.JSONDecodeError:
                continue
            key = f"{s.get('pid')}#{s.get('k')}"
            if key not in seen_s:
                seen_s.add(key)
                scores.append(s)

    rolls = {}
    roll_files = sorted(args.rollouts.rglob("rollouts.jsonl")) if args.rollouts.is_dir() else [args.rollouts]
    for fp in roll_files:
        if not fp.is_file():
            continue
        for l in fp.read_text().splitlines():
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("images"):
                rolls[(r["pid"], r["k"])] = r

    by = defaultdict(list)
    for s in scores:
        by[s["pid"]].append(s)

    src = pq.read_table(str(args.src_parquet))
    src_by = {
        str(src.column("pid")[i].as_py()): {c: src.column(c)[i].as_py() for c in src.column_names}
        for i in range(src.num_rows)
    }

    rows = []
    n_skip = 0
    for pid, ss in by.items():
        sh = [x["u_shuffle"] for x in ss]
        thr = sorted(sh)[max(0, int(0.95 * (len(sh) - 1)))]
        cands = [x for x in ss if x["u_self"] > thr]
        if not cands:
            n_skip += 1
            continue
        best = max(cands, key=lambda x: x["u_self"])
        rr = rolls.get((pid, best["k"]))
        base = src_by.get(pid)
        if not rr or not base:
            n_skip += 1
            continue
        vt_path = Path(rr["images"][0])
        if not vt_path.is_file():
            n_skip += 1
            continue
        rows.append(
            {
                "pid": pid,
                "question": base["question"],
                "answer": base["answer"],
                "problem_image_0": base["problem_image_0"],
                "resoning_thought_0": f"audit_pass u={best['u_self']:.3f}",
                "reasoning_image_0": pil_blob(vt_path),
                "resoning_thought_1": "",
                "task": args.task,
                "full_text_only_thought": base["answer"],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.table({k: [r[k] for r in rows] for k in rows[0]})
        pq.write_table(table, args.out)
    meta = {"n_pass": len(rows), "n_skip": n_skip, "n_items": len(by), "out": args.out.name}
    meta_path = args.meta_path or args.out.parent.parent.joinpath("meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
