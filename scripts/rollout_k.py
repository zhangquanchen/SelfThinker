#!/usr/bin/env python3
"""Stage 1 (Imagine): Sample K interleaved visual-thought trajectories per item."""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_selfthinker import ROOT, load_gen_inferencer


def _bytes_to_pil(blob):
    if isinstance(blob, dict) and "bytes" in blob:
        return Image.open(io.BytesIO(blob["bytes"])).convert("RGB")
    return None


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\{1,2}boxed\{([^}]*)\}", text or "")
    if matches:
        return matches[-1].strip()
    m = re.findall(r"<answer>\s*([^<]+)\s*</answer>", text or "", flags=re.I)
    if m:
        return m[-1].strip()
    return None


def already(path: Path) -> set[str]:
    done = set()
    if not path.is_file():
        return done
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("k") is not None:
                done.add(f"{rec['pid']}#{rec['k']}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data/fresh_pool/Visual_Search/data/train-00000-of-00001.parquet",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/rollout_k6_vsearch")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--max-think-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--prompt-suffix", default="")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(str(args.parquet))
    n = min(args.n, table.num_rows)
    indices = [i for i in range(args.start_index, n) if (i % args.num_shards) == args.shard_id]
    out = args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out / "rollouts.jsonl"
    done = already(jsonl)

    t0 = time.time()
    packed = load_gen_inferencer(model_path=args.model_path, max_mem_gib=70)
    inferencer = packed["inferencer"]

    n_ok = 0
    for i in indices:
        pid = str(table.column("pid")[i].as_py())
        image = _bytes_to_pil(table.column("problem_image_0")[i].as_py())
        question = table.column("question")[i].as_py() or ""
        if args.prompt_suffix.strip():
            question = question.rstrip() + "\n\n" + args.prompt_suffix.strip()
        gt = str(table.column("answer")[i].as_py() or "")
        if image is None:
            continue
        for k in range(args.k):
            key = f"{pid}#{k}"
            if key in done:
                continue
            t_s = time.time()
            torch.manual_seed(args.seed + i * 100 + k)
            torch.cuda.synchronize()
            out_list = inferencer.interleave_inference(
                [image, question],
                think=True,
                understanding_output=False,
                max_think_token_n=args.max_think_tokens,
                do_sample=True,
                text_temperature=0.7,
                cfg_text_scale=4.0,
                cfg_img_scale=2.0,
                cfg_renorm_type="text_channel",
                num_timesteps=args.steps,
                max_rounds=args.max_rounds,
            )
            torch.cuda.synchronize()
            texts, img_paths = [], []
            for j, x in enumerate(out_list):
                if isinstance(x, Image.Image):
                    p = img_dir / f"{pid}_k{k}_{j}.png"
                    x.save(p)
                    img_paths.append(str(p))
                else:
                    texts.append(str(x))
            text = "\n".join(texts)
            rec = {
                "pid": pid,
                "index": i,
                "k": k,
                "gt_answer": gt,
                "pred": extract_boxed(text),
                "n_vt": len(img_paths),
                "images": img_paths,
                "text_head": text[:800],
                "elapsed_s": time.time() - t_s,
            }
            with jsonl.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(key)
            n_ok += 1

    print(json.dumps({"ROLLOUT_OK": True, "n_ok": n_ok, "out": str(jsonl)}), flush=True)


if __name__ == "__main__":
    main()
