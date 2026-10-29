#!/usr/bin/env python3
"""Stage 2 (Verify): Score candidate visual thoughts by conditional PMI u(v)."""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_selfthinker import ROOT, add_sys_path, load_und_model

add_sys_path()
from data.data_utils import pil_img2rgb
from data.opsd_pack_builder import (
    OPSDPackBuilder,
    PackBuilderConfig,
    TEACHER_TRANSITION,
    packed_batch_to_device,
)
from data.transforms import ImageTransform

_FORWARD_KEYS = {
    "sequence_length",
    "packed_text_ids",
    "packed_text_indexes",
    "sample_lens",
    "packed_position_ids",
    "nested_attention_masks",
    "split_lens",
    "attn_modes",
    "ce_loss_indexes",
    "packed_label_ids",
    "packed_vit_tokens",
    "packed_vit_token_indexes",
    "packed_vit_position_ids",
    "vit_token_seqlens",
}


def _bytes_to_pil(blob):
    if isinstance(blob, dict) and "bytes" in blob:
        return Image.open(io.BytesIO(blob["bytes"])).convert("RGB")
    return None


def completion_logprob(model, batch) -> tuple[float, int]:
    fwd = {k: v for k, v in batch.items() if k in _FORWARD_KEYS}
    was_training = model.training
    model.train()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**fwd, return_logits=True)
    finally:
        model.train(was_training)
    logits = out["logits"]
    labels = batch["packed_label_ids"]
    n = min(logits.shape[0], labels.shape[0])
    logp = F.log_softmax(logits[:n].float(), dim=-1)
    token_lp = logp.gather(1, labels[:n].clamp(min=0).unsqueeze(1)).squeeze(1)
    return float(token_lp.sum().item()), int(n)


def control_tensors(vt_pils, kind: str, rng: np.random.Generator, transform):
    out = []
    for im in vt_pils:
        w, h = im.size
        if kind == "noise":
            arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
        elif kind == "blank":
            pil = Image.new("RGB", (w, h), (0, 0, 0))
        else:
            raise ValueError(kind)
        out.append(transform(pil_img2rgb(pil)))
    return out


def load_rollouts(path: Path):
    files = sorted(path.rglob("rollouts.jsonl")) if path.is_dir() else [path]
    recs, seen = [], set()
    for fp in files:
        with fp.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") or not r.get("images"):
                    continue
                key = f"{r.get('pid')}#{r.get('k')}"
                if key not in seen:
                    seen.add(key)
                    recs.append(r)
    return recs


def already(path: Path):
    done = set()
    if not path.is_file():
        return done
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if "u_self" in r:
                done.add(f"{r['pid']}#{r['k']}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, default=ROOT / "outputs/rollout_k6_vsearch/rollouts.jsonl")
    ap.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data/fresh_pool/Visual_Search/data/train-00000-of-00001.parquet",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/score_u_rollout/self.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--direction", choices=("gt", "consensus"), default="gt")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(str(args.parquet))
    by_pid = {}
    for i in range(table.num_rows):
        pid = str(table.column("pid")[i].as_py())
        by_pid[pid] = {
            "problem": _bytes_to_pil(table.column("problem_image_0")[i].as_py()),
            "question": table.column("question")[i].as_py() or "",
            "answer": str(table.column("answer")[i].as_py() or ""),
        }

    recs_all = load_rollouts(args.rollouts)
    consensus = {}
    if args.direction == "consensus":
        votes: dict[str, list[str]] = {}
        for r in recs_all:
            pred = r.get("pred")
            if pred:
                votes.setdefault(str(r.get("pid")), []).append(str(pred).strip())
        for pid, xs in votes.items():
            maj, cnt = Counter(xs).most_common(1)[0]
            consensus[pid] = {"d": maj, "n": cnt, "k": len(xs)}

    recs = [r for i, r in enumerate(recs_all) if (i % args.num_shards) == args.shard_id] if args.num_shards > 1 else recs_all
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already(args.out)

    packed = load_und_model(model_path=args.model_path, device=args.device)
    model = packed["model"]
    tokenizer = packed["tokenizer"]
    builder = OPSDPackBuilder(PackBuilderConfig(), packed["new_token_ids"], system_prompt=None)
    transform = ImageTransform(image_stride=14, max_image_size=980, min_image_size=378)
    transition_ids = tokenizer.encode(TEACHER_TRANSITION, add_special_tokens=False)
    rng = np.random.default_rng(0)
    device = args.device

    n_ok = 0
    for i, rec in enumerate(recs):
        key = f"{rec['pid']}#{rec['k']}"
        if key in done:
            continue
        src = by_pid.get(rec["pid"])
        if not src or src["problem"] is None:
            continue
        if args.direction == "consensus":
            cons = consensus.get(str(rec["pid"]))
            if not cons:
                continue
            d_text = cons["d"]
        else:
            cons = None
            d_text = src["answer"]
        vt = Image.open(rec["images"][0]).convert("RGB")
        shuf_rec = recs[(i + max(3, len(recs) // 5)) % len(recs)]
        shuf_vt = Image.open(shuf_rec["images"][0]).convert("RGB")
        d_ids = tokenizer.encode(d_text, add_special_tokens=False)
        if not d_ids:
            continue
        raw = {
            "problem_image_tensor": transform(pil_img2rgb(src["problem"])),
            "question_text": src["question"],
            "question_ids": tokenizer.encode(src["question"], add_special_tokens=False),
            "reference_vt_tensors": [transform(pil_img2rgb(vt))],
            "reference_answer_ids": d_ids,
            "pid": rec["pid"],
        }
        student = packed_batch_to_device(builder.build_student_sample(raw, completion_ids=d_ids), device)
        teacher = packed_batch_to_device(
            builder.build_teacher_sample(raw, completion_ids=d_ids, ref_intro_ids=None, transition_ids=transition_ids),
            device,
        )
        raw_s = dict(raw, reference_vt_tensors=[transform(pil_img2rgb(shuf_vt))])
        teacher_s = packed_batch_to_device(
            builder.build_teacher_sample(raw_s, completion_ids=d_ids, ref_intro_ids=None, transition_ids=transition_ids),
            device,
        )
        raw_n = dict(raw, reference_vt_tensors=control_tensors([vt], "noise", rng, transform))
        teacher_n = packed_batch_to_device(
            builder.build_teacher_sample(raw_n, completion_ids=d_ids, ref_intro_ids=None, transition_ids=transition_ids),
            device,
        )
        raw_b = dict(raw, reference_vt_tensors=control_tensors([vt], "blank", rng, transform))
        teacher_b = packed_batch_to_device(
            builder.build_teacher_sample(raw_b, completion_ids=d_ids, ref_intro_ids=None, transition_ids=transition_ids),
            device,
        )
        lp_x, _ = completion_logprob(model, student)
        lp_v, _ = completion_logprob(model, teacher)
        lp_s, _ = completion_logprob(model, teacher_s)
        lp_n, _ = completion_logprob(model, teacher_n)
        lp_b, _ = completion_logprob(model, teacher_b)
        out = {
            "pid": rec["pid"],
            "k": rec["k"],
            "gt_answer": src["answer"],
            "direction": args.direction,
            "d_text": d_text,
            "vote_n": (cons["n"] if cons else None),
            "gen_pred": rec.get("pred"),
            "u_self": lp_v - lp_x,
            "u_shuffle": lp_s - lp_x,
            "u_noise": lp_n - lp_x,
            "u_blank": lp_b - lp_x,
            "logp_x": lp_x,
        }
        with args.out.open("a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        n_ok += 1

    print(json.dumps({"SCORE_ROLLOUT_OK": n_ok > 0, "n_ok": n_ok}), flush=True)


if __name__ == "__main__":
    main()
