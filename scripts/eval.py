"""
Phase 2 -- evaluation harness for the weight-space-programming re-test.

Usage (from C:/AI/wsp-check):
  C:/AI/venvs/wsp/Scripts/python.exe scripts/eval.py --model base --bench math500 --n 100 \
      --max-new-tokens 8192 --device cuda --seed 0
  Variants: --model {base|edited|base_reppen|random_dir}; --rep-penalty {1.1|1.3} with base_reppen;
            random_dir uses --seed 0..2 (three runs) and needs out/diff.json from scripts/diff_weights.py.
  Smoke:  --model base --bench math500 --n 1 --max-new-tokens 256 --device cpu --tag smoke

Protocol (authors' logs, github.com/RikiyaT/lm_understand-logs, file exp8_math/code/evaluate.py,
SAMPLING_THINK35 / RESULTS_excerpt.md header "thinking ON, temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5, max 32,768"): temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5,
thinking ON, seed per generation; official Qwen chat template from the repo; user turn = problem +
"Please reason step by step, and put your final answer within \\boxed{}." (mathqa.build_prompt).
Deviation noted: run_eval.ps1 caps at 8192 new tokens, not 32768.

Outputs: out/gen_{tag}_{bench}{_smoke}.jsonl  (one line per generation: prompt id, output text,
tokens, hit_limit, correct, pred), out/eval_summary.json, out/eval_summary.md, out/bench/*.jsonl,
out/subset_ids.json.
"""
import argparse
from transformers import LogitsProcessor, LogitsProcessorList
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "out")
BENCH_DIR = os.path.join(OUT_DIR, "bench")
MODEL_DIRS = {"base": os.path.join(ROOT, "models", "base"),
              "edited": os.path.join(ROOT, "models", "edited")}
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# exp8_math/code/evaluate.py: SAMPLING_THINK35
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


# ---------------- benches ----------------
def prepare_bench(bench):
    path = os.path.join(BENCH_DIR, f"{bench}.jsonl")
    if os.path.exists(path):
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    os.makedirs(BENCH_DIR, exist_ok=True)
    from datasets import load_dataset
    rows = []
    if bench == "math500":
        for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
            rows.append({"id": r["unique_id"], "problem": r["problem"], "answer": str(r["answer"])})
    elif bench == "gsm8k":
        for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
            ans = r["answer"].split("####")[-1].strip().replace(",", "")
            rows.append({"id": f"gsm8k_test_{i}", "problem": r["question"], "answer": ans})
    else:
        raise ValueError(bench)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"prepared {bench}: {len(rows)} problems -> {path}", flush=True)
    return rows


def get_subset(bench, rows):
    """FIXED random subset (seed 0, 100 problems), ids saved to out/subset_ids.json."""
    sp = os.path.join(OUT_DIR, "subset_ids.json")
    ids = [r["id"] for r in rows]
    sub = json.load(open(sp)) if os.path.exists(sp) else {"seed": 0, "size": 100}
    if bench not in sub:
        picked = set(random.Random(0).sample(ids, 100))
        sub[bench] = [i for i in ids if i in picked]        # keep original order
        json.dump(sub, open(sp, "w"), indent=1)
        print(f"subset saved: {bench} {len(sub[bench])} ids", flush=True)
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in sub[bench]]


# ---------------- answer extraction / grading ----------------
def extract_boxed(text):
    """last \\boxed{...} with brace matching; None if absent."""
    out = None
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            return out
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        out = text[j + len("\\boxed{"):k - 1]
        i = k


FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def normalize_answer(s):
    """standard MATH-style normalisation of a boxed answer."""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac").replace(r"^{\circ}", "").replace(r"^\circ", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "").replace(r"\:", "")
    s = re.sub(r"\\(?:left|right|bigg?|Bigg?)", "", s)
    s = s.replace(r"\%", "%").replace(r"\$", "")
    s = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)   # brace-less \frac12 -> \frac{1}{2}
    prev = None
    while prev != s:                     # nested \frac{...}{...}
        prev = s
        s = FRAC_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = s.replace(r"\cdot", "*").replace(r"\div", "/")
    s = s.replace(r"\infty", "infinity").replace(r"\pi", "pi")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)   # thousands separators
    s = re.sub(r"\s+", "", s)            # whitespace never matters
    s = s.strip().strip(".").strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def as_number(s):
    if s is None:
        return None
    t = s.strip().replace("%", "").replace("$", "").replace(",", "")
    m = re.fullmatch(r"\(([^()]+)\)/\(([^()]+)\)", t)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            return a / b if b else None
        except ValueError:
            return None
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", t):
        return float(t)
    return None


def correct_math500(pred, gold):
    npred, ngold = normalize_answer(pred), normalize_answer(gold)
    if npred is None:
        return False
    if npred == ngold:
        return True
    x, y = as_number(npred), as_number(ngold)
    return x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(y))


def gsm8k_last_number(text):
    boxed = extract_boxed(text)
    if boxed:
        n = as_number(normalize_answer(boxed))
        if n is not None:
            return n
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return float(nums[-1].replace(",", "")) if nums else None


def correct_gsm8k(pred_text, gold):
    g = as_number(normalize_answer(gold))
    p = gsm8k_last_number(pred_text)
    return p is not None and g is not None and abs(p - g) <= 1e-6 * max(1.0, abs(g))


# ---------------- models ----------------
def load_model(model_dir, device, allow_oversubscribe=False):
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    if device == "cuda" and not allow_oversubscribe:
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / 1e9
        need_gb = 3.5
        print(f"GPU pre-flight: {free_gb:.2f} GB free of {total / 1e9:.2f} GB (need ~{need_gb} GB)", flush=True)
        if free_gb < need_gb:
            print(f"ABORT: not enough free VRAM. Free the GPU (llama-server holds most of it) "
                  f"or pass --allow-oversubscribe / --device cpu.", flush=True)
            sys.exit(2)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return tok, model


def apply_random_dir(model, device, seed):
    """random_dir baseline: rank-1 edit of the same size, random unit direction,
    same layers, same 64 columns, per-column amplitudes rescaled to the same
    Frobenius norm as the real edit (from out/diff.json)."""
    diff = json.load(open(os.path.join(OUT_DIR, "diff.json")))
    sd = model.state_dict()
    info = []
    for t in diff["changed_tensors"]:
        name = t["name"]
        w = sd[name]                          # live bf16 tensor [1024, 3584]
        cols = t["changed_columns"]
        coefs = torch.tensor(t["col_coefs_signed"], dtype=torch.float64)
        g = torch.Generator(device="cpu").manual_seed(seed)
        u = torch.randn(w.shape[0], generator=g, dtype=torch.float64)
        u = u / u.norm()
        delta = torch.outer(u, coefs)
        delta *= (t["frobenius"] / delta.norm())   # same norm of change as the real edit
        new = (w.detach().to(device).to(torch.float64) + delta.to(device)).to(torch.bfloat16)
        with torch.no_grad():
            w.copy_(new)
        nbits = int((w.view(torch.uint16) != sd[name].view(torch.uint16)).sum())  # 0 post-copy; informational
        info.append({"tensor": name, "seed": seed, "n_columns": len(cols),
                     "frobenius_applied": float(delta.norm()),
                     "entries_written": int(delta.numel())})
    return info


# ---------------- generation ----------------
def run_generation(args, tok, model, rows, tag):
    gen_path = os.path.join(OUT_DIR, f"gen_{tag}_{args.bench}{args.tag}.jsonl")
    if os.path.exists(gen_path) and not args.force:
        print(f"skip (already done): {gen_path}; use --force to rerun", flush=True)
        return None
    do_sample = not args.greedy
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    stop_ids = sorted(s for s in stop_ids if s is not None and s >= 0)
    rep_kwargs = {"repetition_penalty": args.rep_penalty} if args.model == "base_reppen" else {}
    records = []
    t0 = time.time()
    with open(gen_path, "w", encoding="utf-8") as f:
        for pi, r in enumerate(rows):
            messages = [{"role": "user", "content": r["problem"] + "\n" + INSTRUCTION}]
            text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                           enable_thinking=True, tokenize=False)
            inputs = tok(text, return_tensors="pt").to(model.device)
            in_len = inputs["input_ids"].shape[1]
            for s in range(args.samples):
                torch.manual_seed(args.seed * 100000 + pi * 100 + s)
                hf_sampling = {k: v for k, v in SAMPLING.items() if k != "presence_penalty"}
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=do_sample,
                    eos_token_id=stop_ids, pad_token_id=tok.eos_token_id,
                    logits_processor=LogitsProcessorList([PresencePenalty(in_len, SAMPLING["presence_penalty"])]),
                    **(hf_sampling if do_sample else {}), **rep_kwargs)
                gen = out[0][in_len:]
                n = int(gen.shape[0])
                hit = n >= args.max_new_tokens
                out_text = tok.decode(gen, skip_special_tokens=False)
                if args.bench == "math500":
                    pred = extract_boxed(out_text)
                    ok = correct_math500(pred, r["answer"])
                else:
                    pred = str(gsm8k_last_number(out_text))
                    ok = correct_gsm8k(out_text, r["answer"])
                rec = {"id": r["id"], "sample": s, "output": out_text, "tokens": n,
                       "hit_limit": hit, "pred": pred, "answer": r["answer"], "correct": bool(ok)}
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{tag}/{args.bench}] {pi + 1}/{len(rows)} {r['id']} "
                  f"acc={sum(x['correct'] for x in records) / len(records):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return records


class PresencePenalty(LogitsProcessor):
    """vLLM-style presence penalty: subtract a flat penalty from every token already generated (not the prompt)."""
    def __init__(self, prompt_len, penalty):
        self.prompt_len, self.penalty = prompt_len, penalty

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1] == 0 or self.penalty == 0:
            return scores
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, gen, True)
        return scores - self.penalty * mask.to(scores.dtype)


# ---------------- summary ----------------
def update_summary(args, tag, records, extra):
    sp = os.path.join(OUT_DIR, "eval_summary.json")
    res = json.load(open(sp)) if os.path.exists(sp) else {}
    n = len(records)
    if n == 0:
        print("no records; summary skipped", flush=True)
        return
    key = f"{tag}__{args.bench}{args.tag}"
    res[key] = {
        "model_tag": tag, "bench": args.bench, "tag": args.tag, "n_problems": len({r["id"] for r in records}),
        "samples_per_problem": args.samples, "n_generations": n,
        "max_new_tokens": args.max_new_tokens, "device": args.device, "seed": args.seed,
        "sampling": dict(SAMPLING, do_sample=not args.greedy, **extra),
        "accuracy_pct": 100.0 * sum(r["correct"] for r in records) / n,
        "hit_limit_pct": 100.0 * sum(r["hit_limit"] for r in records) / n,
        "mean_tokens": sum(r["tokens"] for r in records) / n,
        "seconds": extra.get("seconds"),
    }
    json.dump(res, open(sp, "w"), indent=1)
    lines = ["| model | bench | n | acc % | hit limit % | mean tokens | max_new_tokens | device | tag |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(res.items()):
        lines.append(f"| {v['model_tag']} | {v['bench']} | {v['n_generations']} | "
                     f"{v['accuracy_pct']:.1f} | {v['hit_limit_pct']:.1f} | {v['mean_tokens']:.0f} | "
                     f"{v['max_new_tokens']} | {v['device']} | {v['tag']} |")
    open(os.path.join(OUT_DIR, "eval_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("updated out/eval_summary.json / eval_summary.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["base", "edited", "base_reppen", "random_dir"])
    ap.add_argument("--bench", required=True, choices=["math500", "gsm8k"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-oversubscribe", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.model in ("base", "base_reppen", "random_dir"):
        model_dir = MODEL_DIRS["base"]
    else:
        model_dir = MODEL_DIRS["edited"]
    if args.model == "random_dir":
        if not os.path.exists(os.path.join(OUT_DIR, "diff.json")):
            sys.exit("random_dir needs out/diff.json -- run scripts/diff_weights.py first")
        tag = f"random_dir_s{args.seed}"
    elif args.model == "base_reppen":
        tag = f"base_reppen{str(args.rep_penalty).replace('.', '_')}"
    else:
        tag = args.model
    if args.tag:
        tag = tag + "_" + args.tag.lstrip("_")

    rows_all = prepare_bench(args.bench)
    rows = get_subset(args.bench, rows_all)[:args.n]
    print(f"{tag} x {args.bench}: {len(rows)} problems, max_new_tokens={args.max_new_tokens}, "
          f"device={args.device}, seed={args.seed}, samples={args.samples}", flush=True)

    t0 = time.time()
    tok, model = load_model(model_dir, args.device, args.allow_oversubscribe)
    extra = {}
    if args.model == "random_dir":
        extra["random_dir"] = apply_random_dir(model, args.device if args.device == "cuda" else "cpu", args.seed)
    records = run_generation(args, tok, model, rows, tag)
    if records is None:
        return
    extra["seconds"] = round(time.time() - t0)
    update_summary(args, tag, records, extra)


if __name__ == "__main__":
    main()
