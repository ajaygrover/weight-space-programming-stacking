"""Summarise per-problem outputs (out/llama/gen_{model}_fullpen_{bench}[.shardK].jsonl) into results/summary.json
and results/summary.md: accuracy, tokens and 32k-cap rate per model, plus paired tests per model pair.
  accuracy: exact two-sided McNemar on problems both models attempted
  tokens:   two-sided sign test on per-problem completion tokens (ties dropped)
usage: WSP_ROOT=<dir containing out/llama> python summarize.py"""
import glob, json, math, os, statistics as st

ROOT = os.environ.get("WSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "out", "llama")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS = {"merged": "Merged (Astra + Opus)", "edited": "Astra (GPT-6) edit", "opus": "Opus 5.5 edit",
          "base": "Base (unedited)", "random_merged_s0": "Random-direction control"}
BENCHES = {"math500": "MATH-500, first 100 (seed-0 subset)", "math500rest": "MATH-500, other 400",
           "humaneval": "HumanEval (Docker-executed)", "mbpp": "MBPP (Docker-executed)",
           "mmlupro200": "MMLU-Pro, 200 sampled (seed 0)", "gsm8k": "GSM8K, first 100"}
PAIRS = [("merged", "edited"), ("merged", "opus"), ("opus", "edited"), ("merged", "base"), ("edited", "base"),
         ("opus", "base"), ("merged", "random_merged_s0"), ("random_merged_s0", "base")]


def load(m, b):
    d = {}
    for p in glob.glob(os.path.join(D, f"gen_{m}_fullpen_{b}.jsonl")) + glob.glob(os.path.join(D, f"gen_{m}_fullpen_{b}.shard*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); d[r["id"]] = r
    return d


def binom_two_sided(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)


def main():
    res = {"settings": "thinking on; temperature 0.6, top_p 0.95, top_k 20; presence penalty 1.5 over the whole output "
                       "(repeat_last_n 32768); max 32,768 new tokens; one sample per problem, per-problem seed shared by all models; "
                       "llama.cpp b11960, bf16 weights", "benchmarks": {}}
    md = ["# Results", "", res["settings"], ""]
    for b, title in BENCHES.items():
        R = {m: load(m, b) for m in MODELS}
        R = {m: v for m, v in R.items() if v}
        if not R:
            continue
        rows = {}
        md += [f"## {title}", "", "| Model | Correct | n | Accuracy | Median tokens | Total tokens | Hit 32k cap |", "|---|---|---|---|---|---|---|"]
        for m, v in R.items():
            rs = list(v.values()); n = len(rs); c = sum(r["correct"] for r in rs)
            rows[m] = {"n": n, "correct": c, "accuracy": round(c / n, 4), "median_tokens": st.median(r["tokens"] for r in rs),
                       "total_tokens": sum(r["tokens"] for r in rs), "capped": sum(r["hit_limit"] for r in rs)}
            md.append(f"| {MODELS[m]} | {c} | {n} | {100*c/n:.1f}% | {rows[m]['median_tokens']:,.0f} | {rows[m]['total_tokens']/1e6:.2f}M | {rows[m]['capped']} |")
        pairs = []
        for a, c in PAIRS:
            if a in R and c in R:
                ids = sorted(set(R[a]) & set(R[c]))
                if len(ids) < 10:
                    continue
                X, Y = R[a], R[c]
                x = sum(X[i]["correct"] and not Y[i]["correct"] for i in ids); y = sum(Y[i]["correct"] and not X[i]["correct"] for i in ids)
                f = sum(X[i]["tokens"] < Y[i]["tokens"] for i in ids); g = sum(X[i]["tokens"] > Y[i]["tokens"] for i in ids)
                tx = sum(X[i]["tokens"] for i in ids); ty = sum(Y[i]["tokens"] for i in ids)
                pairs.append({"a": a, "b": c, "n": len(ids), "a_correct": sum(X[i]["correct"] for i in ids), "b_correct": sum(Y[i]["correct"] for i in ids),
                              "only_a": x, "only_b": y, "p_mcnemar": binom_two_sided(x, y),
                              "a_fewer_tokens": f, "b_fewer_tokens": g, "p_sign_tokens": binom_two_sided(f, g),
                              "a_total_tokens": tx, "b_total_tokens": ty})
        if pairs:
            md += ["", "| Pair | n | Correct | Only first / only second | p (McNemar) | Fewer tokens (first / second) | p (sign) | Total tokens |", "|---|---|---|---|---|---|---|---|"]
            for q in pairs:
                md.append(f"| {MODELS[q['a']]} vs {MODELS[q['b']]} | {q['n']} | {q['a_correct']} vs {q['b_correct']} | {q['only_a']} / {q['only_b']} | "
                          f"{q['p_mcnemar']:.3g} | {q['a_fewer_tokens']} / {q['b_fewer_tokens']} | {q['p_sign_tokens']:.2g} | "
                          f"{q['a_total_tokens']/1e6:.2f}M vs {q['b_total_tokens']/1e6:.2f}M |")
        md.append("")
        res["benchmarks"][b] = {"title": title, "models": rows, "pairs": pairs}
    # MATH-500 as one set of 500 where both parts exist
    M = {m: {**load(m, "math500"), **load(m, "math500rest")} for m in MODELS}
    full = {m: v for m, v in M.items() if len(v) == 500}
    if len(full) >= 2:
        md += ["## MATH-500, all 500", "", "| Pair | Correct | Only first / only second | p (McNemar) | Fewer tokens | p (sign) |", "|---|---|---|---|---|---|"]
        res["math500_all"] = []
        for a, c in PAIRS:
            if a in full and c in full:
                X, Y = full[a], full[c]; ids = sorted(X)
                x = sum(X[i]["correct"] and not Y[i]["correct"] for i in ids); y = sum(Y[i]["correct"] and not X[i]["correct"] for i in ids)
                f = sum(X[i]["tokens"] < Y[i]["tokens"] for i in ids); g = sum(X[i]["tokens"] > Y[i]["tokens"] for i in ids)
                q = {"a": a, "b": c, "a_correct": sum(X[i]["correct"] for i in ids), "b_correct": sum(Y[i]["correct"] for i in ids),
                     "only_a": x, "only_b": y, "p_mcnemar": binom_two_sided(x, y), "a_fewer_tokens": f, "b_fewer_tokens": g,
                     "p_sign_tokens": binom_two_sided(f, g)}
                res["math500_all"].append(q)
                md.append(f"| {MODELS[a]} vs {MODELS[c]} | {q['a_correct']} vs {q['b_correct']} of 500 | {x} / {y} | {q['p_mcnemar']:.3g} | {f} / {g} | {q['p_sign_tokens']:.2g} |")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
