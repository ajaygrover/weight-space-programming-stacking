"""Robustness checks on the 100-problem MATH-500 outputs: second grader (math-verify), bootstrap CIs,
paired exact McNemar tests, and a non-looping-subset comparison. Writes out/llama/robustness.json + .md."""
import json, os, random, math
from math_verify import parse, verify

D = os.path.join(os.environ.get("WSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out", "llama")
MODELS = {"base": "base", "astra": "edited", "opus": "opus", "merged": "merged"}


def load(tag):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, f"gen_{tag}_math500.jsonl"), encoding="utf-8")}


def mv_correct(rec):
    if rec["hit_limit"] or not rec.get("pred"):
        return False
    try:
        gold = parse("$" + rec["answer"] + "$", parsing_timeout=None)
        pred = parse("$" + str(rec["pred"]) + "$", parsing_timeout=None)
        return bool(verify(gold, pred, timeout_seconds=None))
    except Exception:
        return False


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(xs, n=10000, seed=0):
    r = random.Random(seed); m = len(xs)
    means = sorted(sum(xs[r.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return round(100 * means[int(0.025 * n)], 1), round(100 * means[int(0.975 * n)], 1)


def main():
    R = {k: load(v) for k, v in MODELS.items()}
    ids = sorted(set.intersection(*(set(d) for d in R.values())))
    out = {"n_common": len(ids), "models": {}, "pairs": {}, "nonloop": {}}
    for k, d in R.items():
        ours = [int(d[i]["correct"]) for i in ids]
        mv = [int(mv_correct(d[i])) for i in ids]
        out["models"][k] = {"acc_ours": round(100 * sum(ours) / len(ids), 1), "ci95_ours": boot_ci(ours),
                            "acc_mathverify": round(100 * sum(mv) / len(ids), 1), "ci95_mathverify": boot_ci(mv),
                            "grader_disagreements": sum(a != b for a, b in zip(ours, mv)),
                            "hit_limit": sum(d[i]["hit_limit"] for i in ids)}
        for i, v in zip(ids, mv):
            d[i]["mv"] = v
    for a, b in [("merged", "astra"), ("merged", "opus"), ("opus", "astra"), ("astra", "base"), ("opus", "base"), ("merged", "base")]:
        for g in ("correct", "mv"):
            x = sum(R[a][i][g] and not R[b][i][g] for i in ids); y = sum(R[b][i][g] and not R[a][i][g] for i in ids)
            out["pairs"][f"{a}>{b} [{ 'ours' if g == 'correct' else 'math-verify'}]"] = {"a_only": x, "b_only": y, "p_mcnemar": round(mcnemar_exact(x, y), 6)}
    nl = [i for i in ids if not any(R[k][i]["hit_limit"] for k in ("astra", "opus", "merged"))]
    out["nonloop"] = {"n": len(nl), **{k: sum(R[k][i]["correct"] for i in nl) for k in ("astra", "opus", "merged")}}
    json.dump(out, open(os.path.join(D, "robustness.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
