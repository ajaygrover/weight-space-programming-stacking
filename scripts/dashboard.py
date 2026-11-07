"""Build out/dashboard.html (live results page) from out/llama/gen_*.jsonl. Re-run any time; it reads whatever exists."""
import json, math, os, statistics as st, time, html

D = os.path.join(os.environ.get("WSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out", "llama")
OUT = os.path.join(os.path.dirname(D), "dashboard.html")
MODELS = [("merged", "Merged (Astra + Opus)"), ("edited", "Astra (GPT-6) edit"), ("opus", "Opus 5.5 edit"), ("base", "Base (unedited)"),
          ("random_merged_s0", "Random-direction control")]
BENCHES = [("math500", "MATH-500, first 100", 100), ("humaneval", "HumanEval (Docker-executed)", 164),
           ("mmlupro200", "MMLU-Pro, 200 sampled", 200), ("gsm8k", "GSM8K, first 100", 100),
           ("math500rest", "MATH-500, other 400", 400), ("mbpp", "MBPP (Docker-executed)", 500)]


def load(m, b):
    import glob
    ps = glob.glob(os.path.join(D, f"gen_{m}_fullpen_{b}.jsonl")) + glob.glob(os.path.join(D, f"gen_{m}_fullpen_{b}.shard*.jsonl"))
    if not ps:
        return None
    out = {}
    for p in ps:
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l); out[r["id"]] = r
            except Exception:
                pass
    return out or None


def mcnemar(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


def fmt_p(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def section(b, title, total):
    R = {m: load(m, b) for m, _ in MODELS}
    have = [(m, lab) for m, lab in MODELS if R[m]]
    if not have:
        return f"<section><h2>{title}</h2><p class=muted>Queued.</p></section>"
    rows = []
    best = max(sum(r["correct"] for r in R[m].values()) / len(R[m]) for m, _ in have if len(R[m]) == total) if any(len(R[m]) == total for m, _ in have) else None
    for m, lab in have:
        rs = list(R[m].values()); n = len(rs); c = sum(r["correct"] for r in rs)
        acc = 100 * c / n; done = n == total
        status = "done" if done else f"running {n}/{total}"
        rows.append(f"<tr><td>{lab}</td><td class=num><b>{c}</b> / {n}</td><td class=num>{acc:.1f}%</td>"
                    f"<td class=bar><span style='width:{acc:.1f}%' class='{'best' if done and best and abs(acc/100-best)<1e-9 else ''}'></span></td>"
                    f"<td class=num>{st.median(r['tokens'] for r in rs):,.0f}</td><td class=num>{sum(r['tokens'] for r in rs)/1e6:.2f}M</td>"
                    f"<td class=num>{100*sum(r['hit_limit'] for r in rs)/n:.0f}%</td><td class='st {'ok' if done else 'run'}'>{status}</td></tr>")
    pairs = []
    for a, bb in [("merged", "edited"), ("merged", "base"), ("edited", "base"), ("merged", "opus"), ("opus", "edited"),
                  ("merged", "random_merged_s0")]:
        if R[a] and R[bb]:
            ids = sorted(set(R[a]) & set(R[bb]))
            if len(ids) < 10:
                continue
            x = sum(R[a][i]["correct"] and not R[bb][i]["correct"] for i in ids)
            y = sum(R[bb][i]["correct"] and not R[a][i]["correct"] for i in ids)
            both = [i for i in ids if R[a][i]["correct"] and R[bb][i]["correct"]]
            tok = ""
            if len(both) >= 5:
                ta = st.median(R[a][i]["tokens"] for i in both); tb = st.median(R[bb][i]["tokens"] for i in both)
                tok = f"{ta:,.0f} vs {tb:,.0f}"
            p = mcnemar(x, y)
            pairs.append(f"<tr><td>{dict(MODELS)[a]} vs {dict(MODELS)[bb]}</td><td class=num>{len(ids)}</td><td class=num>{x}</td><td class=num>{y}</td>"
                         f"<td class='num {'sig' if p < 0.05 else ''}'>{fmt_p(p)}</td><td class=num>{tok}</td></tr>")
    ptab = ("<h3>Paired comparisons (same problems)</h3><table><tr><th>Pair</th><th>Problems</th><th>Only first solved</th>"
            "<th>Only second solved</th><th>p (exact McNemar)</th><th>Median tokens, both solved</th></tr>" + "".join(pairs) + "</table>") if pairs else ""
    extra = ("<p class=muted>Base has not been rerun with the full-length penalty here. With the old 64-token penalty window it "
             "scored 7/100, and 93 of its answers hit the 32k cap.</p>") if b == "math500" and not R["base"] else ""
    return (f"<section><h2>{title}</h2>{extra}<table><tr><th>Model</th><th>Correct</th><th>Accuracy</th><th></th><th>Median tokens</th><th>Total tokens</th>"
            f"<th>Hit 32k cap</th><th>Status</th></tr>{''.join(rows)}</table>{ptab}</section>")


def main():
    now = time.strftime("%d %b %Y, %H:%M")
    log = open(os.path.join(D, "overnight.log"), encoding="utf-8", errors="replace").read().splitlines()
    ev_all = [l.strip("= ").strip() for l in log if l.startswith("===")]
    # hide the aborted parallel-lane VRAM tests (21:02-21:09): lane on port 8091, guard stops, and their LANE DONE lines
    events, in_test = [], False
    for e in ev_all:
        if "START lane" in e:
            in_test = "8091" in e
        if "8091" in e or "VRAM GUARD" in e or ("LANE DONE" in e and in_test):
            continue
        events.append(e)
    events = events[-8:]
    body = "".join(section(*b) for b in BENCHES)
    page = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Weight-edit evals: live results</title><style>
:root{{--fg:#1b1b1f;--mut:#6b6b76;--line:#e4e4ea;--bg:#fbfbfc;--acc:#3b5bdb;--hi:#2b8a3e}}
@media (prefers-color-scheme:dark){{:root{{--fg:#ececf1;--mut:#9a9aa6;--line:#2c2c33;--bg:#151518;--acc:#7c94f5;--hi:#5fcf7a}}}}
body{{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg);max-width:980px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:34px 0 10px}} h3{{font-size:14px;margin:18px 0 6px;color:var(--mut);font-weight:600}}
.muted{{color:var(--mut)}} table{{border-collapse:collapse;width:100%;font-size:14px}}
th{{text-align:left;font-weight:600;color:var(--mut);font-size:12px;border-bottom:1px solid var(--line);padding:6px 8px}}
td{{border-bottom:1px solid var(--line);padding:7px 8px}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.bar{{width:22%}} .bar span{{display:block;height:9px;border-radius:5px;background:var(--acc);opacity:.55}} .bar span.best{{opacity:1}}
.st{{font-size:12px}} .ok{{color:var(--hi)}} .run{{color:var(--acc)}} .sig{{color:var(--hi);font-weight:700}}
.note{{border-left:3px solid var(--acc);padding:6px 12px;margin:14px 0;color:var(--fg)}} ul{{padding-left:20px}} li{{margin:2px 0}}
code{{font-size:13px}}</style></head><body>
<h1>Weight-edit evals: live results</h1>
<p class=muted>Qwen3.5-0.8B with hand-edited weights. Updated {now}. All runs: thinking on, temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5 over the full context (vLLM-equivalent), 32,768-token cap, llama.cpp BF16.</p>
<div class=note><b>Models.</b> <b>Astra</b> = the paper's GPT-6 edit (rank-1, layers 15 and 19). <b>Opus 5.5</b> = our independent edit
(a "doubt" direction projected out of layers 12–23). <b>Merged</b> = both edits applied together. <b>Base</b> = unedited.</div>
{body}
<section><h2>Run log</h2><ul class=muted>{''.join(f'<li>{html.escape(e)}</li>' for e in events)}</ul></section>
<p class=muted>Accuracy bars: the darker bar marks the best completed model per benchmark. "Only first/second solved" counts problems
one model got right and the other got wrong; p values below 0.05 are green. Ajay Grover, Superposition.</p>
</body></html>"""
    open(OUT, "w", encoding="utf-8").write(page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
