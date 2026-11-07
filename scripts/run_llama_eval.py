"""Run one GGUF variant on the fixed 100+100 subsets through llama-server (parallel slots), authors' think35 sampling.
usage: run_llama_eval.py VARIANT [--rep-penalty X] [--port 8090]
Writes out/llama/gen_{tag}_{bench}.jsonl (resumable) and out/llama/summary.json."""
import argparse, json, os, subprocess, sys, time, urllib.request, concurrent.futures as cf
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval as ev
import benches
SERVER = os.environ.get("LLAMA_SERVER", "llama-server")  # the paper used llama.cpp b11960
OUT = os.path.join(ROOT, "out", "llama"); os.makedirs(OUT, exist_ok=True)
NP, MAXTOK = 24, 32768

def post(port, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def wait_ready(port, proc):
    for _ in range(300):
        if proc.poll() is not None: raise SystemExit("llama-server died; see log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if b"ok" in r.read(): return
        except Exception: pass
        time.sleep(2)
    raise SystemExit("server not ready")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("variant"); ap.add_argument("--rep-penalty", type=float, default=0)
    ap.add_argument("--port", type=int, default=8090); ap.add_argument("--benches", default="math500,gsm8k"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--first", type=int, default=0); ap.add_argument("--np", type=int, default=24); ap.add_argument("--full-penalty", action="store_true")
    ap.add_argument("--shard", default="", help="K/N: only rows with index %% N == K, written to gen_..._{bench}.shardK.jsonl (seeds unchanged)")
    a = ap.parse_args(); global NP; NP = a.np
    tag = a.variant + (f"_rep{a.rep_penalty}" if a.rep_penalty else "") + (f"_smoke{a.limit}" if a.limit else "") + ("_fullpen" if a.full_penalty else "")
    gguf = os.path.join(ROOT, "gguf", f"{a.variant}-bf16.gguf")
    import socket
    with socket.socket() as so:
        if so.connect_ex(("127.0.0.1", a.port)) == 0:
            raise SystemExit(f"port {a.port} already in use: refusing to start (stale server?)")
    log = open(os.path.join(OUT, f"server_{tag}.log"), "w")
    proc = subprocess.Popen([SERVER, "-m", gguf, "--port", str(a.port), "-ngl", "99", "-np", str(NP),
                             "-c", str(NP * (MAXTOK + 1024)), "-fa", "on", "--jinja", "--reasoning-format", "none",
                             "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(a.port, proc)
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/props", timeout=10) as r:
            served = json.loads(r.read()).get("model_path", "?")
        assert os.path.basename(served) == os.path.basename(gguf), f"server is serving {served}, expected {gguf}"
        print("serving", served, flush=True)
        summ_p = os.path.join(OUT, "summary.json")
        for bench in a.benches.split(","):
            rows = benches.rows(bench) if bench in benches.NAMES else ev.get_subset(bench, ev.prepare_bench(bench))
            rows = rows[:a.limit] if a.limit else rows
            rows = rows[:a.first] if a.first else rows
            gp = os.path.join(OUT, f"gen_{tag}_{bench}.jsonl")
            done = {json.loads(l)["id"] for l in open(gp, encoding="utf-8")} if os.path.exists(gp) else set()
            if a.shard:  # rows already in the main file are skipped; this shard writes its own file
                k, nsh = map(int, a.shard.split("/"))
                keep = {r["id"] for i, r in enumerate(rows) if i % nsh == k}
                gp = os.path.join(OUT, f"gen_{tag}_{bench}.shard{k}.jsonl")
                if os.path.exists(gp):
                    done |= {json.loads(l)["id"] for l in open(gp, encoding="utf-8")}
                done |= {r["id"] for r in rows if r["id"] not in keep}
            todo = [r for r in rows if r["id"] not in done]
            print(f"[{tag}/{bench}] {len(done)} done, {len(todo)} to go", flush=True)
            def one(pi_r):
                pi, r = pi_r
                body = {"messages": [{"role": "user", "content": r.get("prompt") or (r["problem"] + "\n" + ev.INSTRUCTION)}],
                        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                        "max_tokens": MAXTOK, "seed": pi, "chat_template_kwargs": {"enable_thinking": True}}
                if a.rep_penalty: body["repeat_penalty"] = a.rep_penalty
                if a.full_penalty: body["repeat_last_n"] = 32768
                res = post(a.port, body); ch = res["choices"][0]
                text = (ch["message"].get("reasoning_content") or "") + (ch["message"].get("content") or "")
                if bench in benches.NAMES:
                    pred, ok = benches.grade(bench, r, text)
                elif bench == "math500":
                    pred = ev.extract_boxed(text); ok = ev.correct_math500(pred, r["answer"])
                else:
                    pred = str(ev.gsm8k_last_number(text)); ok = ev.correct_gsm8k(text, r["answer"])
                return {"id": r["id"], "tokens": res["usage"]["completion_tokens"], "finish": ch["finish_reason"],
                        "hit_limit": ch["finish_reason"] == "length", "pred": pred,
                        "correct": bool(ok), "answer": r.get("answer"), "output": text}
            t0 = time.time(); n = 0
            with open(gp, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(NP) as ex:
                idx = {r["id"]: i for i, r in enumerate(rows)}
                for rec in ex.map(one, [(idx[r["id"]], r) for r in todo]):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1
                    if n % 10 == 0: print(f"  {n}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
            recs = [json.loads(l) for l in open(gp, encoding="utf-8")]
            entry = {"n": len(recs), "accuracy_pct": round(100 * sum(r["correct"] for r in recs) / len(recs), 1),
                     "hit_limit_pct": round(100 * sum(r["hit_limit"] for r in recs) / len(recs), 1),
                     "mean_tokens": round(sum(r["tokens"] for r in recs) / len(recs))}
            print(tag, bench, entry, flush=True)
            try:  # two lanes may write summary.json at once; the jsonl files are the source of truth
                s = json.load(open(summ_p)) if os.path.exists(summ_p) else {}
                s[f"{tag}/{bench}"] = entry; json.dump(s, open(summ_p, "w"), indent=1)
            except Exception as e:
                print("summary.json write skipped:", e, flush=True)
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        proc.wait(30)

if __name__ == "__main__":
    main()
