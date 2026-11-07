"""Extra benchmarks for run_llama_eval.py: humaneval (code, graded in a Docker sandbox), mmlupro200 (fixed random 200),
math500rest (the 400 MATH-500 problems not in our first 100). Each row has id, prompt, and what grading needs."""
import json, os, random, re, subprocess, tempfile, uuid

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BENCH_DIR = os.path.join(ROOT, "out", "bench"); os.makedirs(BENCH_DIR, exist_ok=True)
SANDBOX = os.path.join(ROOT, "sandbox", "runs"); os.makedirs(SANDBOX, exist_ok=True)
DOCKER = os.environ.get("DOCKER", "docker")
NAMES = ("humaneval", "mmlupro200", "math500rest", "mbpp")
LETTERS = "ABCDEFGHIJ"


def _cache(name, build):
    p = os.path.join(BENCH_DIR, f"{name}.jsonl")
    if not os.path.exists(p):
        rows = build()
        with open(p, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def rows(name):
    from datasets import load_dataset
    if name == "humaneval":
        def b():
            ds = load_dataset("openai/openai_humaneval", split="test")
            return [{"id": r["task_id"], "entry_point": r["entry_point"], "test": r["test"], "code_prompt": r["prompt"],
                     "prompt": "Complete the following Python function. Reply with the complete function (including any "
                               "imports it needs) in a single ```python code block.\n\n```python\n" + r["prompt"] + "```"}
                    for r in ds]
        return _cache(name, b)
    if name == "mmlupro200":
        def b():
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            idx = sorted(random.Random(0).sample(range(len(ds)), 200))
            out = []
            for i in idx:
                r = ds[i]; opts = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(r["options"]))
                out.append({"id": f"mmlupro_{r['question_id']}", "answer": r["answer"], "category": r["category"],
                            "prompt": f"{r['question']}\n\n{opts}\n\nPlease reason step by step, and put the letter of your "
                                      "final answer within \\boxed{}."})
            return out
        return _cache(name, b)
    if name == "mbpp":
        def b():
            ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
            return [{"id": f"mbpp_{r['task_id']}", "tests": r["test_list"], "setup": r["test_setup_code"],
                     "prompt": "You are an expert Python programmer, and here is your task: " + r["text"] +
                               " Your code should pass these tests:\n\n" + "\n".join(r["test_list"]) +
                               "\n\nReply with the complete solution (including any imports it needs) in a single ```python code block."}
                    for r in ds]
        return _cache(name, b)
    if name == "math500rest":
        import eval as ev
        allrows = ev.prepare_bench("math500")
        first = set(json.load(open(os.path.join(ROOT, "out", "subset_ids.json")))["math500"])
        return [dict(r, prompt=r["problem"] + "\n" + ev.INSTRUCTION) for r in allrows if r["id"] not in first]
    raise KeyError(name)


def _final(text):
    return text.split("</think>")[-1]


def grade(name, row, text):
    if name == "math500rest":
        import eval as ev
        pred = ev.extract_boxed(text); return pred, ev.correct_math500(pred, row["answer"])
    if name == "mmlupro200":
        import eval as ev
        b = ev.extract_boxed(_final(text)) or ""
        m = re.search(r"\b([A-J])\b", b.replace("\\text", " ").replace("{", " ").replace("}", " "))
        pred = m.group(1) if m else None
        return pred, pred == row["answer"]
    if name == "mbpp":
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", _final(text), re.S)
        if not blocks:
            return None, False
        prog = blocks[-1] + "\n\n" + (row["setup"] or "") + "\n" + "\n".join(row["tests"]) + "\n"
        ok = _run_sandboxed(prog)
        return ("pass" if ok else "fail"), ok
    if name == "humaneval":
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", _final(text), re.S)
        if not blocks:
            return None, False
        code = blocks[-1]
        if f"def {row['entry_point']}" not in code:      # model gave only the body: prepend the signature
            code = row["code_prompt"] + code
        prog = code + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"
        ok = _run_sandboxed(prog)
        return ("pass" if ok else "fail"), ok
    raise KeyError(name)


def _run_sandboxed(prog):
    """Run a program in a throwaway container: no network, read-only, 256 MB, 1 CPU, 64 pids, 15 s. True if exit code 0."""
    d = os.path.join(SANDBOX, uuid.uuid4().hex); os.makedirs(d)
    open(os.path.join(d, "prog.py"), "w", encoding="utf-8").write(prog)
    try:
        r = subprocess.run([DOCKER, "run", "--rm", "--network", "none", "-m", "256m", "--cpus", "1", "--pids-limit", "64",
                            "--read-only", "-v", f"{d}:/w:ro", "python:3.12-slim", "timeout", "15", "python", "/w/prog.py"],
                           capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, PATH=os.path.dirname(DOCKER) + os.pathsep + os.environ.get("PATH", "")))
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
