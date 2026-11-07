import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benches as b

FENCE = "`" * 3
h = b.rows("humaneval")[0]
good = h["code_prompt"] + (
    "    for i, a in enumerate(numbers):\n"
    "        for j, c in enumerate(numbers):\n"
    "            if i != j and abs(a - c) < threshold:\n"
    "                return True\n"
    "    return False\n")
bad = h["code_prompt"] + "    return True\n"
evil = "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)\n" + good
print("good solution  :", b.grade("humaneval", h, f"<think>x</think>{FENCE}python\n{good}{FENCE}"))
print("wrong solution :", b.grade("humaneval", h, f"<think>x</think>{FENCE}python\n{bad}{FENCE}"))
print("network attempt:", b.grade("humaneval", h, f"<think>x</think>{FENCE}python\n{evil}{FENCE}"))
m = b.rows("mmlupro200")[0]
print("mmlu right     :", b.grade("mmlupro200", m, "<think>..</think> so \\boxed{" + m["answer"] + "}"), "gold", m["answer"])
print("mmlu \\text{}   :", b.grade("mmlupro200", m, "</think> \\boxed{\\text{" + m["answer"] + "}}"))
print("mmlu wrong     :", b.grade("mmlupro200", m, "</think> \\boxed{" + ("A" if m["answer"] != "A" else "B") + "}"))
