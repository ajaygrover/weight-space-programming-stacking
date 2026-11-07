"""Headline figure for the README as plain SVG (no dependencies), light and dark:
MATH-500 (first 100) problems solved and tokens used, for the five models. Two panels with their own scales
(no dual axis). Merged is the accent; the other models are neutral.
usage: python make_figure.py  -> results/fig_math500_light.svg, results/fig_math500_dark.svg"""
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS = ["Merged (Astra + Opus)", "Astra edit (GPT-6)", "Opus edit (Claude)", "Random edit, same size", "Base, unedited"]
ACC = [55, 47, 40, 20, 6]                 # correct of 100
TOK = [1.46, 2.25, 2.06, 2.84, 3.14]      # total completion tokens, millions, same 100 problems
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  base="#c3c2b7", accent="#2a78d6", other="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 base="#383835", accent="#3987e5", other="#5a5955"),
}
W, H = 980, 330
LABEL_W = 190            # model-name column
PANEL_W = 330            # width of each bar area
GAP = 110                # gap between panels (room for value labels)
TOP, ROW, BAR = 70, 40, 22
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


def rounded_bar(x, y, w, h, r, fill):
    """Bar anchored square at the baseline (left), 4px rounded at the data end (right)."""
    r = min(r, w / 2, h / 2)
    return (f"<path d='M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} "
            f"Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z' fill='{fill}'/>")


def panel(x0, vals, xmax, ticks, fmt_tick, fmt_val, title, t):
    s = [f"<text x='{x0}' y='{TOP - 28}' {FONT} font-size='15' font-weight='700' fill='{t['ink']}'>{title}</text>"]
    bottom = TOP + ROW * len(vals)
    for tv in ticks:
        x = x0 + PANEL_W * tv / xmax
        s.append(f"<line x1='{x}' y1='{TOP - 8}' x2='{x}' y2='{bottom}' stroke='{t['grid']}' stroke-width='1'/>")
        s.append(f"<text x='{x}' y='{bottom + 18}' {FONT} font-size='11' fill='{t['muted']}' text-anchor='middle'>{fmt_tick(tv)}</text>")
    s.append(f"<line x1='{x0}' y1='{TOP - 8}' x2='{x0}' y2='{bottom}' stroke='{t['base']}' stroke-width='1'/>")
    for i, v in enumerate(vals):
        y = TOP + ROW * i + (ROW - BAR) / 2
        fill = t["accent"] if i == 0 else t["other"]
        w = PANEL_W * v / xmax
        s.append(rounded_bar(x0, y, w, BAR, 4, fill))
        weight, col = ("700", t["ink"]) if i == 0 else ("400", t["ink2"])
        s.append(f"<text x='{x0 + w + 8}' y='{y + BAR / 2 + 5}' {FONT} font-size='13' font-weight='{weight}' fill='{col}'>{fmt_val(v)}</text>")
    return s


def svg(t):
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' role='img' "
         f"aria-label='MATH-500 first 100: problems solved and tokens used by five models'>",
         f"<rect width='{W}' height='{H}' rx='8' fill='{t['surface']}'/>"]
    for i, m in enumerate(MODELS):
        y = TOP + ROW * i + ROW / 2 + 5
        weight, col = ("700", t["ink"]) if i == 0 else ("400", t["ink"])
        s.append(f"<text x='{LABEL_W - 12}' y='{y}' {FONT} font-size='13' font-weight='{weight}' fill='{col}' text-anchor='end'>{m}</text>")
    x1 = LABEL_W; x2 = LABEL_W + PANEL_W + GAP
    s += panel(x1, ACC, 60, [0, 20, 40, 60], str, str, "Problems solved (of 100)", t)
    s += panel(x2, TOK, 3.5, [0, 1, 2, 3], lambda v: f"{v}M", lambda v: f"{v:.2f}M", "Tokens used on the same 100 problems", t)
    s.append(f"<text x='16' y='{H - 14}' {FONT} font-size='11' fill='{t['muted']}'>Qwen3.5-0.8B, MATH-500 (fixed random 100), "
             f"one sample each, 32k-token limit, bf16, llama.cpp. Merged vs random edit: 37 vs 2 problems solved only by one (p = 3e-9).</text>")
    s.append("</svg>")
    return "\n".join(s)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, t in THEMES.items():
        p = os.path.join(OUT, f"fig_math500_{name}.svg")
        open(p, "w", encoding="utf-8").write(svg(t))
        print("wrote", p)


if __name__ == "__main__":
    main()
