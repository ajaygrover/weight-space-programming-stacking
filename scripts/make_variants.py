"""Build GGUF variants by patching raw tensor bytes in the base BF16 GGUF (no converter needed).
Variants: edited (authors' published weights), random_dir_s{0,1,2} (eval.py's apply_random_dir).
Safety check first: base GGUF bytes for the target tensors must equal the base HF safetensors bytes."""
import glob, json, os, shutil, struct, sys
import numpy as np, torch
from safetensors.torch import load_file

ROOT = os.environ.get("WSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); G = os.path.join(ROOT, "gguf")
MAP = {f"model.language_model.layers.{L}.mlp.down_proj.weight": f"blk.{L}.ffn_down.weight" for L in (15, 19)}

def read_gguf_index(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF"; ver = rd("I"); nt = rd("Q"); nkv = rd("Q")
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t == 8: return rs()
        if t == 9:
            et = rd("I"); n = rd("Q"); return [val(et) for _ in range(n)]
        return rd(sizes[t])
    align = 32
    for _ in range(nkv):
        k = rs(); t = rd("I"); v = val(t)
        if k == "general.alignment": align = v
    tensors = {}
    for _ in range(nt):
        name = rs(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; typ = rd("I"); off = rd("Q")
        tensors[name] = (dims, typ, off)
    start = f.tell(); start += (-start) % align
    return ver, tensors, start

def bf16_bytes(t):
    return t.contiguous().view(torch.int16).numpy().tobytes()

def patch(src, dst, new_tensors, idx, start):
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        for hf_name, t in new_tensors.items():
            dims, typ, off = idx[MAP[hf_name]]
            b = bf16_bytes(t); f.seek(start + off); old = f.read(len(b))
            assert len(old) == len(b); f.seek(start + off); f.write(b)

def main():
    base_g = os.path.join(G, "base-bf16.gguf")
    ver, idx, start = read_gguf_index(base_g)
    base = load_file(glob.glob(os.path.join(ROOT, "models/base/*.safetensors"))[0])
    edited = load_file(glob.glob(os.path.join(ROOT, "models/edited/*.safetensors"))[0])
    report = {"gguf_version": ver, "checks": {}}
    with open(base_g, "rb") as f:
        for hf_name, g_name in MAP.items():
            dims, typ, off = idx[g_name]; b = bf16_bytes(base[hf_name])
            f.seek(start + off); same = f.read(len(b)) == b
            report["checks"][g_name] = {"dims": dims, "ggml_type": typ, "bytes_equal_base_hf": same}
            print(g_name, dims, "type", typ, "equal to HF base:", same, flush=True)
            assert typ == 30 and same, "layout mismatch: do not patch"   # 30 = GGML_TYPE_BF16
    if not os.path.exists(os.path.join(G, "edited-bf16.gguf")):
        patch(base_g, os.path.join(G, "edited-bf16.gguf"), {k: edited[k] for k in MAP}, idx, start)
    print("wrote edited-bf16.gguf", flush=True)
    # random-direction controls: same layers, same 64 columns, same per-column coefficients and same
    # Frobenius norm as the real edit; only the 1024-dim output direction u is random (fresh per layer).
    diff = json.load(open(os.path.join(ROOT, "out", "diff.json")))
    report["random_dir"] = {}
    for seed in (0, 1, 2):
        new = {}
        for li, t in enumerate(diff["changed_tensors"]):
            k = t["name"]; w = base[k].to(torch.float64).clone()
            cols = torch.tensor(t["changed_columns"]); coefs = torch.tensor(t["col_coefs_signed"], dtype=torch.float64)
            g = torch.Generator().manual_seed(seed * 10 + li)
            u = torch.randn(w.shape[0], generator=g, dtype=torch.float64); u /= u.norm()
            delta = torch.outer(u, coefs); delta *= t["frobenius"] / delta.norm()
            w[:, cols] += delta
            new[k] = w.to(torch.bfloat16)
            report["random_dir"].setdefault(seed, {})[k] = {"changed": int((new[k] != base[k]).sum()),
                "fro_change": float((new[k].float() - base[k].float()).norm()),
                "cos_with_real_u": None}
        patch(base_g, os.path.join(G, f"random_dir_s{seed}-bf16.gguf"), new, idx, start)
        print("wrote random_dir", seed, report["random_dir"][seed], flush=True)
    report["edited_fro_change"] = {k: float((edited[k].float() - base[k].float()).norm()) for k in MAP}
    json.dump(report, open(os.path.join(ROOT, "out", "variants.json"), "w"), indent=1)
    print(json.dumps(report["edited_fro_change"]))

if __name__ == "__main__":
    main()
