"""Random-direction control for the MERGED model (gap 2).
merged = Astra edit (layers 15/19 down_proj, 64 cols) + Opus edit (W <- (I - u u^T) W on attn-out and down_proj, layers 12-23).
random_merged_s{S} = random_dir_s{S} (Astra-matched random edit, same cols/coefs/norm, random output direction)
                     + the Opus operation with a seeded random unit direction r instead of the doubt direction u.
Steps, each asserted before anything is written:
  1. map every HF key in merged_edit.safetensors to the GGUF tensor whose bytes equal it in merged-bf16.gguf;
  2. confirm the composition rule (bit-exact): merged == bf16( (I - u u^T) W_base + (W_astra - W_base) );
  3. confirm random_dir_s{S}.gguf differs from base only in the two Astra tensors;
  4. write random_merged_s{S}-bf16.gguf: same rule with r and the random Astra-matched delta:
     bf16( (I - r r^T) W_base + (W_random_dir - W_base) ).
usage: make_random_merged.py [seed]"""
import glob, os, shutil, sys
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_variants as mv

ROOT = os.environ.get("WSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); G = os.path.join(ROOT, "gguf")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
U = torch.load(os.path.join(ROOT, "edits", "doubt_direction_u.pt")).double()


def gguf_tensor(path, idx, start, name, shape):
    dims, typ, off = idx[name]
    assert typ == 30, (name, typ)
    n = 1
    for d in dims: n *= d
    with open(path, "rb") as f:
        f.seek(start + off); b = f.read(2 * n)
    return torch.frombuffer(bytearray(b), dtype=torch.bfloat16).reshape(shape)


def main():
    mg, bg, rg = (os.path.join(G, f) for f in ("merged-bf16.gguf", "base-bf16.gguf", f"random_dir_s{SEED}-bf16.gguf"))
    _, idx, start = mv.read_gguf_index(mg)
    _, idx_b, start_b = mv.read_gguf_index(bg)
    _, idx_r, start_r = mv.read_gguf_index(rg)
    assert idx.keys() == idx_b.keys() == idx_r.keys() and start == start_b == start_r
    fm = safe_open(os.path.join(ROOT, "models", "merged_edit.safetensors"), "pt")
    fb = safe_open(glob.glob(os.path.join(ROOT, "models", "base", "*.safetensors"))[0], "pt")
    fe = safe_open(glob.glob(os.path.join(ROOT, "models", "edited", "*.safetensors"))[0], "pt")

    # 1. name map by exact bytes
    gmap = {}
    for k in fm.keys():
        L = int(k.split(".layers.")[1].split(".")[0]); t = fm.get_tensor(k); want = mv.bf16_bytes(t)
        numel = lambda g: int(torch.tensor(idx[g][0]).prod())
        hits = [g for g in idx if g.startswith(f"blk.{L}.") and idx[g][1] == 30 and numel(g) == t.numel()
                and mv.bf16_bytes(gguf_tensor(mg, idx, start, g, t.shape)) == want]
        assert len(hits) == 1, (k, hits)
        gmap[k] = hits[0]
    print("1. mapped", len(gmap), "tensors, e.g.", list(gmap.items())[:3])

    # 2. composition rule: merged == bf16( bf16((I - u u^T) W_base) + (W_astra - W_base) )  [opus edit saved in bf16, then astra delta added]
    worst = 0.0
    for k in fm.keys():
        B = fb.get_tensor(k).double(); E = fe.get_tensor(k).double()
        opus = (B - torch.outer(U, U @ B)).to(torch.bfloat16).double()
        pred = (opus + (E - B)).to(torch.bfloat16)
        diff = (pred.float() - fm.get_tensor(k).float()).abs().max().item(); worst = max(worst, diff)
    print(f"2. max |pred - merged| over 24 tensors = {worst:.3g}")
    assert worst == 0.0, "composition rule does not reproduce merged exactly; stop"

    # 3. random_dir_s{SEED} differs from base only in the Astra tensors
    astra_g = set(mv.MAP.values()); changed = []
    for g in gmap.values():
        k = [h for h, gg in gmap.items() if gg == g][0]; shp = fm.get_tensor(k).shape
        if not torch.equal(gguf_tensor(rg, idx, start, g, shp), gguf_tensor(bg, idx, start, g, shp)):
            changed.append(g)
    print("3. random_dir tensors differing from base among the 24:", changed)
    assert set(changed) <= astra_g

    # 4. build: same operation with a seeded random unit direction
    gen = torch.Generator().manual_seed(1000 + SEED)
    r = torch.randn(U.shape[0], generator=gen, dtype=torch.float64); r /= r.norm()
    print(f"4. random direction seed {1000 + SEED}, cos(r, doubt u) = {float(r @ U):+.4f}")
    dst = os.path.join(G, f"random_merged_s{SEED}-bf16.gguf"); tmp = dst + ".tmp"
    shutil.copyfile(rg, tmp)
    with open(tmp, "r+b") as f:
        for k, g in gmap.items():
            shp = fm.get_tensor(k).shape
            B = gguf_tensor(bg, idx, start, g, shp).double(); R = gguf_tensor(rg, idx, start, g, shp).double()
            ctl = (B - torch.outer(r, r @ B)).to(torch.bfloat16).double()
            new = (ctl + (R - B)).to(torch.bfloat16)
            b = mv.bf16_bytes(new); f.seek(start + idx[g][2]); assert len(f.read(len(b))) == len(b)
            f.seek(start + idx[g][2]); f.write(b)
            dm = (new.float() - gguf_tensor(bg, idx, start, g, shp).float()).norm().item()
            dr = (fm.get_tensor(k).float() - gguf_tensor(bg, idx, start, g, shp).float()).norm().item()
            print(f"   {g:28s} |control - base| {dm:8.3f}   |merged - base| {dr:8.3f}")
    os.replace(tmp, dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()
