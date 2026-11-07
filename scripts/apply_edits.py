"""Apply the weight edits from "Two Cures Beat One" to Qwen3.5-0.8B (Hugging Face safetensors).

  opus   : remove the doubt direction u from every residual-writing matrix in layers 12-23:
           W <- bf16( (I - u u^T) W )   on self_attn.o_proj / linear_attn.out_proj / mlp.down_proj
  merged : opus edit, then add Astra's exact weight difference (Takehi et al. 2026) on top:
           W <- bf16( opus(W) + (W_astra - W_base) )

Only the changed tensors are written (a small safetensors file); copy it over the base checkpoint's tensors,
or pass --full to write a complete model directory.

usage:
  python apply_edits.py --base Qwen/Qwen3.5-0.8B --edit opus   --out opus_edit.safetensors
  python apply_edits.py --base Qwen/Qwen3.5-0.8B --edit merged --astra RikiyaT/Qwen3.5-0.8B-weight-space-programming --out merged_edit.safetensors
  add --check REFERENCE.safetensors to assert the result is bit-identical to a reference file.
"""
import argparse, glob, json, os, re, shutil
import torch

from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = range(12, 24)
TARGET = re.compile(r"\.layers\.(\d+)\.(self_attn\.o_proj|linear_attn\.out_proj|mlp\.down_proj)\.weight$")


def local_dir(name):
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    return snapshot_download(name, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"])


DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
          "I64": torch.int64, "I32": torch.int32, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def header(p):
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little"); hdr = json.loads(f.read(n))
    return hdr, 8 + n


def read_safetensors(p, keys=None):
    """Plain reader: 8-byte header length, JSON header, raw little-endian bytes. Reads only `keys` if given
    (avoids loading the whole checkpoint, and the mmap crashes seen on Windows under memory pressure)."""
    hdr, base = header(p); out = {}
    with open(p, "rb") as f:
        for k, v in hdr.items():
            if k == "__metadata__" or (keys is not None and k not in keys):
                continue
            s, e = v["data_offsets"]; f.seek(base + s)
            t = torch.frombuffer(bytearray(f.read(e - s)), dtype=DTYPES[v["dtype"]]) if e > s else torch.empty(0, dtype=DTYPES[v["dtype"]])
            out[k] = t.reshape(v["shape"])
    return out, hdr.get("__metadata__")


def index(d):
    """key -> safetensors file, for every tensor in a checkpoint directory."""
    return {k: p for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))) for k in header(p)[0] if k != "__metadata__"}


def load(ix, keys):
    out = {}
    for p in set(ix[k] for k in keys):
        out.update(read_safetensors(p, set(k for k in keys if ix[k] == p))[0])
    return out


def raw(p, k):
    hdr, base = header(p); s, e = hdr[k]["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + s); return f.read(e - s)


def load_u(path):
    if path.endswith(".json"):
        return torch.tensor(json.load(open(path))["u"], dtype=torch.float32).double()
    return torch.load(path).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--edit", choices=["opus", "merged"], required=True)
    ap.add_argument("--astra", default="RikiyaT/Qwen3.5-0.8B-weight-space-programming")
    ap.add_argument("--u", default=os.path.join(HERE, "..", "edits", "doubt_direction_u.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="")
    ap.add_argument("--full", action="store_true", help="write a complete model directory at --out instead")
    a = ap.parse_args()

    u = load_u(a.u)  # used exactly as saved (float32 values, norm 1 to ~1e-8); do not renormalise: the edit was built this way
    bdir = local_dir(a.base); BI = index(bdir)
    keys = [k for k in BI if (m := TARGET.search(k)) and int(m.group(1)) in LAYERS]
    assert len(keys) == 24, f"expected 24 target tensors, found {len(keys)}"
    B = load(BI, keys); new = {}
    for k in keys:
        W = B[k].double()
        new[k] = (W - torch.outer(u, u @ W)).to(torch.bfloat16)
    if a.edit == "merged":
        AI_ = index(local_dir(a.astra)); assert set(AI_) == set(BI)
        changed = [k for k in BI if raw(BI[k], k) != raw(AI_[k], k)]
        print("Astra changes", changed)
        Bc = load(BI, changed); Ac = load(AI_, changed)
        for k in changed:
            start = new[k] if k in new else Bc[k]
            new[k] = (start.double() + (Ac[k].double() - Bc[k].double())).to(torch.bfloat16)
    print(f"{a.edit}: {len(new)} edited tensors")

    if a.check:
        ref = read_safetensors(a.check)[0]
        assert set(ref) == set(new), (sorted(set(ref) ^ set(new)))[:5]
        bad = [k for k in new if not torch.equal(ref[k], new[k])]
        assert not bad, f"differs from reference: {bad[:5]}"
        print("bit-identical to", a.check)

    if a.full:
        os.makedirs(a.out, exist_ok=True)
        for p in glob.glob(os.path.join(bdir, "*")):
            if os.path.isfile(p) and not p.endswith(".safetensors"):
                shutil.copy(p, a.out)
        for p in sorted(glob.glob(os.path.join(bdir, "*.safetensors"))):
            sd, meta = read_safetensors(p)  # whole shard; needs ~2x model size in RAM
            sd = {k: (new[k] if k in new else v) for k, v in sd.items()}
            save_file(sd, os.path.join(a.out, os.path.basename(p)), metadata=meta)
        print("wrote model directory", a.out)
    else:
        save_file({k: v.contiguous() for k, v in new.items()}, a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
