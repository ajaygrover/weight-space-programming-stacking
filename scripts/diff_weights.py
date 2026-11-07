"""
Phase 1 -- weight diff between the base and edited checkpoints (CPU only).

Usage:  C:/AI/venvs/wsp/Scripts/python.exe scripts/diff_weights.py
Output: out/diff.json  (+ console summary; tee it via run_phase1.ps1)

Compares models/base (Qwen/Qwen3.5-0.8B, commit 2fc06364715b967f1860aea9cf38778875588b17)
with models/edited (RikiyaT/Qwen3.5-0.8B-weight-space-programming), tensor by tensor:
  * key sets, shapes, dtypes
  * changed elements: raw bfloat16 bit diffs, float != 0, and |diff| > 1e-6; max |diff|
  * for changed tensors: which layers / columns / rows changed, totals
  * rank-1 test: SVD of the diff restricted to the changed columns, s1/sum(s),
    cross-layer cosine of the top left-singular vectors
  * sha256 of raw tensor bytes; cross-check the authors' published bit-exact diff
    (logs/exp8_math/runs/p50c/edit: edit.npz + manifest.json)
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT, "models", "base")
EDIT_DIR = os.path.join(ROOT, "models", "edited")
OUT_DIR = os.path.join(ROOT, "out")
PUB_DIR = os.path.join(ROOT, "logs", "exp8_math", "runs", "p50c", "edit")
os.makedirs(OUT_DIR, exist_ok=True)

# sha256 of raw bfloat16 bytes as stated on the edited model card
# https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming (Verifying the weights)
CARD_SHA256 = {
    "model.language_model.layers.15.mlp.down_proj.weight":
        "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
    "model.language_model.layers.19.mlp.down_proj.weight":
        "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
}
CLAIM = {"entries_touched": 131072, "layers": [15, 19], "columns_per_layer": 64,
         "share_pct": 0.016}


def st_file(d):
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            return os.path.join(d, f)
    raise FileNotFoundError(f"no .safetensors file in {d}")


def sha256_raw(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    bpath, epath = st_file(BASE_DIR), st_file(EDIT_DIR)
    print(f"base   file: {bpath}")
    print(f"edited file: {epath}", flush=True)
    fb = safe_open(bpath, framework="pt")
    fe = safe_open(epath, framework="pt")
    kb, ke = set(fb.keys()), set(fe.keys())
    shared = sorted(kb & ke)

    rep = {"script": "scripts/diff_weights.py", "torch": torch.__version__,
           "numpy": np.__version__,
           "base_file": os.path.relpath(bpath, ROOT), "edited_file": os.path.relpath(epath, ROOT),
           "key_stats": {"n_base": len(kb), "n_edited": len(ke), "n_shared": len(shared),
                         "only_in_base": sorted(kb - ke), "only_in_edited": sorted(ke - kb)},
           "dtype_census": {}, "tensor_report": [], "changed_tensors": [],
           }

    total_elems = 0
    dtype_census = {}
    changed_names = []
    D_cache = {}
    for name in shared:
        tb, te = fb.get_tensor(name), fe.get_tensor(name)
        same_dtype = tb.dtype == te.dtype
        ck = str(tb.dtype) if same_dtype else f"{tb.dtype}->{te.dtype}"
        dtype_census[ck] = dtype_census.get(ck, 0) + 1
        if tuple(tb.shape) != tuple(te.shape):
            rep["tensor_report"].append({"name": name, "shape_mismatch": [list(tb.shape), list(te.shape)]})
            changed_names.append(name)
            continue
        total_elems += tb.numel()
        n_bits = None
        if tb.dtype == torch.bfloat16 and same_dtype:
            n_bits = int((tb.view(torch.uint16) != te.view(torch.uint16)).sum())
        if n_bits == 0:
            rep["tensor_report"].append({"name": name, "shape": list(tb.shape),
                                         "dtype": str(tb.dtype), "n_elems": tb.numel(),
                                         "changed_bits": 0, "changed_ne_f32": 0,
                                         "changed_gt1e-6": 0, "max_abs_diff": 0.0})
            continue
        df = te.to(torch.float64) - tb.to(torch.float64)
        ne = int((df != 0).sum())
        gt = int((df.abs() > 1e-6).sum())
        mx = float(df.abs().max())
        if n_bits is None and ne == 0:
            rep["tensor_report"].append({"name": name, "shape": list(tb.shape),
                                         "dtype": str(tb.dtype), "n_elems": tb.numel(),
                                         "changed_bits": 0, "changed_ne_f32": 0,
                                         "changed_gt1e-6": 0, "max_abs_diff": 0.0})
            continue
        changed_names.append(name)
        item = {"name": name, "shape": list(tb.shape), "dtype": str(tb.dtype),
                "n_elems": tb.numel(), "changed_bits": n_bits, "changed_ne_f32": ne,
                "changed_gt1e-6": gt, "max_abs_diff": mx}
        if "mlp.down_proj.weight" in name and tb.dim() == 2:
            m = re.search(r"layers\.(\d+)\.", name)
            item["layer"] = int(m.group(1)) if m else None
            mask = df != 0
            cols = torch.nonzero(mask.any(dim=0)).flatten().tolist()
            item["n_changed_columns"] = len(cols)
            item["changed_columns"] = cols
            item["n_changed_rows_any"] = int(mask.any(dim=1).sum())
            item["entries_touched_full_columns"] = len(cols) * tb.shape[0]
            D = df[:, cols].to(torch.float64)  # rows x changed_cols
            fro = float(D.norm())
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            sv = S.tolist()
            u1 = U[:, 0]
            coef = u1 @ D                    # per-column signed amplitude along u1
            resid = float((D - torch.outer(u1, coef)).norm() / D.norm())
            item["frobenius"] = fro
            item["singular_values_top10"] = [float(x) for x in sv[:10]]
            item["s1_over_sum_s"] = sv[0] / sum(sv)
            item["s1_over_s2"] = sv[0] / sv[1]
            item["rank1_residual_rel"] = resid
            item["near_rank1"] = resid < 0.05 and sv[0] / sum(sv) > 0.95
            item["col_coefs_signed"] = [float(x) for x in coef.tolist()]
            item["u1_top_left_singular_vector"] = [float(x) for x in u1.tolist()]
            D_cache[name] = D
            item["sha256_base_raw"] = sha256_raw(tb)
            item["sha256_edited_raw"] = sha256_raw(te)
            item["sha256_match_model_card"] = {
                "edited": item["sha256_edited_raw"] == CARD_SHA256.get(name),
            }
            rep["changed_tensors"].append(item)
        rep["tensor_report"].append(item if name in changed_names else None)
        if rep["tensor_report"][-1] is None:
            rep["tensor_report"].pop()

    # cross-layer direction check (both down_proj tensors)
    dt = rep["changed_tensors"]
    if len(dt) >= 2:
        u_a = torch.tensor(dt[0]["u1_top_left_singular_vector"], dtype=torch.float64)
        u_b = torch.tensor(dt[1]["u1_top_left_singular_vector"], dtype=torch.float64)
        rep["cross_layer"] = {
            "tensors": [dt[0]["name"], dt[1]["name"]],
            "cosine_signed": float(u_a @ u_b),
            "cosine_abs": float(abs(u_a @ u_b)),  # SVD sign is arbitrary
            "same_direction_near1": abs(float(u_a @ u_b)) > 0.99,
        }

    rep["dtype_census"] = dtype_census

    # shared-direction test: does ONE direction explain both layers' diffs jointly?
    if len(D_cache) == 2:
        (n_a, D_a), (n_b, D_b) = list(D_cache.items())
        D_cat = torch.cat([D_a, D_b], dim=1)
        Uc, Sc, _ = torch.linalg.svd(D_cat, full_matrices=False)
        u_sh = Uc[:, 0]
        def fit_resid(D, u):
            return float((D - torch.outer(u, u @ D)).norm() / D.norm())
        rep["shared_direction_test"] = {
            "cosine_u15_u19": float(torch.tensor(rep["cross_layer"]["cosine_signed"])),
            "s1_over_sum_s_concatenated": float(Sc[0] / Sc.sum()),
            "residual_layer15_with_shared_u": fit_resid(D_a, u_sh),
            "residual_layer19_with_shared_u": fit_resid(D_b, u_sh),
            "residual_layer15_own_u1": rep["changed_tensors"][0]["rank1_residual_rel"],
            "residual_layer19_own_u1": rep["changed_tensors"][1]["rank1_residual_rel"],
            "interpretation": ("own-u1 residuals ~0.2-0.3% => each layer's diff is near rank-1; "
                               "shared-u residuals show whether one single direction fits BOTH layers"),
        }

    layers = sorted({t.get("layer") for t in dt})
    touched = sum(t.get("entries_touched_full_columns", 0) for t in dt)
    bits = sum(t.get("changed_bits") or t.get("changed_ne_f32") for t in dt)
    rep["totals"] = {"total_params_shared": total_elems,
                     "changed_tensors": changed_names,
                     "layers_touched": layers,
                     "entries_touched_full_columns": touched,
                     "entries_changed_bits": bits,
                     "share_of_params_pct_changed_bits": 100.0 * bits / total_elems,
                     "share_of_params_pct_touched": 100.0 * touched / total_elems}
    rep["claims_check"] = {
        "claim": CLAIM,
        "measured_layers_only_15_19": layers == CLAIM["layers"],
        "measured_columns_per_layer_64": all(t.get("n_changed_columns") == 64 for t in dt),
        "measured_entries_touched_equals_claim": touched == CLAIM["entries_touched"],
        "measured_bits_changed": bits,
        "note": ("the claim counts entries the edit TOUCHED (64 columns x 1024 rows x 2 layers); "
                 "entries that round back to the same bfloat16 value are not bit-changed"),
    }

    # cross-check the authors' published bit-exact diff against these checkpoints
    if os.path.isdir(PUB_DIR):
        man = json.load(open(os.path.join(PUB_DIR, "manifest.json")))
        z = np.load(os.path.join(PUB_DIR, "edit.npz"))
        pub = {"manifest_path": os.path.relpath(os.path.join(PUB_DIR, "manifest.json"), ROOT),
               "tensors": []}
        for i, t in enumerate(man["tensors"]):
            name = t["name"]
            base_t = fb.get_tensor(name)
            bits_arr = base_t.view(torch.uint16).numpy().copy().reshape(-1)
            idx = z[f"t{i}_idx"]
            val = z[f"t{i}_val"]
            if val.dtype == np.int16:
                val = val.view(np.uint16)
            bits_arr[idx] = val
            rebuilt = torch.from_numpy(bits_arr).view(torch.bfloat16).reshape(base_t.shape)
            ed = fe.get_tensor(name)
            match = bool(torch.equal(rebuilt.view(torch.uint16), ed.view(torch.uint16)))
            h_base = sha256_raw(base_t)
            h_edit = sha256_raw(ed)
            pub["tensors"].append({
                "name": name,
                "manifest_base_sha256_match": h_base == t["base_sha256"],
                "manifest_edited_sha256_match": h_edit == t["edited_sha256"],
                "model_card_edited_sha256_match": h_edit == CARD_SHA256.get(name),
                "published_idx_val_reproduces_checkpoint_bit_exact": match,
                "changed_entries": int(idx.shape[0]),
            })
        rep["published_diff_crosscheck"] = pub

    json.dump(rep, open(os.path.join(OUT_DIR, "diff.json"), "w"), indent=1)

    print(json.dumps({"key_stats": rep["key_stats"], "dtype_census": rep["dtype_census"],
                      "totals": rep["totals"],
                      "claims_check": rep["claims_check"],
                      "cross_layer": rep.get("cross_layer"),
                      "shared_direction_test": rep.get("shared_direction_test"),
                      "changed": [{k: t[k] for k in
                                   ("name", "shape", "changed_bits", "changed_ne_f32",
                                    "changed_gt1e-6", "max_abs_diff", "n_changed_columns",
                                    "s1_over_sum_s", "s1_over_s2", "rank1_residual_rel")
                                   if k in t} for t in dt]}, indent=1), flush=True)
    print("wrote out/diff.json", flush=True)


if __name__ == "__main__":
    main()
