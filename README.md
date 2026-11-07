# Two Cures Beat One: AI-Written Weight Edits Replicate, Stack, and Think Shorter

**Replicating and stacking weight-space programming.** An independent replication of Takehi et al. (2026),
*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, plus the first test of
**stacking** two weight edits written by different AIs (GPT-6 Astra and Claude Opus 5.5).
By **Ajay Grover** (independent researcher, Superposition).

> **Status: preprint, not peer reviewed.** All MATH-500 results are final, including base and the random-direction
> control. Base on MMLU-Pro, GSM8K and MBPP is still running; `results/summary.md` is regenerated as runs finish.
> Per-problem outputs (all model text) are attached to the GitHub release.

## What this is

Takehi et al. reported that GPT-6 Astra, editing Qwen3.5-0.8B's weights with no training, raised MATH-500 from 7.9% to 48.1%.
We:

1. **Verify** the published edit bit for bit (2 tensors, rank 1, layers 15 and 19).
2. **Replicate** its score: 46.8% on MATH-500 (paper: 48.1%). This requires applying the presence penalty over the
   whole output, as vLLM does. With llama.cpp's default 64-token window, the same edit scores only 28%.
3. Give **Claude Opus 5.5** the same task. It found a different failure (the model doubting a correctly read problem) and
   removed one direction, nearly orthogonal to Astra's, from 24 matrices in layers 12–23.
4. **Merge** the two edits. Merged beats each single edit on hard maths and uses far fewer tokens.

## Headline results (MATH-500, all 500 problems, one sample each)

| | Merged | Astra |
|---|---|---|
| Correct | **261 (52.2%)** | 234 (46.8%) |
| Solved only by this model | **70** | 43 |
| Exact McNemar p | **0.014** | |
| Fewer tokens on (sign test) | **328 problems** | 69 problems (p ≈ 10⁻⁴¹) |
| Total tokens | **7.93M** | 11.50M |

**Controls (MATH-500, first 100, same problems):**

| Model | Correct | Answers cut off at 32k |
|---|---|---|
| Merged (Astra + Opus) | **55** | 27 |
| Astra edit | 47 | 40 |
| Opus edit | 40 | 58 |
| Random-direction edit, same size and structure | 20 | 80 |
| Base (unedited; paper: 7.9%) | 6 | 95 |

A size-matched random edit helps a little (the base model sits near a tipping point), but merged solves 37 problems
it misses against 2 the other way (p = 3 × 10⁻⁹).

Results for HumanEval, MBPP, MMLU-Pro and GSM8K, with paired tests, are in [`results/summary.md`](results/summary.md).

## Contents

| Path | What |
|---|---|
| `edits/doubt_direction_u.json`, `.pt` | Opus's doubt direction u (1,024 float32 values). Use as stored; do not renormalise. |
| `scripts/apply_edits.py` | Builds the Opus or merged edit from the Hugging Face checkpoints; `--check` verifies bit-identity. |
| `scripts/make_variants.py` | Writes edited tensors into a bf16 GGUF of the base model, byte for byte. |
| `scripts/make_random_merged.py` | Builds the size-matched random-direction control. |
| `scripts/diff_weights.py` | Verifies the published Astra edit (changed values, rank). |
| `scripts/run_llama_eval.py`, `scripts/benches.py`, `scripts/eval.py` | Evaluation through llama-server; graders; Docker sandbox for code. |
| `scripts/summarize.py` | Accuracy, tokens and paired tests from the per-problem outputs. |
| `runners/lane.sh` | Runs one model on its own llama-server. |
| `results/` | Summary tables. Per-problem outputs (all model text) are archived separately (see below). |

## Reproduce

```bash
pip install torch safetensors huggingface_hub datasets math-verify numpy
# 1. Build the edits (downloads the base and Astra checkpoints)
python scripts/apply_edits.py --edit opus   --out opus_edit.safetensors
python scripts/apply_edits.py --edit merged --out merged_edit.safetensors
# 2. Evaluate (needs llama.cpp's llama-server on PATH or in LLAMA_SERVER, and Docker for code benchmarks)
LLAMA_SERVER=/path/to/llama-server bash runners/lane.sh merged 8090 6 math500,math500rest
WSP_ROOT=. python scripts/summarize.py
```

Settings used for every run: thinking on; temperature 0.6, top_p 0.95, top_k 20; presence penalty 1.5 with
`repeat_last_n` 32,768; at most 32,768 new tokens; bf16 weights; llama.cpp b11960; one RTX 5080 (16 GB).

## Licences

- Code in this repository: MIT (see `LICENSE`).
- Qwen3.5-0.8B and the Astra edited model are Apache-2.0. Edited weights derived from them are distributed under
  Apache-2.0 with the changes described above (see `NOTICE`).

## Citation

Grover, A. (2026). *Two Cures Beat One: AI-Written Weight Edits Replicate, Stack, and Think Shorter.* Preprint.

Takehi, R., Teoh, J., Zhang, A., Khattab, O. (2026). *Sparks of Weight-Space Programming: LLMs can natively read & write
neural network weights.* https://rikiyat.github.io/blog/weight-space-programming/

## AI-use disclosure

Claude Opus 5.5, run as a separate agent, wrote the second weight edit as the experimental subject. Claude built and ran
the evaluation pipeline and drafted text under the author's direction. Ajay Grover directed the research and is
responsible for all claims.
