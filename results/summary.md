# Results

thinking on; temperature 0.6, top_p 0.95, top_k 20; presence penalty 1.5 over the whole output (repeat_last_n 32768); max 32,768 new tokens; one sample per problem, per-problem seed shared by all models; llama.cpp b11960, bf16 weights

## MATH-500, first 100 (seed-0 subset)

| Model | Correct | n | Accuracy | Median tokens | Total tokens | Hit 32k cap |
|---|---|---|---|---|---|---|
| Merged (Astra + Opus) | 55 | 100 | 55.0% | 8,562 | 1.46M | 27 |
| Astra (GPT-6) edit | 47 | 100 | 47.0% | 26,438 | 2.25M | 40 |
| Opus 5.5 edit | 40 | 100 | 40.0% | 32,768 | 2.06M | 58 |
| Base (unedited) | 4 | 70 | 5.7% | 32,768 | 2.19M | 66 |
| Random-direction control | 15 | 76 | 19.7% | 32,768 | 2.13M | 60 |

| Pair | n | Correct | Only first / only second | p (McNemar) | Fewer tokens (first / second) | p (sign) | Total tokens |
|---|---|---|---|---|---|---|---|
| Merged (Astra + Opus) vs Astra (GPT-6) edit | 100 | 55 vs 47 | 17 / 9 | 0.169 | 70 / 10 | 3.2e-12 | 1.46M vs 2.25M |
| Merged (Astra + Opus) vs Opus 5.5 edit | 100 | 55 vs 40 | 19 / 4 | 0.0026 | 49 / 24 | 0.0046 | 1.46M vs 2.06M |
| Opus 5.5 edit vs Astra (GPT-6) edit | 100 | 40 vs 47 | 8 / 15 | 0.21 | 42 / 23 | 0.025 | 2.06M vs 2.25M |
| Merged (Astra + Opus) vs Base (unedited) | 70 | 38 vs 4 | 34 / 0 | 1.16e-10 | 51 / 0 | 8.9e-16 | 1.04M vs 2.19M |
| Astra (GPT-6) edit vs Base (unedited) | 70 | 30 vs 4 | 26 / 0 | 2.98e-08 | 36 / 1 | 5.5e-10 | 1.63M vs 2.19M |
| Opus 5.5 edit vs Base (unedited) | 70 | 27 vs 4 | 24 / 1 | 1.55e-06 | 27 / 1 | 2.2e-07 | 1.52M vs 2.19M |
| Merged (Astra + Opus) vs Random-direction control | 76 | 43 vs 15 | 29 / 1 | 5.77e-08 | 55 / 1 | 1.6e-15 | 1.11M vs 2.13M |
| Random-direction control vs Base (unedited) | 70 | 14 vs 4 | 11 / 1 | 0.00635 | 13 / 3 | 0.021 | 1.96M vs 2.19M |

## MATH-500, other 400

| Model | Correct | n | Accuracy | Median tokens | Total tokens | Hit 32k cap |
|---|---|---|---|---|---|---|
| Merged (Astra + Opus) | 206 | 400 | 51.5% | 12,076 | 6.48M | 124 |
| Astra (GPT-6) edit | 187 | 400 | 46.8% | 26,434 | 9.25M | 150 |

| Pair | n | Correct | Only first / only second | p (McNemar) | Fewer tokens (first / second) | p (sign) | Total tokens |
|---|---|---|---|---|---|---|---|
| Merged (Astra + Opus) vs Astra (GPT-6) edit | 400 | 206 vs 187 | 53 / 34 | 0.053 | 258 / 59 | 8e-31 | 6.48M vs 9.25M |

## HumanEval (Docker-executed)

| Model | Correct | n | Accuracy | Median tokens | Total tokens | Hit 32k cap |
|---|---|---|---|---|---|---|
| Merged (Astra + Opus) | 57 | 164 | 34.8% | 812 | 0.32M | 3 |
| Astra (GPT-6) edit | 51 | 164 | 31.1% | 899 | 0.28M | 0 |
| Base (unedited) | 47 | 164 | 28.7% | 1,254 | 1.15M | 29 |

| Pair | n | Correct | Only first / only second | p (McNemar) | Fewer tokens (first / second) | p (sign) | Total tokens |
|---|---|---|---|---|---|---|---|
| Merged (Astra + Opus) vs Astra (GPT-6) edit | 164 | 57 vs 51 | 17 / 11 | 0.345 | 90 / 74 | 0.24 | 0.32M vs 0.28M |
| Merged (Astra + Opus) vs Base (unedited) | 164 | 57 vs 47 | 27 / 17 | 0.174 | 116 / 47 | 6.5e-08 | 0.32M vs 1.15M |
| Astra (GPT-6) edit vs Base (unedited) | 164 | 51 vs 47 | 21 / 17 | 0.627 | 115 / 49 | 2.7e-07 | 0.28M vs 1.15M |

## MBPP (Docker-executed)

| Model | Correct | n | Accuracy | Median tokens | Total tokens | Hit 32k cap |
|---|---|---|---|---|---|---|
| Merged (Astra + Opus) | 97 | 500 | 19.4% | 790 | 0.45M | 0 |
| Astra (GPT-6) edit | 92 | 500 | 18.4% | 922 | 0.67M | 1 |

| Pair | n | Correct | Only first / only second | p (McNemar) | Fewer tokens (first / second) | p (sign) | Total tokens |
|---|---|---|---|---|---|---|---|
| Merged (Astra + Opus) vs Astra (GPT-6) edit | 500 | 97 vs 92 | 39 / 34 | 0.64 | 334 / 166 | 4.7e-14 | 0.45M vs 0.67M |

## MMLU-Pro, 200 sampled (seed 0)

| Model | Correct | n | Accuracy | Median tokens | Total tokens | Hit 32k cap |
|---|---|---|---|---|---|---|
| Merged (Astra + Opus) | 90 | 200 | 45.0% | 3,439 | 2.10M | 37 |
| Astra (GPT-6) edit | 78 | 200 | 39.0% | 15,670 | 3.46M | 27 |
| Base (unedited) | 17 | 118 | 14.4% | 32,768 | 3.31M | 96 |

| Pair | n | Correct | Only first / only second | p (McNemar) | Fewer tokens (first / second) | p (sign) | Total tokens |
|---|---|---|---|---|---|---|---|
| Merged (Astra + Opus) vs Astra (GPT-6) edit | 200 | 90 vs 78 | 25 / 13 | 0.073 | 157 / 28 | 5.7e-23 | 2.10M vs 3.46M |
| Merged (Astra + Opus) vs Base (unedited) | 118 | 55 vs 17 | 42 / 4 | 5.1e-09 | 107 / 0 | 1.2e-32 | 0.80M vs 3.31M |
| Astra (GPT-6) edit vs Base (unedited) | 118 | 50 vs 17 | 35 / 2 | 1.02e-08 | 96 / 11 | 4.3e-18 | 1.71M vs 3.31M |

## GSM8K, first 100

| Model | Correct | n | Accuracy | Median tokens | Total tokens | Hit 32k cap |
|---|---|---|---|---|---|---|
| Merged (Astra + Opus) | 68 | 100 | 68.0% | 1,675 | 0.44M | 6 |
| Astra (GPT-6) edit | 71 | 100 | 71.0% | 9,840 | 1.28M | 14 |

| Pair | n | Correct | Only first / only second | p (McNemar) | Fewer tokens (first / second) | p (sign) | Total tokens |
|---|---|---|---|---|---|---|---|
| Merged (Astra + Opus) vs Astra (GPT-6) edit | 100 | 68 vs 71 | 7 / 10 | 0.629 | 91 / 5 | 1.6e-21 | 0.44M vs 1.28M |

## MATH-500, all 500

| Pair | Correct | Only first / only second | p (McNemar) | Fewer tokens | p (sign) |
|---|---|---|---|---|---|
| Merged (Astra + Opus) vs Astra (GPT-6) edit | 261 vs 234 of 500 | 70 / 43 | 0.0141 | 328 / 69 | 1.8e-41 |
