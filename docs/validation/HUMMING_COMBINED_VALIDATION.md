# Qwen3.5 Humming + W4A16 validation on DGX Spark

Date: 2026-09-15  
GPU: NVIDIA DGX Spark, SM121  
Humming source: `vllm-project/humming@a74973b5079e42ef861720b62f847ce9d33447f5`  
Workload: text-only completions, exactly 2200 prompt tokens and 27 generated tokens

## Variants

| Variant | Transformer Linear GEMM | LM Head |
|---|---|---|
| Baseline | FlashInfer-CUTLASS NVFP4 W4A4 | BF16 |
| Humming only | Humming ModelOpt NVFP4 W4A4 | BF16 |
| LM Head only | FlashInfer-CUTLASS NVFP4 W4A4 | Humming NVFP4 W4A16 |
| Combined | Humming ModelOpt NVFP4 W4A4 | Humming NVFP4 W4A16 |

FlashAttention 2 was held constant. Each candidate ran in an isolated container
alongside the unchanged baseline service. Prefix and multimodal caches were
disabled.

## Performance

### Sequential request

| Variant | E2E (ms) | TTFT (ms) | TPOT (ms) | Requests/s |
|---|---:|---:|---:|---:|
| Baseline | 363.29 | 93.04 | 10.39 | 2.752 |
| Humming only | 351.82 | 110.82 | 9.27 | 2.842 |
| LM Head only | 256.93 | 90.63 | 6.40 | 3.891 |
| Combined | **243.00** | 105.99 | **5.27** | **4.113** |

Relative to baseline, the combined candidate reduces E2E by 33.1% and TPOT by
49.3%, but increases TTFT by 13.9%. The LM Head-only candidate gives most of the
useful gain: E2E -29.3%, TPOT -38.5%, and TTFT -2.6%.

### Concurrency 8

| Variant | E2E (ms) | TTFT (ms) | TPOT (ms) | Requests/s | Output tokens/s |
|---|---:|---:|---:|---:|---:|
| Baseline | 1174.83 | 730.85 | 17.08 | 5.954 | 160.75 |
| Humming only | 1252.56 | 799.94 | 17.41 | 5.678 | 153.29 |
| LM Head only | **1093.82** | **731.05** | 13.95 | **6.377** | **172.18** |
| Combined | 1151.95 | 803.82 | **13.39** | 5.951 | 160.69 |

At concurrency 8, the combined candidate does not improve throughput. The
LM Head-only candidate improves request/output-token throughput by 7.1%.

## Precision canary

Method: temperature 0, fixed seed, 27-token cap, eight Chinese/English prompts
covering explanation, arithmetic, code, translation and reasoning. The metrics
compare candidate generation with the baseline and compare first-token top-5
logprobs. This detects numerical drift; it is not a substitute for a domain
validation set.

| Variant | Exact text | Position-token match | First-token match | First-token top-5 overlap |
|---|---:|---:|---:|---:|
| Humming only | 25.0% | 47.7% | 87.5% | 87.5% |
| LM Head only | **50.0%** | **71.3%** | **100.0%** | **91.7%** |
| Combined | 25.0% | 29.6% | 75.0% | 84.5% |

## Decision

The combined candidate is functional and fast for single-request decode, but it
does not pass the current precision or concurrency-throughput gate. Do not make
it the production default yet.

The best next candidate is FlashInfer-CUTLASS for the transformer linears plus
Humming W4A16 for the LM Head. It preserves first-token decisions on all eight
canaries, supplies most of the sequential speedup, and improves concurrency-8
throughput. Before deployment, run it against the real VLX validation corpus and
set task-level tolerances (for example bbox/answer accuracy), rather than relying
only on token agreement.

## Reproduction

The feature flags are:

```bash
VLLM_NVFP4_GEMM_BACKEND=humming
VLLM_HUMMING_LM_HEAD_W4A16=1
```

Run the combined candidate on the Spark host:

```bash
bash scripts/run-humming-combined-validation.sh 8101 humming 1
```

Run the two ablations:

```bash
bash scripts/run-humming-combined-validation.sh 8101 humming 0
bash scripts/run-humming-combined-validation.sh 8101 flashinfer-cutlass 1
```

Raw results are stored under `.spark-results/20260915T100827Z-humming-combined-v3-84369`
and `.spark-results/20260915T101454Z-humming-ablation-85088`.

