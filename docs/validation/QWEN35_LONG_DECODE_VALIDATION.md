# Qwen3.5 long-decode optimization validation

Date: 2026-09-16  
GPU: NVIDIA DGX Spark / GB10 / SM121  
Workload: text-only, exactly 2200 input tokens, exactly 256 or 512 output
tokens, `ignore_eos=true`, no vision path

## Variants

Baseline:

- ModelOpt NVFP4 W4A4 transformer linears
- FlashInfer-CUTLASS GEMM
- BF16 LM head
- Triton/FLA GDN prefill
- FlashAttention 2

Optimized:

- ModelOpt NVFP4 W4A4 transformer linears
- FlashInfer-CUTLASS GEMM
- Humming NVFP4 W4A16 LM head
- FlashInfer GDN prefill
- FlashAttention 2

Every concurrency shape was warmed before measurement.

## 2200 input / 256 output

| Concurrency | Baseline E2E | Optimized E2E | E2E reduction | Baseline output tok/s | Optimized output tok/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2752.12 ms | 1711.49 ms | 37.81% | 93.01 | 149.57 | 60.80% |
| 2 | 2516.33 ms | 1774.65 ms | 29.47% | 203.41 | 288.43 | 41.80% |
| 4 | 2808.59 ms | 2080.27 ms | 25.93% | 363.69 | 491.83 | 35.23% |
| 8 | 3380.53 ms | 2503.35 ms | 25.95% | 603.06 | 815.33 | 35.20% |

Sequential TPOT decreased from 10.419 ms to 6.383 ms (38.74%). Sequential
TTFT decreased from 95.18 ms to 83.94 ms (11.81%).

## 2200 input / 512 output

| Concurrency | Baseline E2E | Optimized E2E | E2E reduction | Baseline output tok/s | Optimized output tok/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5429.13 ms | 3363.02 ms | 38.06% | 94.30 | 152.24 | 61.43% |
| 2 | 4877.54 ms | 3415.10 ms | 29.98% | 209.91 | 299.81 | 42.83% |
| 4 | 5304.08 ms | 3902.06 ms | 26.43% | 385.67 | 524.63 | 36.03% |
| 8 | 6136.45 ms | 4469.90 ms | 27.16% | 665.48 | 914.35 | 37.40% |

Sequential TPOT decreased from 10.439 ms to 6.416 ms (38.54%). Sequential
TTFT decreased from 94.93 ms to 84.44 ms (11.05%).

## Conclusion

The optimized Qwen3.5 stack remains effective at both long-decode lengths.
Unlike the 27-token workload, the LM-head saving is amortized over hundreds of
decode steps: sequential output throughput improves by about 61% and E2E by
about 38%. At concurrency 8, output throughput still improves by 35-37%.

The benchmark is performance-only. It uses a synthetic repeated-token prompt
and forced decode length so it does not measure response quality.

Raw results:

```text
.spark-results/20260916T031949Z-qwen35-long-decode-ab-78175
```
