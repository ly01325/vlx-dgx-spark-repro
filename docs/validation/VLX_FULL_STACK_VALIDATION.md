# VLX C-RADIO + Qwen3.5 full-stack benchmark

Date: 2026-09-16  
GPU: NVIDIA DGX Spark / GB10 / SM121  
Workload: 1920x1080 image, 100 bounding boxes, 2271 actual prompt tokens,
exactly 27 generated tokens, prefix and multimodal caches disabled

## Compared deployments

Baseline:

- C-RADIO PyTorch BF16
- Qwen3.5 ModelOpt NVFP4 W4A4 with FlashInfer-CUTLASS GEMM
- BF16 tied LM head
- Triton/FLA GDN prefill
- FlashAttention 2

Optimized:

- C-RADIO TensorRT BF16 dynamic engine (`L=8170` route)
- Qwen3.5 ModelOpt NVFP4 W4A4 with FlashInfer-CUTLASS GEMM
- Humming NVFP4 W4A16 LM head
- FlashInfer GDN prefill
- FlashAttention 2

All packed/concurrency shapes were warmed before measurement. The request set
used `min_tokens=max_tokens=27` and `ignore_eos=true`, so both deployments did
the same amount of decode work.

## Sequential result

| Metric | Baseline | Optimized | Improvement |
|---|---:|---:|---:|
| E2E | 1203.57 ms | 934.58 ms | 22.35% lower |
| TTFT | 934.72 ms | 771.59 ms | 17.45% lower |
| TPOT | 10.341 ms | 6.269 ms | 39.38% lower |
| Request throughput | 0.831 req/s | 1.070 req/s | 28.78% higher |
| Output throughput | 22.43 tok/s | 28.89 tok/s | 28.78% higher |

## Module breakdown

| Module | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| C-RADIO aux vision tower | 437.78 ms | 285.27 ms | 34.84% |
| Multimodal embedding total | 793.82 ms | 642.05 ms | 19.12% |
| Qwen language-model prefill forward | 77.61 ms | 70.80 ms | 8.78% |
| Serving prefill total | 886.80 ms | 725.13 ms | 18.23% |
| Serving decode total | 270.69 ms | 166.21 ms | 38.60% |
| Client E2E | 1200.15 ms | 933.98 ms | 22.18% |

`lm_decode_forward_mean` excludes LM-head/logits work and was statistically
flat (5.107 vs 5.131 ms). The W4A16 LM-head benefit appears in serving decode,
which includes logits generation.

## Concurrency sweep

| Concurrency | Baseline req/s | Optimized req/s | Throughput gain | Baseline E2E | Optimized E2E | E2E reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.832 | 1.070 | 28.52% | 1201.39 ms | 934.80 ms | 22.19% |
| 2 | 0.967 | 1.203 | 24.41% | 2068.38 ms | 1662.57 ms | 19.62% |
| 4 | 1.138 | 1.298 | 14.01% | 3511.64 ms | 3080.58 ms | 12.27% |
| 8 | 1.244 | 1.345 | 8.15% | 6422.33 ms | 5939.75 ms | 7.51% |

Every measured request succeeded and every baseline/optimized request reported
2271 prompt tokens and 27 completion tokens.

## Reproduction

Run:

```bash
scripts/spark-submit --sync --name vlx-full-stack-fixed27-ab -- \
  scripts/run-vlx-full-stack-ab.sh
```

Raw result directory:

```text
.spark-results/20260916T023145Z-vlx-full-stack-fixed27-ab-73439
```

The earlier natural-stop diagnostic run is intentionally not used for the
performance conclusion because optimized responses sometimes encountered a
stop token before 27 tokens. Its raw directory is
`.spark-results/20260916T022114Z-vlx-full-stack-ab-72426`.
