# FlashInfer GDN Prefill Validation on DGX Spark

Date: 2026-09-15

## Configuration

- GPU: NVIDIA GB10, compute capability 12.1
- Kernel: `7.0.0-1019-nvidia`
- Driver: 580.178.04
- CUDA: 13.2
- PyTorch: 2.12.0a0+0291f960b6.nv26.04.48445190
- FlashInfer: 0.6.18
- NVIDIA CUTLASS DSL: 4.7.1 (CUDA 13 package)
- Model: `vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct`
- Other backends: FlashInfer CUTLASS NVFP4 GEMM and FlashAttention 2
- Workload: 2200 input tokens, exactly 27 output tokens, caches disabled

The server log confirms that the candidate selected the FlashInfer GDN prefill
kernel on SM121. The control selected Triton/FLA. All other tested settings were
the same.

## GDN Microbenchmark

| Sequence layout | FlashInfer | Triton/FLA | Speedup |
| --- | ---: | ---: | ---: |
| 128 | 0.1260 ms | 0.0925 ms | 0.73x |
| 512 | 0.2489 ms | 0.1812 ms | 0.73x |
| 2200 | 0.5652 ms | 0.9296 ms | 1.64x |
| packed 700 + 1500 | 0.3636 ms | 0.9749 ms | 2.68x |

FlashInfer is not beneficial for short sequences, but the target 2200-token
prefill and packed workloads are materially faster.

## Numerical Comparison

At 2200 tokens:

- Output: max absolute error 9.16e-5, RMSE 7.98e-6, cosine 0.9999856
- Final state: max absolute error 6.80e-4, RMSE 7.80e-5, cosine 0.9999899

For packed 700 + 1500 tokens:

- Output: max absolute error 9.16e-5, RMSE 7.93e-6, cosine 0.9999856
- Final state: max absolute error 7.23e-4, RMSE 7.72e-5, cosine 0.9999900

These differences are consistent with alternate BF16 kernel accumulation.

## End-to-End A/B

### Sequential request

| Metric | FlashInfer | Triton/FLA | Change |
| --- | ---: | ---: | ---: |
| Request throughput | 2.815 req/s | 2.736 req/s | +2.85% |
| E2E mean | 355.10 ms | 365.33 ms | -2.80% |
| TTFT mean | 87.25 ms | 94.79 ms | -7.96% |
| TPOT mean | 10.302 ms | 10.405 ms | -0.99% |

### Concurrency 8, second pass in the same server process

| Metric | FlashInfer | Triton/FLA | Change |
| --- | ---: | ---: | ---: |
| Request throughput | 8.125 req/s | 7.386 req/s | +10.01% |
| Input throughput | 17,875 tok/s | 16,249 tok/s | +10.01% |
| E2E mean | 817.93 ms | 913.87 ms | -10.50% |
| TTFT mean | 408.89 ms | 483.98 ms | -15.51% |
| TPOT mean | 15.732 ms | 16.534 ms | -4.85% |

The second pass is used for the steady-state comparison. The first FlashInfer
concurrency-8 pass was dominated by JIT/shape warm-up and achieved only about
2.43 req/s. After the same process had seen the shape, it reached 8.13 req/s.

## Recommendation

Enable FlashInfer GDN prefill for this 2200-token target, with these conditions:

1. Ship FlashInfer 0.6.18, CUTLASS DSL 4.7.1 CUDA 13, and a compatible
   Apache TVM-FFI version together.
2. Start vLLM with `--gdn-prefill-backend flashinfer`.
3. Before admitting traffic, run warm-up requests that cover the production
   sequence and packed/concurrency shapes. A single concurrency-1 warm-up is
   insufficient.
4. Retain `--gdn-prefill-backend triton` as the rollback switch and consider
   Triton for workloads dominated by sequences of 512 tokens or fewer.

## Result Artifacts

- FlashInfer full run: `.spark-results/20260915T153000Z-gdn-flashinfer-steady-retest-17292`
- Triton full run: `.spark-results/20260915T153401Z-gdn-triton-steady-retest-17634`
- First successful microbenchmark: `.spark-results/20260915T152045Z-gdn-sm121-final-bench-v2-16414`

Text-level live-server comparison was skipped because the pre-existing baseline
container on port 8000 was only running `sleep infinity`. Kernel output and
recurrent final state were compared directly against Triton/FLA instead.
