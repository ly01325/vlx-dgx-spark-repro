# NVFP4 W4A4 backend re-sweep

Date: 2026-09-16  
GPU: NVIDIA DGX Spark / GB10 / SM121  
Runtime: FlashInfer 0.6.18, CUDA 13.2, CUTLASS DSL 4.7.1  
Workload: exactly 2200 input tokens; exactly 27, 256, or 512 output tokens

Controlled settings: BF16 LM head, FlashInfer GDN prefill, FlashAttention 2.
Only the transformer NVFP4 W4A4 GEMM backend changes.

Candidates:

- FlashInfer-CUTLASS
- vLLM CUTLASS
- FlashInfer-cuDNN
- FlashInfer-TRTLLM
- Humming

FlashInfer-TRTLLM still fails engine initialization on SM121. The other four
backends complete successfully.

## Long decode, concurrency 1

| Output | Backend | E2E | TPOT | Output tok/s |
|---:|---|---:|---:|---:|
| 256 | Humming | 2496.07 ms | 9.388 ms | 102.55 |
| 256 | FlashInfer-cuDNN | 2687.98 ms | 10.197 ms | 95.23 |
| 256 | FlashInfer-CUTLASS | 2698.54 ms | 10.242 ms | 94.86 |
| 256 | vLLM CUTLASS | 2723.15 ms | 10.331 ms | 94.00 |
| 512 | Humming | 4915.72 ms | 9.418 ms | 104.15 |
| 512 | FlashInfer-cuDNN | 5304.79 ms | 10.212 ms | 96.51 |
| 512 | FlashInfer-CUTLASS | 5350.30 ms | 10.298 ms | 95.69 |
| 512 | vLLM CUTLASS | 5375.06 ms | 10.346 ms | 95.25 |

Relative to FlashInfer-CUTLASS, Humming improves sequential output throughput
by 8.1% at 256 output tokens and 8.8% at 512 output tokens. Its TTFT is about
17% worse (102-103 ms vs 87-88 ms).

## Long decode, concurrency 8

| Output | Backend | E2E | TPOT | Output tok/s |
|---:|---|---:|---:|---:|
| 256 | Humming | 3176.62 ms | 10.198 ms | 641.63 |
| 256 | FlashInfer-CUTLASS | 3227.12 ms | 10.792 ms | 631.72 |
| 256 | FlashInfer-cuDNN | 3252.71 ms | 10.883 ms | 626.63 |
| 256 | vLLM CUTLASS | 3294.65 ms | 11.026 ms | 618.77 |
| 512 | Humming | 5718.93 ms | 10.047 ms | 714.40 |
| 512 | FlashInfer-CUTLASS | 5923.88 ms | 10.657 ms | 689.33 |
| 512 | FlashInfer-cuDNN | 5974.85 ms | 10.757 ms | 683.68 |
| 512 | vLLM CUTLASS | 6059.43 ms | 10.907 ms | 674.18 |

Relative to FlashInfer-CUTLASS, Humming improves concurrency-8 output
throughput by 1.6% at 256 tokens and 3.6% at 512 tokens. Humming TTFT is
21-22% worse.

## Decision

Humming is the current pure-performance winner for long decode. It is not the
current production default because the previous Humming-W4A4 precision canary
only achieved 25% exact-text agreement, 47.7% position-token agreement, and
87.5% first-token agreement. This performance sweep does not remove that
precision failure.

Keep FlashInfer-CUTLASS as the production W4A4 backend and Humming W4A16 only
for the LM head. Promote Humming W4A4 only after it passes a task-level VLX
accuracy evaluation with the private trained checkpoint.

The first concurrency-8 run at 27 output tokens includes FlashInfer GDN packed
shape JIT warm-up, so it is excluded from the steady-state backend decision.
Sequential 27-token results show the same tradeoff: Humming is about 2% faster
in E2E but about 16% slower in TTFT.

Raw results:

```text
.spark-results/20260916T034349Z-w4a4-backend-resweep-80088
```
