# VLX optimization status and next-step plan

Date: 2026-09-16  
Target: NVIDIA DGX Spark, GB10, SM121

## Recommended deployment today

Language model:

- Transformer linears: ModelOpt NVFP4 W4A4, FlashInfer-CUTLASS backend.
- LM head: Humming NVFP4 W4A16, subject to task-level accuracy sign-off.
- GDN prefill: FlashInfer 0.6.18, with production-shape warm-up.
- Attention: FlashAttention 2.

Vision/object path:

- C-RADIO blocks: TensorRT BF16 for validated 480P and 1080P token lengths.
- 720P: PyTorch fallback for latency-first deployment.
- Every other resolution: PyTorch fallback.
- ROI Align, SimpleFPN, CPE, and object projector remain in PyTorch.

## Confirmed optimizations

### 1. Qwen3.5 BF16 to NVFP4 W4A4

This is the foundational language-model optimization.

| Metric | BF16 | NVFP4 W4A4 | Change |
|---|---:|---:|---:|
| Full VLX E2E | 1513 ms | 1189 ms | 21% lower |
| LM prefill | 125.8 ms | 77.6 ms | 38% lower |
| LM decode GPU sum | 404.8 ms | 132.1 ms | 67% lower |
| Client TPOT | 17.14 ms | 8.68 ms | 49% lower |
| GPU weight use | 5.17 GiB | 3.34 GiB | 35% lower |

Status: enabled. Current production backend is FlashInfer-CUTLASS.

### 2. W4A4 GEMM backend selection

Current-version sweep: FlashInfer 0.6.18, CUDA 13.2, CUTLASS DSL 4.7.1;
BF16 LM head, FlashInfer GDN, and FA2 held constant.

Sequential long decode:

| Output | Backend | E2E | Output tok/s |
|---:|---|---:|---:|
| 256 | Humming | 2496.07 ms | 102.55 |
| 256 | FlashInfer-CUTLASS | 2698.54 ms | 94.86 |
| 256 | FlashInfer-cuDNN | 2687.98 ms | 95.23 |
| 256 | vLLM CUTLASS | 2723.15 ms | 94.00 |
| 512 | Humming | 4915.72 ms | 104.15 |
| 512 | FlashInfer-CUTLASS | 5350.30 ms | 95.69 |
| 512 | FlashInfer-cuDNN | 5304.79 ms | 96.51 |
| 512 | vLLM CUTLASS | 5375.06 ms | 95.25 |

Humming W4A4 is the pure-performance winner: 8.1-8.8% more sequential output
throughput than FlashInfer-CUTLASS for 256/512 tokens. At concurrency 8 its
gain is 1.6-3.6%, but TTFT is 17-22% worse.

Humming W4A4 is not the default because its precision canary achieved only
25% exact text, 47.7% position-token match, and 87.5% first-token match.
FlashInfer-TRTLLM still fails initialization on SM121.

Status: FlashInfer-CUTLASS retained as the accuracy-safe default. Humming W4A4
is an experimental candidate pending task-level accuracy validation.

### 3. Humming W4A16 LM head

With transformer linears kept on FlashInfer-CUTLASS, changing only the LM head
from BF16 to Humming W4A16 produced most of the useful short-decode gain:

- 2200 -> 27 sequential E2E: 29.3% lower.
- TPOT: 38.5% lower.
- TTFT: 2.6% lower.
- Concurrency-8 throughput: 7.1% higher.

Precision canary: 50% exact text, 71.3% position-token match, 100% first-token
match, and 91.7% first-token top-5 overlap.

Status: performance-enabled candidate; production release still requires the
private VLX task-level accuracy gate.

### 4. FlashInfer GDN prefill

At 2200 prompt tokens:

- GDN kernel: 0.930 ms -> 0.565 ms, 1.64x.
- Packed 700+1500 shape: 0.975 ms -> 0.364 ms, 2.68x.
- Sequential TTFT: 7.96% lower.
- Sequential E2E: 2.80% lower.
- Steady concurrency-8 throughput: 10.01% higher.
- Steady concurrency-8 E2E: 10.50% lower.

Numerics: output cosine 0.9999856; final-state cosine 0.9999899.

Status: enabled for the 2200-token target. Production must warm sequential and
packed concurrency shapes before admitting traffic. Triton remains rollback.

### 5. Best Qwen3.5 stack

Baseline: W4A4 FlashInfer-CUTLASS + BF16 LM head + Triton GDN.  
Optimized: W4A4 FlashInfer-CUTLASS + Humming W4A16 LM head + FlashInfer GDN.

For 2200 -> 27:

| Concurrency | E2E reduction | Throughput gain | TTFT reduction | TPOT reduction |
|---:|---:|---:|---:|---:|
| 1 | 30.70% | 44.28% | 12.31% | 37.35% |
| 2 | 22.30% | 28.72% | 11.17% | 30.20% |
| 4 | 20.45% | 26.12% | 2.53% | 33.78% |
| 8 | 18.98% | 23.88% | 11.17% | 27.12% |

For long decode:

| Output | Concurrency | E2E reduction | Output-throughput gain |
|---:|---:|---:|---:|
| 256 | 1 | 37.81% | 60.80% |
| 256 | 8 | 25.95% | 35.20% |
| 512 | 1 | 38.06% | 61.43% |
| 512 | 8 | 27.16% | 37.40% |

Status: confirmed performance configuration. Long output increases the value
of W4A16 LM-head acceleration.

### 6. C-RADIO TensorRT BF16

One restricted dynamic profile covers token lengths 1600, 3610, and 8170,
corresponding to prepared 480P, 720P, and 1080P inputs. Runtime dispatch is
controlled by an exact allowlist; all other lengths fall back to PyTorch.

| Input | C-RADIO reduction | Full VLX E2E reduction |
|---|---:|---:|
| 480P | 41.77% | 7.38% |
| 720P | 3.27% | -0.59% |
| 1080P | 35.90% | 13.63% |

Standalone C-RADIO block speedups were 1.96x, 1.04x, and 1.59x respectively.
Minimum per-tap cosine across the three shapes was 0.999891.

Status: enable 480P and 1080P. Use PyTorch for 720P in a latency-first policy.

### 7. Full optimized VLX pipeline

1080P image, 100 boxes, actual 2271 prompt tokens, exactly 27 output tokens.
The baseline already uses W4A4 FlashInfer-CUTLASS; the optimized deployment
adds C-RADIO TensorRT, Humming W4A16 LM head, and FlashInfer GDN.

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| E2E | 1203.57 ms | 934.58 ms | 22.35% lower |
| TTFT | 934.72 ms | 771.59 ms | 17.45% lower |
| TPOT | 10.341 ms | 6.269 ms | 39.38% lower |
| Request throughput | 0.831 req/s | 1.070 req/s | 28.78% higher |

Concurrency throughput gains are 28.52%, 24.41%, 14.01%, and 8.15% at
concurrency 1, 2, 4, and 8.

Status: all optimized components run together successfully with zero request
failures in the fixed-27 performance test.

## Rejected, deferred, or conditional options

| Option | Decision | Reason |
|---|---|---|
| Humming W4A4 default | Conditional | Faster long decode, but precision canary failed and TTFT regressed. |
| FlashInfer-TRTLLM W4A4 | Rejected for now | Engine initialization fails on SM121. |
| FlashInfer-cuDNN W4A4 | Not selected | Near-equal sequential result, weaker concurrency balance than FI-CUTLASS. |
| TensorRT at 720P | Disabled for latency-first | Full E2E regressed by 0.59%. |
| MXFP4 dense model | Deferred | Dense MXFP4 W4A4 kernel is absent; current path is W4A16 Marlin or BF16 fallback. |
| W4A8 dense model | Deferred | Current CUTLASS W4A8 path is restricted to SM90, not SM121; requires a new checkpoint. |
| Speculative decode | Low priority | Optimized TPOT is already about 6.4 ms; acceptance overhead may erase benefit. |
| TensorRT ROI/SimpleFPN | Deferred | Dynamic boxes and custom ops increase risk; only about 70 ms remains here. |

## Accuracy and benchmark limitations

- The delivered sanitized checkpoint has a randomly initialized private object
  head. Kernel differential tests are valid, but task-level object accuracy is
  not a release gate without the private trained checkpoint.
- The example image needs 159 generated tokens for the current verbose JSON
  answer. A 27-token limit truncates the answer. Either adopt a compact index
  protocol or benchmark the realistic output-length distribution.
- Humming W4A16 and especially Humming W4A4 change token generation. Performance
  results must not be interpreted as task-accuracy approval.

## Recommended next steps

### Priority 0: establish a real accuracy gate

1. Run the private trained VLX checkpoint on a labeled image set.
2. Track object-label precision/recall, candidate-index accuracy, exact output
   schema validity, and answer-level quality.
3. Compare BF16, current production stack, Humming W4A16, and Humming W4A4.
4. Do not promote Humming W4A4 solely from its 8-9% long-decode speed gain.

### Priority 1: compact the output protocol

Return candidate indexes directly, for example
`枕头:[23],鞋子:[12,14],柜子:[5]`, instead of verbose bbox JSON. This can make a
complete task fit near the original 27-token budget and removes parsing ambiguity.
Then benchmark the measured production output-length distribution.

### Priority 2: overlap the two vision towers

The main vision tower is about 286 ms and optimized C-RADIO is about 285 ms at
1080P. They are currently serialized. Execute them on separate CUDA streams if
memory and downstream synchronization allow it. This is now the largest single
remaining E2E opportunity; validate contention because both towers are compute
heavy.

### Priority 3: optimize the main vision tower

Build a dynamic-shape TensorRT path for the Qwen vision encoder, using the same
480P/720P/1080P profile policy and exact fallback routing. Preserve dynamic
camera resolutions and compare full-VLX E2E, not just engine latency.

### Priority 4: recover Humming W4A4 accuracy selectively

Identify precision-sensitive transformer layers using per-layer output error or
task ablation. Keep sensitive layers on FlashInfer-CUTLASS and route the rest to
Humming. The target is to retain part of the 8-9% sequential long-decode gain
without the observed token-quality regression.

### Priority 5: improve 720P C-RADIO routing

Try a fixed 720P engine or a profile optimized only for L=3610. Keep it only if
full E2E improves beyond measurement noise; otherwise retain PyTorch fallback.

### Priority 6: productionize and warm kernels

- Package TensorRT Python bindings, the SM120 builder resource, FlashInfer
  0.6.18, CUTLASS DSL 4.7.1, TVM-FFI, and Humming in one immutable image.
- Prebuild C-RADIO engines for the deployed TensorRT version and GPU.
- Warm 2200-token sequential and packed concurrency shapes before traffic.
- Keep rollback switches for C-RADIO TRT, FlashInfer GDN, and W4A16 LM head.

## Source reports

- [`PERFORMANCE.md`](PERFORMANCE.md)
- [`HUMMING_COMBINED_VALIDATION.md`](../validation/HUMMING_COMBINED_VALIDATION.md)
- [`GDN_FLASHINFER_SM121_VALIDATION.md`](../validation/GDN_FLASHINFER_SM121_VALIDATION.md)
- [`C_RADIO_TRT_VALIDATION.md`](../validation/C_RADIO_TRT_VALIDATION.md)
- [`VLX_FULL_STACK_VALIDATION.md`](../validation/VLX_FULL_STACK_VALIDATION.md)
- [`QWEN35_LONG_DECODE_VALIDATION.md`](../validation/QWEN35_LONG_DECODE_VALIDATION.md)
- [`W4A4_BACKEND_RESWEEP.md`](../validation/W4A4_BACKEND_RESWEEP.md)
