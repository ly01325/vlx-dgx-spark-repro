# C-RADIO TensorRT dynamic-shape validation on DGX Spark

Date: 2026-09-16  
GPU: NVIDIA GB10, SM121  
TensorRT: 10.16.1.11  
Precision: BF16 engine and BF16 linear I/O

## Result

The C-RADIO 27-block stack was exported with dynamic token length and four
named intermediate outputs (`feat_l06`, `feat_l13`, `feat_l19`, `feat_l26`).
TensorRT recognized all 27 self-attention blocks as fused `_gemm_mha_v2`
patterns. No explicit quadratic attention buffer was selected.

The full VLX A/B held the optimized Qwen3.5 stack constant and changed only
the C-RADIO execution path.

| Metric | PyTorch C-RADIO | TensorRT C-RADIO | Change |
|---|---:|---:|---:|
| Aux vision tower | 434.51 ms | 275.63 ms | -36.57% |
| Multimodal embedding total | 789.07 ms | 629.76 ms | -20.19% |
| Client E2E (module run) | 1083.51 ms | 942.08 ms | -13.05% |
| Client E2E (latency run) | 1082.11 ms | 946.27 ms | -12.55% |

Full VLX throughput:

| Concurrency | Baseline req/s | TensorRT req/s | Gain | Baseline E2E | TensorRT E2E |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.918 | 1.057 | +15.20% | 1089.57 ms | 945.84 ms |
| 2 | 0.869 | 1.000 | +15.14% | 2300.48 ms | 1996.87 ms |
| 4 | 1.175 | 1.294 | +10.14% | 3403.52 ms | 3080.16 ms |
| 8 | 1.282 | 1.349 | +5.28% | 6130.72 ms | 5893.05 ms |

All sequential and concurrency requests completed successfully.

## Dynamic-shape policy

The engine boundary is token-major rather than image-major:

```text
tokens [1, L, 1152] BF16
  -> C-RADIO blocks 0..26
  -> four outputs [1, L, 1152] BF16

L = 10 + (prepared_height / 16) * (prepared_width / 16)
```

Resize, input conditioning, patch embedding, CPE position embedding, ROI Align,
and object pooling stay in PyTorch. This preserves the existing behavior for
arbitrary camera resolutions and avoids putting the `(H,W)`-dependent CPE
interpolation into TensorRT.

Three engines were evaluated:

- `small_dynamic`: `L=266..8169`, optimized at `L=3610`.
- `main_1920x1088`: fixed `L=8170`, the prepared shape for a 1920x1080 camera.
- `universal_dynamic`: `L=266..16394`, optimized at `L=8170`.

The runtime router uses TensorRT only where it is profitable:

- `L < 4096`: PyTorch fallback by default.
- `4096 <= L <= 8169`: small dynamic engine.
- `L == 8170`: dedicated main engine.
- `8171 <= L <= 15000`: universal dynamic engine.
- `L > 15000`, unsupported dtype/layout, or batch greater than one: PyTorch fallback.

The thresholds are environment-configurable. A fallback is intentional: the
universal engine was slower than PyTorch at the smallest and largest tested
shapes, so dynamic-shape support must not mean forcing every shape through TRT.

## Standalone performance by token length

| L | Approximate class | PyTorch BF16 | TensorRT BF16 | Speedup |
|---:|---|---:|---:|---:|
| 266 | very small | 7.93 ms | 8.57 ms | 0.93x (fallback) |
| 3610 | approximately 720p | 80.58 ms | 80.19 ms | 1.00x |
| 8170 | prepared 1920x1088 | 422.96 ms | 268.02 ms | 1.58x |
| 12298 | large camera input | 735.82 ms | 521.39 ms | 1.41x |
| 16394 | 2048x2048 boundary | 772.69 ms | 841.11 ms | 0.92x (fallback) |

The dedicated `L=8170` engine measured 260.64 ms in the Python A/B and
258.88 ms in `trtexec`, slightly faster than the universal engine.

## Numerical validation

All 330 C-RADIO checkpoint tensors loaded with no missing or unexpected
tensors. The four outputs were compared independently at
`L=266,3610,8170,12298,16394`.

- Minimum per-tap cosine similarity: **0.999885**.
- Every tap at every tested length: cosine >= 0.999.
- Tap order was resolved by tensor name, never by TensorRT binding index.

The sanitized checkpoint has a randomly initialized object head, so production
release still requires task-level evaluation with the private trained object
head. The C-RADIO differential metrics remain valid because its 330 tensors are
the official trained RADIO weights.

## Deployment requirements

- The generated plan is specific to TensorRT 10.16.1 and GB10/SM121.
- The current delivery image lacks the SM120 builder resource and TensorRT
  Python bindings. The validation scripts mount the exact matching files from
  the DGX Spark host read-only. A production image should package them.
- `universal_dynamic.bf16.plan` is approximately 833 MB; the dynamic profile's
  activation allocation is approximately 377 MB at its maximum shape.
- PyTorch block weights are retained to provide shape fallback. They can only be
  released if all accepted production resolutions are covered by validated
  engines.

Raw results:

- `.spark-results/20260915T175214Z-aux-trt-universal-v5-30605`
- `.spark-results/20260915T180654Z-vlx-aux-trt-full-ab-v2-32918`

## Restricted camera profile: 480P / 720P / 1080P

A deployment-specific dynamic engine was also built with one TensorRT
optimization profile and a discrete runtime allowlist:

| Camera input | Prepared input | Token length |
|---|---|---:|
| 854x480 | 848x480 | 1600 |
| 1280x720 | 1280x720 | 3610 |
| 1920x1080 | 1920x1088 | 8170 |

The TensorRT profile is `min=1600`, `opt=3610`, `max=8170`. TensorRT itself
can accept intermediate token lengths, but the VLX router only dispatches the
three validated lengths above; every other resolution uses the PyTorch path.
Portrait inputs with the transposed prepared dimensions produce the same token
length and are covered by the block engine because CPE remains outside TRT.

Standalone C-RADIO block validation:

| Token length | PyTorch BF16 | TensorRT BF16 | Speedup |
|---:|---:|---:|---:|
| 1600 | 54.12 ms | 27.65 ms | 1.96x |
| 3610 | 81.23 ms | 78.28 ms | 1.04x |
| 8170 | 420.76 ms | 265.20 ms | 1.59x |

Minimum per-tap cosine similarity across the three shapes was **0.999891**;
all four taps at all shapes exceeded 0.999.

Full VLX A/B, with the optimized Qwen3.5 stack held constant:

| Input | C-RADIO baseline / TRT | C-RADIO reduction | E2E baseline / TRT | E2E reduction |
|---|---:|---:|---:|---:|
| 480P | 66.65 / 38.81 ms | 41.77% | 342.24 / 316.97 ms | 7.38% |
| 720P | 94.93 / 91.82 ms | 3.27% | 424.91 / 427.42 ms | -0.59% |
| 1080P | 435.31 / 279.02 ms | 35.90% | 1095.23 / 945.92 ms | 13.63% |

The engine supports all three requested resolutions. For a latency-first
production policy, set `OMCHAT_AUX_TRT_CAMERA_LENGTHS=1600,8170` so 720P uses
PyTorch; its small block-level gain did not overcome end-to-end dispatch noise
in this five-run test. Keep `1600,3610,8170` when a uniform three-resolution
TensorRT policy is preferred.

Restricted-profile raw results:

- `.spark-results/20260916T020215Z-aux-trt-camera-profile-69770`
- `.spark-results/20260916T020531Z-vlx-aux-trt-camera-ab-70128`
