# Benchmark entry points

所有 Python benchmark 都从项目根目录运行，输出统一写入
`artifacts/results/` 或 Spark 任务的 `$SPARK_JOB_DIR`。

| 脚本 | 作用 | 典型输出 |
|---|---|---|
| `bench_qwen35.py` | 固定输入/输出长度的 Qwen3.5 延迟与并发吞吐 | `qwen35_benchmark.json` |
| `bench_latency_throughput.py` | 全 VLX TTFT、TPOT、E2E 和并发吞吐 | `latency_throughput.json` |
| `bench_module_latency.py` | 从服务 profiler 日志提取模块耗时 | `module_latency.json` |
| `bench_gdn_prefill.py` | FlashInfer/Triton GDN 数值与 kernel 延迟 | `gdn_prefill.json` |
| `bench_humming_nvfp4.py` | Humming NVFP4 GEMM microbenchmark | JSON |
| `bench_lm_head_precision.py` | LM Head token 精度 canary | JSON |
| `bench_camera_resolutions.py` | 480P/720P/1080P C-RADIO 路由与 E2E | `camera_resolutions.json` |
| `bench_vlx_long_decode.py` | 全 VLX 256/512 token 长 decode | `long_decode.json` |
| `compare_qwen35_accuracy.py` | 比较两份生成结果的 token 一致性 | JSON |
| `compare_qwen35_benchmarks.py` | 汇总 baseline/optimized 性能差异 | JSON + Markdown |

示例：

```bash
python3 benchmarks/bench_qwen35.py \
  --input-tokens 2200 --output-tokens 27 \
  --out artifacts/results/qwen35_2200_27.json

python3 benchmarks/bench_latency_throughput.py \
  --fixed-decode \
  --out artifacts/results/vlx_latency_throughput.json
```

推荐使用 `scripts/` 下的封装入口提交 DGX Spark 任务，它们会记录命令、环境、
GPU 快照、stdout/stderr 和退出状态。
