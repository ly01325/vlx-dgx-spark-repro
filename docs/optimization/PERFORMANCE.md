# vlx-seek-1.5-3B DGX Spark 性能测试总结

日期：2026-09-14  
机器：DGX Spark，NVIDIA GB10 × 1，tensor parallel = 1  
运行时：交付镜像 `vlx_seek1_5_spark:1.0.0_encrypted`，定制 vLLM 0.19.0  
容器：`vlx-seek-15-3b`，host network，端口 `8000`

本文记录原始 BF16 模型与语言侧 NVFP4 量化模型在同一套用户请求上的延迟、吞吐和模块耗时对比。

## 1. 对比对象

| 项目 | 原始模型 | 优化模型 |
|------|----------|----------|
| 目录 | `resources/vlx-seek-1.5-3b-sanitized` | `resources/vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct` |
| served name | `vlx-seek-1.5-3b-sanitized` | `vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct` |
| 精度 | 全模型 BF16 | 语言 Linear：NVFP4 W4A4-direct（group size 16）；视觉塔 / object / embedding / `lm_head`：BF16 |
| GPU 权重占用 | 5.17 GiB | 3.34 GiB（-35%） |
| 量化后端 | 无 | `--quantization modelopt_fp4`，`NvFp4LinearBackend.FLASHINFER_CUTLASS` |

NVFP4 只覆盖 186 个语言 Linear checkpoint（对应 114 个 fused vLLM Linear）。视觉塔、ROI/SimpleFPN、object projector 未被量化，因此预期只加速 LM prefill / decode。

## 2. 测试负载与开关

与 `README.md` 默认场景一致，关闭缓存以测量各阶段真实耗时：

- 图片：`resources/demo_1920_1080.jpeg`（1920×1080）
- Prompt：检测枕头、鞋子、柜子，返回物体索引
- bbox：固定 100 框
- `max_tokens=27`，`temperature=0`
- Prompt tokens = 2271，completion tokens = 27（`finish_reason=length`）
- `NO_PREFIX_CACHE=1` `NO_MM_CACHE=1`
- `OMCHAT_PROFILE_MODEL_PHASES=1` `OMCHAT_PROFILE_OBJECT_FEATURES=1`
- `VLLM_USE_FLASHRT_OPS=0`
- `gpu-memory-utilization=0.3157`（脚本按显存自动计算，两次相同）

串行客户端：预热 3 次 + 正式 15 次，streaming 测 TTFT / TPOT。  
模块拆分：预热 3 次 + 正式 10 次，解析服务端 profiler 日志。  
并发：concurrency = 1 / 2 / 4 / 8。

## 3. 结论

1. NVFP4 把 **LM decode GPU 从 405 ms 降到 132 ms（约 3.1×）**，客户端 TPOT 从 17.1 ms 降到 8.7 ms（-49%）。
2. 端到端只快 **21%**（1513 ms → 1189 ms），因为视觉路径几乎不变。
3. BF16 上 Aux RADIO（431 ms）和 LM decode（405 ms）几乎打平。量化后 decode 不再是第一瓶颈，**Aux RADIO 约占 E2E 的 36%**。
4. 高并发收益有限：c=8 时 rps 仅从 1.14 升到 1.25，视觉编码仍串行排队。

## 4. 端到端延迟与吞吐

串行，预热后 15 次均值：

| 指标 | BF16 | NVFP4 | 变化 |
|------|------|-------|------|
| E2E mean | 1513 ms | 1189 ms | -21% |
| E2E p50 | 1514 ms | 1188 ms | -21% |
| E2E p90 | 1517 ms | 1194 ms | -21% |
| TTFT mean | 1068 ms | 963 ms | -10% |
| TPOT mean | 17.14 ms | 8.68 ms | -49% |
| 单请求 decode | 60.6 tok/s | 119.6 tok/s | +97% |
| 请求吞吐 | 0.66 rps | 0.84 rps | +27% |
| 输出吞吐 | 17.8 tok/s | 22.7 tok/s | +27% |

两次串行结果都很稳：BF16 E2E 极差约 16 ms，NVFP4 约 14 ms。

## 5. 模块耗时

预热后 10 次均值，单位 ms。`roi_extract` 包含 SimpleFPN + ROI + 位置编码 / pooling，没有单独的 SimpleFPN 日志。

| 模块 | 含义 | BF16 | NVFP4 | 变化 |
|------|------|------|-------|------|
| `shared_image_pack` | PIL → CHW tensor | 5.3 | 5.4 | ~0 |
| `main_mm_preprocess` | Qwen ViT 图像预处理 | 17.4 | 17.9 | ~0 |
| `main_vision_tower` | Qwen3.5-2B Vision Tower | 282.7 | 282.6 | ~0 |
| `aux_vision_tower` | C-RADIOv4-SO400M | 430.6 | 434.0 | ~0 |
| `roi_extract` | SimpleFPN + ROI + object 特征 | 71.0 | 70.6 | ~0 |
| `aux_projector` | object MLP | 0.31 | 0.32 | ~0 |
| `multimodal_embed_total` | 多模态 GPU 合计 | 786.6 | 788.3 | ~0 |
| LM prefill forward | 语言 prefill | 125.8 | 77.6 | **-38%** |
| LM decode / token | 语言 decode 单步 | 15.57 | 5.08 | **-67%** |
| LM decode GPU sum | 语言 decode GPU 合计 | 404.8 | 132.1 | **-67%** |
| Engine decode wall | 含采样 / 调度 | 529.8 | 269.2 | **-49%** |
| HTTP E2E | 客户端整请求 | 1510.8 | 1194.3 | **-21%** |

两条视觉塔在本构建中是串行的，没有重叠：Main ViT ≈ 283 ms，Aux RADIO ≈ 431–434 ms。

相对 BF16 端到端 1511 ms 的大致占比：

- Aux RADIO 28.5% → NVFP4 后约 36%
- LM decode GPU 26.8% → NVFP4 后约 11%
- Main ViT 约 19%（两边接近）
- LM prefill 8.3% → 6.5%

## 6. 并发

同一负载，输出仍为 27 token：

| 并发 | BF16 rps | NVFP4 rps | BF16 E2E mean | NVFP4 E2E mean |
|------|----------|-----------|---------------|----------------|
| 1 | 0.66 | 0.84 | 1.52 s | 1.19 s |
| 2 | 0.83 | 0.96 | 2.42 s | 2.09 s |
| 4 | 1.01 | 1.14 | 3.94 s | 3.51 s |
| 8 | 1.14 | 1.25 | 7.01 s | 6.41 s |

并发升高时 TTFT 近似线性变差，decode 单步时间基本不变。限制因素是视觉编码和 prefill 排队，不是 LM decode kernel。

## 7. 如何复现

镜像与定制 vLLM：

```bash
docker load -i vlx_seek1_5.tar.gz
# 容器需 GPU、host network，并挂载本目录到 /workspace/to_nvidia
cp -a ./vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
```

启动 BF16：

```bash
OMCHAT_PROFILE_MODEL_PHASES=1 \
OMCHAT_PROFILE_OBJECT_FEATURES=1 \
NO_PREFIX_CACHE=1 \
NO_MM_CACHE=1 \
VLLM_USE_FLASHRT_OPS=0 \
bash raw_vllm_server/start_vllm_server.sh
```

启动 NVFP4：

```bash
MODEL_PATH=/workspace/to_nvidia/resources/vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct \
MODEL_NAME=vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct \
VLLM_USE_NATIVE_NVFP4=1 \
OMCHAT_PROFILE_MODEL_PHASES=1 \
OMCHAT_PROFILE_OBJECT_FEATURES=1 \
NO_PREFIX_CACHE=1 \
NO_MM_CACHE=1 \
VLLM_USE_FLASHRT_OPS=0 \
bash raw_vllm_server/start_vllm_server.sh --quantization modelopt_fp4
```

压测脚本：

```bash
python benchmarks/bench_latency_throughput.py \
  --model <served-model-name> \
  --out artifacts/results/latency_throughput_<tag>.json

MODEL_NAME=<served-model-name> \
VLLM_SERVER_LOG=/workspace/to_nvidia/vllm_server_<tag>.log \
MODULE_BENCH_OUT=/workspace/to_nvidia/artifacts/results/module_latency_<tag>.json \
python benchmarks/bench_module_latency.py
```

## 8. 原始数据

| 文件 | 内容 |
|------|------|
| `results/reference/latency_throughput.json` | BF16 串行 + 并发 |
| `results/reference/latency_throughput_nvfp4.json` | NVFP4 串行 + 并发 |
| `results/reference/module_latency.json` | BF16 模块拆分 |
| `results/reference/module_latency_nvfp4.json` | NVFP4 模块拆分 |
| `artifacts/logs/vllm_server.log` | BF16 服务日志 |
| `artifacts/logs/vllm_server_nvfp4.log` | NVFP4 服务日志 |
| `benchmarks/bench_latency_throughput.py` | 客户端 TTFT / TPOT / 并发脚本 |
| `benchmarks/bench_module_latency.py` | 解析 profiler 日志的模块脚本 |

记录时 NVFP4 服务仍在 `http://127.0.0.1:8000` 上运行。请求 JSON 中的 `model` 必须与 served name 一致：`vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct`。

## 9. 后续方向

若还要降端到端延迟，优先 Aux RADIO（C-RADIOv4），而不是继续压 LM。可选方向：量化或裁剪辅助视觉塔、主/辅 ViT 并行（`OMCHAT_PARALLEL_AUX_VIT`）、或打开 prefix / MM cache 测稳态吞吐（与本次「关缓存测阶段耗时」设定不同）。
