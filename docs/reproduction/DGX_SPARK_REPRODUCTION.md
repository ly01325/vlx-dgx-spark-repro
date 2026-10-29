# DGX Spark benchmark reproduction

本文档用于在另一台 DGX Spark（GB10 / SM121）上复现当前 VLX 优化测试。
所有命令默认从项目根目录执行。

## 1. 固定环境

已验证环境：

- DGX Spark / GB10 / SM121
- 容器：`registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted`
- Python 3.12
- vLLM 0.19.0 基线，加仓库中的 `vllm_/` 覆盖层
- CUDA 13.x
- FlashInfer 0.6.18
- CUTLASS DSL 4.7.1

驱动、内核、CUDA 和 TensorRT 会显著影响结果。每次远程任务都会把系统信息和
`nvidia-smi` 保存到 `.spark-results/<job-id>/`，发布数字时必须同时保存 job id。

## 2. 准备资产

必须提供：

```text
resources/vlx-seek-1.5-3b-sanitized/
resources/vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct/
resources/demo_1920_1080.jpeg
```

Humming 测试还需要：

```text
third_party/sources/humming-a74973b.tar.gz
```

检查目录和 wheel 完整性：

```bash
bash scripts/verify-repro-assets.sh
bash scripts/verify-repro-assets.sh --require-humming  # Humming/整栈测试
```

## 3. 配置 Mac → Spark 工作流

```bash
cp .spark.env.example .spark.env
```

修改 `SPARK_HOST` 和 `SPARK_PROJECT_DIR`。详细机制见
[`SPARK_WORKFLOW.md`](SPARK_WORKFLOW.md)。模型目录和运行结果不会被 `--sync`
覆盖；固定 wheel 和 Humming 源码归档会随项目同步。

## 4. 安装容器内依赖

仓库 wheel 只适用于 Linux aarch64/Python 3.12：

```bash
bash scripts/install-dgx-spark-deps.sh --flashinfer
bash scripts/install-dgx-spark-deps.sh --humming   # 可选
cp -a vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
```

各 A/B 脚本会在一次性容器中重复这些步骤，避免污染长期运行的服务。

## 5. 建议复现顺序

### 5.1 Qwen3.5 固定 2200 → 27 基线

```bash
scripts/benchmark-qwen35.sh --sync
scripts/spark-status latest --watch
scripts/spark-fetch latest
```

### 5.2 FlashInfer GDN prefill

分别跑 FlashInfer 和 Triton 对照：

```bash
scripts/spark-submit --sync --name gdn-flashinfer -- \
  bash scripts/run-gdn-sm121-validation.sh 8102 flashinfer
scripts/spark-submit --name gdn-triton -- \
  bash scripts/run-gdn-sm121-validation.sh 8103 triton
```

检查 kernel 延迟、Qwen E2E、TTFT，以及 output/final-state cosine。

### 5.3 Humming W4A16 LM Head

```bash
scripts/spark-submit --sync --name humming-combined -- \
  bash scripts/run-humming-combined-validation.sh
```

同时检查性能和 token 精度 canary；不要只用吞吐结果决定上线。

### 5.4 C-RADIO TensorRT 动态 shape

```bash
scripts/spark-submit --sync --name c-radio-camera -- \
  bash scripts/run-aux-trt-camera-profile.sh

scripts/spark-submit --name c-radio-ab -- \
  bash scripts/run-vlx-aux-trt-camera-ab.sh
```

验证 480P/720P/1080P 的路由、per-tap cosine、模块时延和全 VLX E2E。

### 5.5 最佳整栈 A/B

```bash
scripts/spark-submit --sync --name vlx-full-stack -- \
  bash scripts/run-vlx-full-stack-ab.sh
```

Baseline：FlashInfer-CUTLASS W4A4 + BF16 LM Head + Triton GDN + PyTorch C-RADIO。

Optimized：FlashInfer-CUTLASS W4A4 + Humming W4A16 LM Head + FlashInfer GDN +
C-RADIO TensorRT。

## 6. 结果与追踪

每个 Spark job 至少应保留：

- `command.sh`：实际执行命令
- `environment.txt`：主机、Python、Git 和内核信息
- `nvidia-smi-before.txt` / `nvidia-smi-after.txt`
- `stdout.log` / `stderr.log`
- benchmark JSON
- 服务日志和 backend 选择日志

本地完整记录位于 `.spark-results/<job-id>/`。经过筛选、准备发布的 JSON 可以复制
到 `artifacts/results/<experiment>/`，但不要用汇总文件替代原始 job 目录。

## 7. 可比性要求

对比实验必须固定：

- 模型权重、容器、wheel checksum 和 vLLM 覆盖层
- 输入 token、输出 token、图片、候选框和采样参数
- warmup 次数、测量次数和并发请求数
- GPU memory utilization、prefix/MM cache 设置
- TensorRT engine/profile 和 shape 路由策略

涉及 Humming 时还必须报告精度 canary 或私有任务级 accuracy gate。当前 sanitized
checkpoint 的私有 object head 不能代替正式任务精度验证。
