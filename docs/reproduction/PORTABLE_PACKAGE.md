# VLX DGX Spark 便携复现包

该压缩包保留定制 vLLM、benchmark、量化工具、C-RADIO TensorRT 工具、运行
脚本、示例配置和参考结果。为控制体积，不包含模型权重、TensorRT engine、ONNX、
原始日志和第三方 wheel；安装脚本会按已验证的版本在线下载依赖。

## 1. 解压与完整性检查

```bash
tar -xzf vlx-dgx-spark-repro-YYYYMMDD.tar.gz
cd vlx-dgx-spark-repro
shasum -a 256 -c PACKAGE_MANIFEST.sha256
bash scripts/check-dgx-spark-env.sh --source-only
```

Linux 上也可以用 `sha256sum -c PACKAGE_MANIFEST.sha256`。

## 2. 准备基础环境

已验证目标是 DGX Spark（GB10 / SM121）、Linux aarch64、Python 3.12、CUDA
13，并使用包含 PyTorch、vLLM 0.19.0 和 TensorRT 的 VLX 基础容器。基础镜像不在
压缩包内；接收方需要取得同一镜像，或提供等价环境。

进入容器并把本目录挂载为 `/workspace/to_nvidia`，然后执行：

```bash
cd /workspace/to_nvidia
bash scripts/prepare-repro-env.sh
cp -a vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
```

`prepare-repro-env.sh` 会在线下载固定版本的 FlashInfer、CUTLASS DSL、TVM FFI
和 Humming，并在 `third_party/` 下建立本地缓存。历史 benchmark 脚本可以继续使用
这些缓存。若要先在联网机器准备缓存：

```bash
bash scripts/prepare-repro-env.sh --download-only
```

然后重新打包时使用 `scripts/build-repro-package.sh --include-wheel-cache`。

如果收到的压缩包已经包含离线依赖，完全不访问网络的安装方式为：

```bash
bash scripts/prepare-repro-env.sh --offline
```

## 3. 放置模型权重

权重未包含在压缩包中。取得原始和 NVFP4 模型后，将完整目录放到任意大容量磁盘，
通过 `MODEL_PATH` 指向它。目录必须包含根 `config.json`、权重文件，以及模型配置要求
的 C-RADIO 和 Qwen Vision Tower 文件。

```bash
export MODEL_PATH=/data/models/vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct
bash scripts/check-dgx-spark-env.sh --model-path "${MODEL_PATH}"
```

`resources/` 中保留的是配置、模型自定义代码和量化 provenance，不是完整权重。

## 4. 复现优化测试

主要入口：

```bash
# Qwen3.5 基准
bash scripts/benchmark-qwen35.sh

# GDN prefill：FlashInfer 与 Triton 对照
bash scripts/run-gdn-sm121-validation.sh 8102 flashinfer
bash scripts/run-gdn-sm121-validation.sh 8103 triton

# W4A4 backend sweep
bash scripts/run-w4a4-backend-sweep.sh

# C-RADIO 动态 shape TensorRT profile
bash scripts/run-aux-trt-camera-profile.sh

# 完整优化栈 A/B
bash scripts/run-vlx-full-stack-ab.sh
```

远程提交方式、环境变量和各测试解释见同目录的
`DGX_SPARK_REPRODUCTION.md` 与 `SPARK_WORKFLOW.md`。参考结果在
`results/reference/`，新的完整结果默认写入 `.spark-results/` 或 `artifacts/`。

## 5. 重新生成压缩包

标准在线安装包：

```bash
bash scripts/build-repro-package.sh
```

接收环境无法联网时，可以先下载依赖，再生成较大的离线包：

```bash
bash scripts/prepare-repro-env.sh --download-only
bash scripts/build-repro-package.sh --include-wheel-cache
```

无论使用哪个模式，模型权重、engine、ONNX 和本地密钥都不会被打入压缩包。
