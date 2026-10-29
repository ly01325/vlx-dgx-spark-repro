# VLX on DGX Spark

本仓库包含 VLX 在 NVIDIA DGX Spark（GB10 / SM121）上的定制 vLLM、量化、
TensorRT、benchmark 和复现脚本。默认模型为
`vlx-seek-1.5-3b-sanitized`。

`launch_vllm.py` 是一个很薄的启动适配器。它只在进入 vLLM 前关闭 cuDNN，以兼容目标 GPU 上模型 SimpleFPN 的转置卷积。

## 仓库结构

```text
vlx_seek_1_5_3b/
├── benchmarks/          # Python benchmark 与结果比较工具
├── scripts/             # DGX Spark 提交、A/B 测试和验证入口
├── raw_vllm_server/     # 服务启动与请求示例
├── vllm_/               # 定制 vLLM 源码覆盖层
├── trt_aux/             # C-RADIO TensorRT 导出与验证
├── quant/               # ModelOpt NVFP4 量化工具
├── third_party/
│   ├── wheels/          # 固定版本的 aarch64 wheel 与 SHA256
│   └── sources/         # Humming 等外部源码归档（本地提供）
├── docs/
│   ├── reproduction/    # 从环境准备到复现实验的操作手册
│   ├── optimization/    # 优化方案和性能基线
│   └── validation/      # 各优化项验证报告
├── artifacts/           # 本地生成的结果与日志，不作为源码同步
├── results/reference/   # 可随项目发布的参考 benchmark JSON
├── deliverables/        # 汇报材料
└── resources/           # 模型和样例资源，不通过 --sync 复制
```

首次复现请从 [`docs/reproduction/DGX_SPARK_REPRODUCTION.md`](docs/reproduction/DGX_SPARK_REPRODUCTION.md)
开始，并先执行：

```bash
bash scripts/verify-repro-assets.sh
```

如果通过压缩包交付，优先阅读
[`docs/reproduction/PORTABLE_PACKAGE.md`](docs/reproduction/PORTABLE_PACKAGE.md)。生成默认的
在线安装便携包：

```bash
bash scripts/build-repro-package.sh
```

该包保留源码、测试、文档、示例配置和参考结果，但排除模型权重、第三方 wheel、
TensorRT engine、ONNX、原始日志及缓存。接收方运行
`bash scripts/prepare-repro-env.sh` 即可在线下载固定版本依赖。

## 交付目录

交付时需要包含以下内容：

```text
package/
├── benchmarks/ # benchmark与对比工具
├── scripts/ # Spark任务与验证入口
├── raw_vllm_server/ # server启动脚本及request测试脚本
├── resources/
│   └── vlx-seek-1.5-3b-sanitized/ # 原模型
│   └── vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct/ # LM所有线性层nvfp4量化
├── third_party/ # 固定版本wheel与外部源码归档
├── vllm_/ # 定制vllm
└── quant/ # ModelOpt nvfp4量化相关,模型已经导出,这里附上导出脚本及量化范围
```

必须提供完整的 `vlx-seek-1.5-3b-sanitized` 模型目录，包括其中的
`C-RADIOv4-SO400M` 和 `Qwen3.5-2B-Vision_Tower` 子目录。

## load 镜像 vlx_seek1_5.tar.gz
> 镜像名：registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted vllm版本0.19.0
```bash
docker load -i vlx_seek1_5.tar.gz 

```
## 安装定制版 vLLM

需要先根据上述镜像启动容器，在容器环境中执行一次：

```bash
cp -a ./vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
```

启动脚本会检查安装后的 vLLM 是否包含自定义模型架构
`OmChatQwen3_5ForCausalLM`。如果复制不完整，会在加载模型前直接报错。

## 启动服务

在交付目录根目录执行：
> OMCHAT_PROFILE_MODEL_PHASES=1 OMCHAT_PROFILE_OBJECT_FEATURES=1 开启各阶段的耗时打印，设置为0关闭，NO_PREFIX_CACHE=1 NO_MM_CACHE=1 关闭缓存，测试各阶段耗时时建议设置为1关闭；VLLM_USE_FLASHRT_OPS=0显式设置不使用flashrt kernel
```bash
OMCHAT_PROFILE_MODEL_PHASES=1 \
OMCHAT_PROFILE_OBJECT_FEATURES=1 \
NO_PREFIX_CACHE=1 \
NO_MM_CACHE=1 \
VLLM_USE_FLASHRT_OPS=0 \
bash raw_vllm_server/start_vllm_server.sh
```

默认监听地址为 `http://0.0.0.0:8000`，OpenAI 兼容接口为
`http://127.0.0.1:8000/v1/chat/completions`。

可以通过环境变量覆盖模型和服务参数，例如：

> MODEL_PATH模型路径，MODEL_NAME指定模型名：vlx-seek-1.5-3b-sanitized bf16
> vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct nvfp4
```bash
MODEL_PATH=/workspace/to_nvidia/resources/vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct \
MODEL_NAME=vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct \
HOST=0.0.0.0 \
PORT=8000 \
GPU_MEMORY_UTILIZATION=0.8 \
bash raw_vllm_server/start_vllm_server.sh
```

其中：

- `MODEL_PATH`：模型在磁盘上的实际目录。
- `MODEL_NAME`：vLLM 对外暴露的模型名称，请求 JSON 中的 `model` 必须与它一致。
- `HOST`：监听地址，默认 `0.0.0.0`。
- `PORT`：监听端口，默认 `8000`。
- `TENSOR_PARALLEL_SIZE`：张量并行使用的 GPU 数量，默认使用全部可见 GPU。
- `GPU_MEMORY_UTILIZATION`：GPU 显存使用比例；未设置时按照原服务逻辑自动计算。
- `MAX_MODEL_LEN`：最大模型上下文长度，默认 `16384`。
- `MAX_NUM_SEQS`：最大并发序列数，默认 `256`。
- `MAX_NUM_BATCHED_TOKENS`：单批最大 token 数，默认 `4096`。
- `MAX_IMAGES`：每个请求允许的最大图片数，默认 `1`。
- `APIKEY`：可选的接口鉴权密钥。

还可以在启动命令末尾追加其他 vLLM 命令行参数。

`OD_URL`、`OD_MODEL_ID` 和 `OD_CONF` 属于外层 OmChat 服务，原生 vLLM
不会使用这些变量。如果检测框来自目标检测服务，需要先调用目标检测服务，再把
检测框传给 `vllm_request.py`。

## 发送默认请求

直接运行：
> 使用默认图片（默认1920*1082）和prompt，bboxes（固定100个）输出token默认27（用户场景一般不是出太多token）

```bash
python raw_vllm_server/vllm_request.py
```

无参数运行时使用与原 `request.py` 相同的默认值：

- 图片：`/workspace/omchat/demo_1920_1080.jpeg`
- 问题：`检测图像中的枕头和鞋子和柜子，生成对应的标签并返回物体的索引`
- 检测框：完整的 100 个框
- `max_tokens`：`27`
- 模型名称：`vlx-seek-1.5-3b-sanitized`
- 接口地址：`http://127.0.0.1:8000/v1/chat/completions`

## 覆盖请求参数

使用其他图片和检测框：

```bash
python raw_vllm_server/vllm_request.py \
  --image /path/to/image.jpg \
  --prompt '检测所有的车，返回物体索引' \
  --bbox-json '[[0, 201.9, 717.4, 886.7], [2780.8, 351.6, 2887.8, 469.8]]'
```

检测框较多时可以放在 JSON 文件中：

```bash
python raw_vllm_server/vllm_request.py \
  --image /path/to/image.jpg \
  --bbox-file boxes.json
```

其他常用参数：

- `--url`：修改 vLLM 接口地址。
- `--model`：修改请求中的模型名称。
- `--max-tokens`：修改最大输出 token 数。
- `--temperature`、`--top-p`、`--top-k`：修改采样参数。
- `--stream`：使用流式响应。
- `--raw`：输出完整 JSON 响应。
- `--api-key`：设置接口鉴权密钥。

本地图片会自动编码为 data URL，因此请求脚本和 vLLM 服务可以运行在不同机器上。
`--image` 同时支持 HTTP 或 HTTPS 图片地址。多张图片可以重复传入 `--image`，但
需要相应提高服务端的 `MAX_IMAGES`。

## bbox 请求格式

直接调用 vLLM 时，检测框必须放在 `mm_processor_kwargs.bbox_list` 中：

```json
{
  "model": "vlx-seek-1.5-3b-sanitized",
  "mm_processor_kwargs": {
    "bbox_list": [[0, 0, 100, 100]]
  }
}
```

外层 OmChat 接口接受顶层的 `bbox_list`，原生 vLLM 接口不接受这种格式。

每个框使用 `[xmin, ymin, xmax, ymax]` 像素坐标。框在列表中的顺序对应
`<obj0>`、`<obj1>` 等对象索引。

## 健康检查

```bash
curl http://127.0.0.1:8000/health
```

查看已加载模型：

```bash
curl http://127.0.0.1:8000/v1/models
```

## 从 Mac 提交 DGX Spark 任务

Spark 无法访问 Codex/OpenAI 时，可以让 Codex 在 Mac 上修改代码，再通过 SSH
提交远端任务并取回日志、GPU 快照和 benchmark 结果。首次使用先执行：

```bash
cp .spark.env.example .spark.env
# 如果远端目录不是 /workspace/to_nvidia，请编辑 SPARK_PROJECT_DIR

scripts/benchmark.sh --sync --model vlx-seek-1.5-3b-sanitized
scripts/spark-status latest --watch
scripts/spark-fetch latest
```

完整用法和安全边界见 [`docs/reproduction/SPARK_WORKFLOW.md`](docs/reproduction/SPARK_WORKFLOW.md)。
