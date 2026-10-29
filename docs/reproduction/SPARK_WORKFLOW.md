# DGX Spark 远程任务工作流

这组脚本让 Mac 上的 Codex 修改代码，并通过 SSH 把任务提交到 DGX Spark。
Spark 本身不需要访问 Codex 或 OpenAI。

## 一次性配置

确认 Mac 可以执行：

```bash
ssh spark-3946
```

然后在项目根目录创建本地配置：

```bash
cp .spark.env.example .spark.env
```

默认认为项目在 Spark 主机的 `/home/liuyang/vlx_seek_1_5_3b`。容器内对应挂载点
是 `/workspace/to_nvidia`。如果主机实际目录不同，请修改
`.spark.env` 中的 `SPARK_PROJECT_DIR`。该文件不会进入 Git，也不会被同步。

## 提交任务

远端已有最新代码时：

```bash
scripts/spark-submit -- python3 -m pytest
```

先同步代码再运行。同步不会删除远端文件，并默认排除 `resources/`、模型、日志和
测试产物：

```bash
scripts/spark-submit --sync -- python3 -m pytest
```

需要管道、重定向或环境变量时使用 `--shell`：

```bash
scripts/spark-submit --name profile --shell \
  'CUDA_VISIBLE_DEVICES=0 python3 benchmarks/bench_module_latency.py'
```

## 查看和取回结果

每次提交都会输出 job id。省略 job id 或使用 `latest` 表示最近任务：

```bash
scripts/spark-status latest
scripts/spark-status latest --watch
scripts/spark-fetch latest
```

取回的文件位于 `.spark-results/<job-id>/`，其中包括：

- `stdout.log`、`stderr.log`
- `state`、`exit_code` 和开始/结束时间
- Spark 系统、Python 和 Git 信息
- 执行前后的 `nvidia-smi` 快照
- 任务写入 `$SPARK_JOB_DIR` 的其他结果文件

## VLX 性能测试

确保远端 vLLM 服务已经启动，然后提交现有吞吐/延迟测试：

```bash
scripts/benchmark.sh --sync \
  --model vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct

scripts/spark-status latest --watch
scripts/spark-fetch latest
```

基准 JSON 会随任务日志一起保存为 `benchmark.json`。

只测试 Qwen3.5 语言路径，并把输入/输出固定为 2200/27 token：

```bash
scripts/benchmark-qwen35.sh --sync
scripts/spark-status latest --watch
scripts/spark-fetch latest
```

该测试直接向 `/v1/completions` 提交 2200 个 token ID，不执行视觉塔、ROI 或
Projector。结果保存为 `qwen35_benchmark.json`。

## 安全边界

脚本只连接 `.spark.env` 指定的主机，并只在 `SPARK_PROJECT_DIR` 和它的
`.spark-jobs` 目录中写入。`--sync` 不使用 `--delete`。提交的命令与普通 SSH
命令权限相同，因此运行前仍应检查命令内容。
