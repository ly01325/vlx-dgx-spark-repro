# Script entry points

脚本分为四类：

- `spark-submit` / `spark-status` / `spark-fetch`：Mac 到 DGX Spark 的任务生命周期。
- `benchmark*.sh`：已有服务的通用性能测试。
- `run-*-validation.sh`、`run-*-ab.sh`、`run-*-sweep.sh`：一次性容器中的优化验证。
- `install-dgx-spark-deps.sh`、`verify-repro-assets.sh`：环境安装与资产完整性检查。

建议不要直接复制脚本中的内部 Docker 命令。优先从
[`../docs/reproduction/DGX_SPARK_REPRODUCTION.md`](../docs/reproduction/DGX_SPARK_REPRODUCTION.md)
选择实验入口，这样生成的 job 目录会保留完整命令和环境信息。
