# Results

- `reference/` 保存可随项目发布的基线 JSON，用于检查指标字段和回归趋势。
- 新实验先写入 `artifacts/results/` 或 `.spark-results/<job-id>/`。
- 只有确认环境、命令和输入口径后，才把代表性 JSON 复制到 `reference/`。

参考 JSON 不能替代 `.spark-results/<job-id>/` 中的命令、环境和日志证据。
