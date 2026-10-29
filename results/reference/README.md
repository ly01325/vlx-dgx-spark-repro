# Reference baseline results

这些文件来自目录整理前的 `bench_out/`：

- `latency_throughput.json`：BF16 全 VLX 延迟与吞吐
- `latency_throughput_nvfp4.json`：NVFP4 全 VLX 延迟与吞吐
- `module_latency.json`：BF16 模块级耗时
- `module_latency_nvfp4.json`：NVFP4 模块级耗时

它们用于历史对照；新的正式性能数字应同时记录对应 Spark job id 和环境快照。
