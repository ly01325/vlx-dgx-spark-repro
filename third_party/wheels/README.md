# DGX Spark wheel cache

这里缓存当前 SM121 测试使用的安装包。便携复现压缩包默认不携带 wheel；在 DGX
Spark 容器中运行 `bash scripts/prepare-repro-env.sh` 后会从 PyPI 下载并填充本目录。
这些 wheel 面向 Linux aarch64、Python 3.12 和 CUDA 13，不应在 macOS 安装。

固定版本：

- FlashInfer Python 0.6.18
- Apache TVM FFI 0.1.13.post3
- NVIDIA CUTLASS DSL 4.7.1
- NVIDIA CUDA nvdisasm 13.4.49

如下载产物与最初验证版本一致，可以校验：

```bash
cd /workspace/to_nvidia
sha256sum -c third_party/wheels/SHA256SUMS
bash scripts/install-dgx-spark-deps.sh --all
```

不要安装“最新版”；`requirements/dgx-spark-kernels.txt` 固定了 benchmark 所用版本。
