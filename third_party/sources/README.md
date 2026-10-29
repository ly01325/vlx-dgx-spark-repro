# External source archives

需要 Humming W4A16/W4A4 测试时，运行 `scripts/prepare-repro-env.sh`。它会下载
固定提交对应的源码归档到：

```text
third_party/sources/humming-a74973b.tar.gz
```

便携包默认不包含该归档。归档名中的 `a74973b` 是测试所用提交标识；不要在复现
实验时静默替换为其他提交。

离线依赖版便携包会包含该归档和 `SHA256SUMS`，安装脚本会在使用前校验源码包。
