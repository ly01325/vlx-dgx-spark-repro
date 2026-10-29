# Reproduction dependencies

`dgx-spark-kernels.txt` records the exact online package versions used by the
SM121 benchmark. The base VLX container must already provide CUDA 13, PyTorch,
vLLM 0.19.0 and TensorRT. Run this inside that container:

```bash
bash scripts/prepare-repro-env.sh
bash scripts/check-dgx-spark-env.sh --model-path /path/to/model
```

Humming is installed from commit `a74973b` of
`https://github.com/inclusionAI/humming`. It is intentionally fetched by the
preparation script instead of being vendored into the archive.
