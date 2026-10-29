# VLX ModelOpt NVFP4 W4A4 export bundle

This directory contains the sanitized BF16 source model and everything specific
to reproducing its language-only ModelOpt NVFP4 W4A4-direct export.

## Configure model names

Edit `quant_config.json`:

- `source_model`: input model directory under `model_root`.
- `output_model`: new output directory under `model_root`.
- `model_root`: directory used to resolve both model names. `.` keeps the
  bundle self-contained.

The default direct mode writes `input_scale=1.0`; activation blocks remain
dynamically quantized per group at runtime. Optional calibrated modes are
`input_scale`, `input_amax`, and `input_scale_json`.

## Export

Use a CUDA-capable Python environment with the matching PyTorch build. Install
the exporter dependencies, inspect the plan, and run it:

```bash
cd quant
python -m pip install -r requirements.txt
python run_export.py --dry-run
python run_export.py
python verify_export.py
python smoke_load.py
```

The exporter refuses to overwrite a non-empty output directory. Change
`output_model` in `quant_config.json` when reproducing an existing export.

The scope in `language_all_linear_scope/flashrt_nvfp4_manifest.json` contains
186 language Linear checkpoint keys, corresponding to 114 fused vLLM runtime
Linear modules. Vision towers, object modules, embeddings, norms, and lm_head
remain BF16.

For inference with the patched OmChat vLLM runtime, select the
`modelopt_fp4` quantization backend (the service uses
`VLLM_USE_NATIVE_NVFP4=1`).

`verify_export.py` checks the tensor set, ModelOpt scale layout, BF16 exclusions,
and bitwise equality for every unquantized tensor. It writes `export_report.json`.
`smoke_load.py` performs a full native vLLM load without running generation.

Inside the exported model, `source_sanitization_report.json` and
`source_tensor_provenance.json` describe the sanitized BF16 input. The native
NVFP4 export itself is described by `native_nvfp4_w4a4_manifest.json` and
`QUANTIZATION.md`.
