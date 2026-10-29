# ModelOpt NVFP4 W4A4-direct checkpoint

This checkpoint was exported from the sanitized BF16 source. It quantizes 186
language Linear weights into ModelOpt NVFP4 with group size 16 and direct input
scale 1.0. Vision towers, object modules, embeddings, norms, and lm_head remain
BF16.

The inherited sanitization evidence is named
`source_sanitization_report.json` and `source_tensor_provenance.json` because
their checkpoint hash and tensor manifest refer to the BF16 source model.
