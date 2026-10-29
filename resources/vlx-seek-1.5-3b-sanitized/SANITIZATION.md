# Sanitized VLX checkpoint

This package preserves the VLX checkpoint structure but contains no private trained tensor values. Qwen3.5 and C-RADIO tensors come from their official checkpoints. VLX-only projector and SimpleFPN tensors were freshly initialized with seed 20260910.

See `sanitization_report.json` and `tensor_provenance.json` for the machine-readable audit trail. The custom modules require adaptation or training before model-quality evaluation.
