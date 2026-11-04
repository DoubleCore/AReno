# issue-591 notes

- Feature under test: PR #646 (`f4564c4`, imported as local base commit
  `2ed39ee` on `issue-591-actcomp-matrix`). Public surface:
  `areno.engine.runtime.activation_compression`.
- Matrix target: `quantize` / `dequantize` for bits `{2,4,8}` and dtypes
  `{bfloat16,float16}` against an independent NumPy FP32 reference (codes,
  group minima/steps, exact outlier channels, reconstruction).
- Gradient target: uniform and adaptive plans on a small residual MLP; relative
  L2 is measured over all parameter gradients by `relative_error`.
- Failure paths covered in `tests/test_activation_compression_matrix.py`:
  invalid bits (`<=0`, unsupported, `>8`), unsupported saved dtypes/layouts,
  single-batch and empty calibration, over-budget check fallback, and packing
  OOM fallback.
- OOM hardening: `_Capture.pack` now catches `torch.cuda.OutOfMemoryError` /
  `MemoryError`, clears the CUDA cache, keeps the original tensor, and accounts
  it as `oom_fallback_slots` instead of failing the step.
- No model weights or external datasets are used; all inputs are fixed
  generators, so results are reproducible from the seed in the test file.
