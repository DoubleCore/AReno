## Summary

Add cross-configuration and failure-path regressions for opt-in saved-activation
compression (GACT, PR #646), plus one small hardening change so a packing-time
CUDA OOM falls back to the uncompressed path instead of failing the step.

- `tests/test_activation_compression_matrix.py`: bits `{2,4,8}` × dtype
  `{bfloat16,float16}` quantize/dequantize matrix against an independent NumPy
  FP32 reference; GPU kernel-vs-PyTorch bitwise matrix.
- Failure paths: invalid bits (`<=0`, unsupported, `>8`), unsupported saved
  dtype, unsupported model layouts, packing OOM, single-batch calibration,
  empty eligible slot set, and over-budget recalibration that must fall back.
- `areno/engine/runtime/activation_compression.py`: `_Capture.pack` catches
  `torch.cuda.OutOfMemoryError` / `MemoryError`, clears the CUDA cache, keeps
  the original tensor and reports `actc_oom_fallback_slots`.

The branch imports PR #646's feature commit as local base `2ed39ee` because
`origin/main` does not contain it yet; review the commits after that base for
the changes in this PR. The feature commit is `f4564c4` from
`inclusionAI/AReno` PR #646.

## Validation

Environment: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, NumPy 2.5.4,
pytest 9.2.0, one visible RTX 5880 Ada (sm_89).

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m pytest \
  tests/test_activation_compression_cpu.py \
  tests/test_activation_compression_cuda.py \
  tests/test_activation_compression_matrix.py \
  tests/test_train_cli_config_cpu.py -q
```

Result: `261 passed, 1 skipped` (the skip is the existing flash-attn-only test;
`flash-attn` is not installed).

Key measured results (all from `artifacts/logs/issue-591-matrix.log`):

| Check | Result |
| --- | --- |
| Quantization matrix | 6/6 combinations exactly match independent FP32 reference (`atol=0, rtol=0`) |
| Reconstruction rel-L2 | 8-bit: 0.00839 (bf16), 0.00838 (fp16); 4-bit: 0.14038 / 0.14028; 2-bit: 0.79894 / 0.79877 |
| Gradient rel-L2, 8-bit | 0.00414 (bf16), 0.00349 (fp16) |
| Gradient rel-L2, 4-bit | 0.05751 (bf16), 0.05748 (fp16) |
| Gradient rel-L2, 2-bit | 0.28564 (bf16), 0.28554 (fp16) |
| 1% adaptive budget | calibration 0.00338862, held-out 0.00340465 |
| Over-budget fallback | check error 0.0129626 -> plan `[16,16,16]`, `uncompressed_fallback=1.0` |
| Packing OOM | 3/3 slots kept uncompressed, gradients identical to reference |
| CUDA kernel matrix | 6/6 bitwise equal to PyTorch path |

CPU-only run: `25 passed, 6 skipped` (CUDA tests gated by
`pytest.mark.skipif`).

Full suite caveat: `328 failed, 2352 passed, 1785 skipped` under the mandated
single-visible-GPU environment; failures are outside the new file and concentrate
in tests that assume multiple visible devices (e.g. `devices are outside
CUDA_VISIBLE_DEVICES: [1, 2, 3]`) or that hit the NPU stub state during the full
ordering. The activation-compression CUDA file passes alone (`75 passed,
1 skipped`). See `REPORT.md` and `artifacts/logs/issue-591-full-tests.log`.

## Risks / limitations

- The OOM fallback covers packing (`quantize`), where the extra workspace is
  allocated. OOM elsewhere in the model forward still propagates normally.
- The full-suite failures above were not re-run against a clean `main` baseline;
  they are reported as environment/order sensitivity, not claimed as pre-existing.
