# Issue #591 — activation compression matrix and failure-path report

## Scope and commits

- Branch: `issue-591-actcomp-matrix`
- Feature base: `2ed39ee` (`feat(engine): add opt-in adaptive compression of saved activations`, imported from PR #646 commit `f4564c4`)
- Tests and failure-path hardening: `0258043` (`test(engine): add activation compression matrix and failure-path regressions`)
- New test file: `tests/test_activation_compression_matrix.py`
- No external model or dataset is used. Every input is generated from a fixed `torch.Generator` seed.

## Environment

Raw log: `artifacts/logs/issue-591-env.log`

| Item | Value |
| --- | --- |
| Python | 3.12.3 |
| Platform | Linux-7.0.0-42-generic-x86_64-with-glibc2.39 |
| PyTorch | 2.8.0+cu128 |
| CUDA build | 12.8 |
| GPU | NVIDIA RTX 5880 Ada Generation |
| Compute capability | 8.9 |
| Visible GPUs | 1 (`CUDA_VISIBLE_DEVICES=1`) |
| NumPy | 2.5.4 |
| pytest | 9.2.0 |

## Commands

Targeted CPU + CUDA suite:

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m pytest \
  tests/test_activation_compression_cpu.py \
  tests/test_activation_compression_cuda.py \
  tests/test_activation_compression_matrix.py \
  tests/test_train_cli_config_cpu.py -q \
  2>&1 | tee artifacts/logs/issue-591-targeted-tests-final.log
```

Matrix run used for the tables below:

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m pytest tests/test_activation_compression_matrix.py -q -s \
  2>&1 | tee artifacts/logs/issue-591-matrix.log
```

Full suite (single visible GPU, tmux session `issue-591-run`):

```bash
tmux new-session -d -s issue-591-run -c "$PWD" \
  'env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests -q \
   2>&1 | tee -a artifacts/logs/issue-591-full-tests.log'
```

## Results

### 1. Quantization / dequantization matrix vs independent FP32 reference

Input: shape `(1024, 96)`, stochastic rounding, `seed=2026`, test seed `100 + bits`.
`exact_match` means the packed codes, group minima, group steps and outlier
channel indices equal the independent NumPy FP32 reference exactly, and the
dequantized tensor equals the reference reconstruction with `atol=0, rtol=0`.

| bits | dtype | exact_match | reconstruction rel-L2 | max abs error | stored fraction |
| --- | --- | --- | --- | --- | --- |
| 2 | bfloat16 | true | 0.79893845 | 17.53125 | 0.208415 |
| 4 | bfloat16 | true | 0.14037621 | 3.63671875 | 0.333415 |
| 8 | bfloat16 | true | 0.00839448 | 0.21875 | 0.583415 |
| 2 | float16 | true | 0.79876614 | 17.51171875 | 0.208415 |
| 4 | float16 | true | 0.14027509 | 3.63525391 | 0.333415 |
| 8 | float16 | true | 0.00837823 | 0.21484375 | 0.583415 |

`reconstruction rel-L2` and `max abs error` are against the original tensor;
`exact_match` is against the independent FP32 reference. All 6 combinations
matched the reference exactly. Raw metrics are `section="quantization_matrix"`
lines in `artifacts/logs/issue-591-matrix.log`.

### 2. Gradient matrix, uniform plan over 3 saved slots

Model: residual MLP (`Linear(128,512) -> SiLU -> Linear(512,128)` plus residual),
input `(512, 128)`, 3 saved slots, CPU, `seed=1`.

| bits | dtype | gradient rel-L2 | stored fraction |
| --- | --- | --- | --- |
| 2 | bfloat16 | 0.28564066 | 0.203247 |
| 4 | bfloat16 | 0.05751461 | 0.328247 |
| 8 | bfloat16 | 0.00413882 | 0.578247 |
| 2 | float16 | 0.28553788 | 0.203247 |
| 4 | float16 | 0.05748345 | 0.328247 |
| 8 | float16 | 0.00349401 | 0.578247 |

The test also asserts the ordering `rel-L2(8-bit) <= rel-L2(4-bit) <= rel-L2(2-bit)`
for both dtypes. Raw metrics are `section="gradient_matrix"` lines.

### 3. 1% budget and over-budget fallback

Adaptive calibration on the same residual MLP, CPU, budget `0.01`:

| metric | value |
| --- | --- |
| calibration error | 0.00338862 |
| held-out error | 0.00340465 |
| stored fraction | 0.296936 |
| uncompressed fallback | false |

Both measured errors are inside the 1% budget. Raw metric: `section="budget_1pct"`.

Over-budget check: the same model calibrated at a 2% budget (`[8,4,8]`, error
`0.0129626`), then the budget was lowered to `1e-7` and `check()` was run:

| metric | value |
| --- | --- |
| check error | 0.0129626 |
| recalibrated | 1.0 |
| resulting plan | `[16,16,16]` (all uncompressed) |
| uncompressed fallback | 1.0 |

This verifies the fallback really occurs when the measured gradient error
exceeds the budget. Raw metric: `section="fallback"`.

### 4. Failure paths

| Failure path | Assertion / observed result |
| --- | --- |
| Invalid ratio (`-8,-1,0,1,3,5,6,7,9,16,32`) | `quantize`, `ActivationCompressor` and `RuntimeConfig` each raise `ValueError` |
| Unsupported saved dtype | integer embedding ids are recorded in the capture's uncompressed `held` set, no accessor claims they are slots; direct `quantize(int64)` raises `ValueError` |
| Unsupported model layout | MoE, LoRA and multi-GPU layouts are rejected explicitly by `EngineConfig` |
| Packing OOM | 3/3 slots fall back to uncompressed (`oom_fallback_slots=3`, stored fraction `1.0`); backward gradients are identical to the uncompressed reference (`relative_error=0.0`) |
| Single calibration batch | `calibrate(run, None)` succeeds, calibration error `0.01305134`, no held-out metric, held-out error is `NaN` |
| Empty eligible slot set | calibration succeeds with `slots=0`, plan `[]`, stored fraction `0.0`, fallback `true` |

OOM hardening is implemented in `areno/engine/runtime/activation_compression.py`:
`_Capture.pack` catches `torch.cuda.OutOfMemoryError` / `MemoryError`, clears the
CUDA cache, keeps the original tensor and counts `actc_oom_fallback_slots`.

### 5. CUDA kernel matrix

Kernel path (`use_kernel=True`) vs PyTorch path (`use_kernel=False`) for all
`bits × dtype` combinations was bitwise equal in codes, minima, steps, outlier
indices, outliers and dequantized output. Reconstruction rel-L2 (test seed
`200 + bits`, so inputs differ from the CPU table):

| bits | dtype | bitwise_equal | reconstruction rel-L2 |
| --- | --- | --- | --- |
| 2 | bfloat16 | true | 0.79597986 |
| 4 | bfloat16 | true | 0.14123535 |
| 8 | bfloat16 | true | 0.00826044 |
| 2 | float16 | true | 0.79580319 |
| 4 | float16 | true | 0.14119779 |
| 8 | float16 | true | 0.00823953 |

Raw metrics are `section="cuda_matrix"` lines.

## Test outcomes

Targeted suite (`artifacts/logs/issue-591-targeted-tests-final.log`):

```text
261 passed, 1 skipped in 13.33s
```

The skipped test is the existing flash-attn-only CUDA test; `flash-attn` is not
installed in this environment. Matrix file alone
(`artifacts/logs/issue-591-matrix.log`):

```text
31 passed in 7.85s
```

CPU-only run (no visible GPU, `CUDA_VISIBLE_DEVICES=""`) is in
`artifacts/logs/issue-591-matrix-cpu-only.log`:

```text
25 passed, 6 skipped in 5.61s
```

The 6 skips are the CUDA matrix tests, gated by `pytest.mark.skipif`.

## Full-suite caveat

The full suite in the mandated single-visible-GPU environment finished as
`328 failed, 2352 passed, 1785 skipped` in 554.37s
(`artifacts/logs/issue-591-full-tests.log`). These failures are not in the new
test file; they concentrate in tests that assume several visible devices or
that run GPU kernels after an NPU-stub test has changed process state:

| test file | failed |
| --- | --- |
| `test_routing.py` | 84 |
| `test_mst_models.py` | 66 |
| `test_attention_native.py` | 61 |
| `test_moe_native.py` | 38 |
| `test_logprobs_cuda.py` | 16 |
| `test_grouped_linear.py` | 15 |
| remaining files | 48 |

Two concrete failure reasons seen in the log:

- `ValueError: devices are outside CUDA_VISIBLE_DEVICES: [1, 2, 3]` in
  `test_config_data_cpu.py`, because only GPU 1 is visible.
- `RuntimeError: PyTorch is not linked with support for npu devices` in
  `test_activation_compression_cuda.py` during the full ordering; the same file
  passes alone (`75 passed, 1 skipped`).

A clean-main baseline full-suite run was not performed, so this report does not
claim these failures are pre-existing. It does record that the targeted
activation-compression suite, including the new matrix/failure-path tests, is
green on the committed tree.
