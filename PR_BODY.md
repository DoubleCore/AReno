## Summary

GPU-side completion for issue #588 / PR #657's opt-in cut cross-entropy (CCE)
LM-head path. PR #657 landed the pure-PyTorch reference and CPU baseline but
explicitly had no CUDA host, so its GPU table was empty. This branch adds real
RTX 5880 Ada measurements, a CUDA packed-head test, a reproducible GPU
benchmark script, and fills the README table. **The CCE arithmetic, the
default-off frontend switch and the explicit `NotImplementedError`s for
`tp_size>1`, LM-head bias and `label_smoothing` are unchanged.**

Branch: `issue-588-cce-gpu`, based on `eaaae93` (PR #657 `issue-588-cce`).

## Changes

- `tests/test_cut_cross_entropy.py`: add a CUDA-only packed-head test
  (`test_packed_head_cut_cross_entropy_matches_full_projection_cuda`) parametrized
  over FP32 (outputs + gradients) and BF16 (outputs), so the deferred-head
  integration is covered on GPU, not just CPU.
- `examples/cce/bench_gpu.py`: reproducible GPU benchmark with three modes:
  - `accuracy`: FP32/BF16 error at full vocabulary;
  - `synthetic`: output-head peak and `max single allocation` via
    `torch.cuda.memory._record_memory_history`, proving no `tokens x vocab`
    block is ever allocated;
  - `model`: real Qwen3-0.6B full training step (forward with deferred head,
    SFT loss, backward, FP32 main-grad folding, AdamW), warmup 2 + 10 steps.
- `examples/cce/README.md`: replace the empty GPU TODO table with the measured
  numerical, synthetic-memory, chunk-sweep and real-model tables.
- `REPORT.md`: commands, raw-log paths, configuration, all result tables and
  environment details (commit, model hashes, seed, dtype, batch/context, versions).

## Validation (all numbers from `artifacts/logs/`, single RTX 5880 Ada 48 GB)

`tests/test_cut_cross_entropy.py`: **36 passed**, including the two tests that
skip without CUDA (CUDA peak-memory assertion and native-backend config test).

Numerical agreement (`tokens=4096 hidden=1024 vocab=151936 chunk=4096`):

| dtype | max abs | mean abs | max rel | mean rel | scalar loss rel diff |
| --- | --- | --- | --- | --- | --- |
| FP32 | 1.678e-04 | 1.360e-05 | 2.121e-06 | 1.065e-07 | 1.071e-07 |
| BF16 | 8.452e-01 | 3.742e-02 | 1.173e-02 | 2.950e-04 | 9.508e-07 |

Synthetic output head (bf16, fwd+bwd, 4096 tokens, hidden 1024):

| vocab | path | median ms | peak alloc delta MB | max single alloc MB | full logits block MB |
| --- | --- | --- | --- | --- | --- |
| 32768 | materialized | 7.21 | 768.0 | 256.0 | 256.0 |
| 32768 | CCE chunk 4096 | 23.16 | 344.3 | 128.0 | 256.0 |
| 65536 | materialized | 15.73 | 1536.0 | 512.0 | 512.0 |
| 65536 | CCE chunk 4096 | 51.75 | 408.6 | 128.0 | 512.0 |

CCE's largest single allocation is 128 MB, strictly below one full logits
block: the materialized `[tokens, vocab]` tensor is never requested.
Peak-allocated delta is 2.2x / 3.8x lower than materialized.

Real Qwen3-0.6B full step (bf16, seed 0, lr 1e-4, packed 8192 tokens/context
8192, 2 warmup + 10 measured steps):

| path | step mean ms | tok/s | peak allocated MB | peak alloc delta MB |
| --- | --- | --- | --- | --- |
| materialized logits | 8320.5 | 985 | 22039.1 | 12201.9 |
| deferred head (engine default) | 8175.1 | 1002 | 18111.9 | 8270.3 |
| CCE chunk 4096 | 8360.6 | 980 | 18107.2 | 8269.8 |

CCE saves 3.9 GB peak allocated (-17.8% total, -32.2% incremental head memory)
at +0.48% step time vs materialized logits. It matches the existing deferred
head on peak and is 2.3% slower; the remaining opportunity is a fused kernel.

Public `Trainer` end-to-end (GSPO + `--cut-cross-entropy` on GSM8K, 2 steps +
save + 1-step resume). Native `train/logp_abs_diff_mean` (TensorBoard and
`train_stats` in the logs, not recomputed): 0.048248 / 0.043334 for the two
steps and 0.049003 on the resumed step. Updating steps had non-zero
`grad_norm` (7.2398, 5.3779).

Raw logs: `artifacts/logs/task1_gpu_pytest.log`,
`task1b_gpu_accuracy.log`, `task2_gpu_synthetic.log`,
`task3_gpu_model.log`, `task4_e2e_cce.log`, `task4_e2e_cce_resume.log`.

## Notes / limitations

- The pure-PyTorch reference is slower than one big matmul on both CPU and
  GPU; the GPU payoff measured here is memory, not throughput. A Triton/CUDA
  fused log-sum-exp is left as the follow-up already listed in the README.
- Full `pytest tests -q` on this host is not green for a pre-existing, unrelated
  reason: `tests/npu_stub.py` renames PyTorch's PrivateUse1 backend to `npu`,
  which makes later CUDA `backward()` calls fail with
  `PyTorch is not linked with support for npu devices`. The CCE test file is green
  in isolation (36 passed). This is documented in `REPORT.md`; the branch does
  not change test isolation.
- No push / PR was attempted from this host (no GitHub write credentials);
  the human maintainer opens the PR from this branch.
