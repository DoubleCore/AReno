# Issue #592 — shared prompt KV: correctness, lifecycle, and performance

## Branch and base

- Branch: `issue-592-shared-kv-tests`.
- Feature core: PR #639 `feat(runtime): share prompt KV across rollout samples`
  (upstream head `f44b8a4f4572ef4b495499caf0e9c99481cdf5c8`, tree
  `d4e6bd5428e7510423768abe0d62ec3e258db5bb`, parent `25f5fed`).
- This test branch is stacked directly on the PR head. The PR was imported as a
  local commit `6de6453f843c60d93dad7a90fc1bf984c92a061b` because the machine
  cannot fetch the fork over git; its tree is byte-identical to the PR head.
  No code under `areno/` is modified by this task.

### Why the tests run on the PR head instead of current `main`

A trial merge onto the worktree's `main` (`db3dbdf`) exposed two integration
drifts that belong to the feature owner, not to this test-only issue:

1. Current `main` caps `max_prefill_tokens` at 8192. A 4096-token group with
   >2 samples therefore falls back to independent caches, so the required
   4096 x 4/8 shared matrix cannot run there.
2. Current `main` routes ordinary decode through `DecodeMetadata`, which calls
   `state.ensure_decode_blocks()` but drops the returned copy-on-write tail
   copies, so non-block-aligned shared prompts would silently share a tail.

Neither condition exists on the PR head, so all required measurements below are
taken on the PR head exactly as submitted. This branch should be reviewed/merged
after #639.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-11-04 (UTC+08:00) |
| GPU | 1 x NVIDIA RTX 5880 Ada Generation, 49140 MiB, driver 580.173.02 |
| CUDA visible device | `CUDA_VISIBLE_DEVICES=1` (single GPU) |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Model | `/home/ps/models/Qwen3-0.6B` (Qwen3 dense, tied embeddings) |
| Model safetensors SHA-256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Metadata model revision | `local-qwen3-0.6b` |
| Attention backend | native |
| Sampling | greedy / fixed-token replay, seed 1234 |
| Benchmark shape | `max_new_tokens=16`, warmup 2, 10 measured steps |

## Deliverable test file

`tests/test_shared_prompt_kv.py` (19 tests) exercises only the external rollout
surface plus the scheduler's observable block counters:

- opt-in default (`RuntimeConfig.share_prompt_kv is False`);
- shared prompt blocks stored once, refcount = sample count;
- blocks return to the free pool only after the last owner exits;
- decode response blocks are private per sample;
- mid-rollout cancellation releases only the cancelled row's references;
- `n_samples=1`, differing prompts, running-budget/prefill-budget overflow,
  unsupported model/TP/device fall back to independent caches;
- CUDA baseline-vs-shared equality for prompt lengths 256/512 and samples 1/4/8;
- CUDA cancellation followed by a fresh shared rollout (pool reuse).

Commands and raw results:

```bash
# CPU-only selection used during development
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD .venv/bin/python -m pytest -q \
  tests/test_shared_prompt_kv.py -k "not cuda"
# 12 passed, 1 skipped

# Full file, including the CUDA gates on GPU 1
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q \
  tests/test_shared_prompt_kv.py
# 19 passed in 19.76 s
```

Full CPU suite (same environment):

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD .venv/bin/python -m pytest tests -q
# 3 failed, 1567 passed, 2241 skipped
```

The three failures are the pre-existing MoE fixture tests
(`tests/test_moe_sequence_parallel_cpu.py`, 3 cases). PR #639's own results
document records the same three failures on the unchanged target branch; none
involve shared prompt KV. Log:
`artifacts/logs/issue-592-cpu-suite.log`.

## Required BF16 matrix

Raw logs: `artifacts/logs/issue-592-matrix-bf16/` (one directory per case;
`coordinator.jsonl`, `worker.jsonl`, `metadata.json`, `comparison.json`).
Runner used: `artifacts/logs/issue-592-run.sh`; full console log:
`artifacts/logs/issue-592-matrix-bf16.log`.
The comparison compares every response id and every response log-prob across
all 12 iterations (including warmups) with `atol=rtol=1e-2`.

| Prompt / samples | Response ids | Compared log-probs | Max abs logprob error |
| --- | --- | --- | --- |
| 1024 / 1 | identical | 192 | 0.0 |
| 1024 / 4 | identical | 768 | 0.0 |
| 1024 / 8 | identical | 1536 | 0.0 |
| 4096 / 1 | identical | 192 | 0.0 |
| 4096 / 4 | identical | 768 | 0.0 |
| 4096 / 8 | identical | 1536 | 0.0 |

## FP32 numerical check

Same fixed-token replay, `--dtype fp32`, no warmup, one measured step,
`atol=rtol=1e-5`.

| Prompt / samples | Response ids | Compared log-probs | Max abs logprob error |
| --- | --- | --- | --- |
| 1024 / 4 | identical | 64 | 0.0 |
| 4096 / 4 | identical | 64 | 0.0 |

Raw: `artifacts/logs/issue-592-matrix-bf16/fp32-prompt*/`.

## Performance and memory (BF16, 2 warmups + 10 measured steps)

All values are medians/maxima over the measured iterations from
`worker.jsonl` and `coordinator.jsonl`; `baseline / shared` order.

| Prompt / samples | Peak allocated GiB | Peak reserved GiB | Median wall s | Pool blocks | Peak live full prompt blocks |
| --- | --- | --- | --- | --- | --- |
| 1024 / 1 | 2.67 / 2.67 | 2.81 / 2.81 | 0.861 / 0.856 | 5 / 5 | 4 / 4 |
| 1024 / 4 | 3.96 / 3.63 | 4.18 / 3.80 | 2.395 / 2.436 | 20 / 8 | 16 / 4 |
| 1024 / 8 | 5.67 / 4.91 | 6.08 / 5.26 | 4.460 / 4.461 | 40 / 12 | 32 / 4 |
| 4096 / 1 | 3.87 / 3.87 | 4.07 / 4.07 | 8.143 / 8.098 | 17 / 17 | 16 / 16 |
| 4096 / 4 | 8.77 / 7.46 | 9.53 / 8.20 | 30.998 / 30.931 | 68 / 20 | 64 / 16 |
| 4096 / 8 | 15.30 / 12.23 | 16.72 / 13.62 | 61.711 / 61.652 | 136 / 24 | 128 / 16 |

Prefill / decode throughput (median physical prefill tokens/s, decode tokens/s):

| Prompt / samples | Prefill tok/s baseline → shared | Decode tok/s baseline → shared |
| --- | --- | --- |
| 1024 / 1 | 2140 → 2131 | 42.5 → 42.5 |
| 1024 / 4 | 2149 → 2133 | 159.9 → 148.0 |
| 1024 / 8 | 2084 → 2072 | 309.3 → 295.2 |
| 4096 / 1 | 535 → 536 | 42.3 → 40.6 |
| 4096 / 4 | 538 → 539 | 126.5 → 135.0 |
| 4096 / 8 | 539 → 539 | 167.1 → 175.3 |

### Evidence that full prompt blocks are stored only once

For prompt-aligned cases the worker records live block categories directly:

- 1024-token prompt = 4 blocks. `prompt_full` peaks at 16 / 4 / 32 blocks for
  4/8 independent samples, but only 4 shared blocks for both 4 and 8 samples.
- 4096-token prompt = 16 blocks. `prompt_full` peaks at 64 / 128 blocks for
  4/8 independent samples, but only 16 shared blocks for both 4 and 8 samples.
- Pool usage confirms the accounting: 4096 x 8 uses 136 blocks baseline
  (32 prompt blocks per sample plus response blocks) versus 24 shared
  (16 prompt blocks once + 8 private response blocks).

## Interpretation / limits

- Numerical agreement is exact in this fixed-token replay; the implementation
  deliberately preserves the independent-request prefill compute shape.
- This establishes a KV-memory reduction, not a prefill/decode speedup:
  physical prefill token counts and throughput are essentially unchanged.
- Measurements are single-GPU, native-attention, Qwen3-0.6B, BF16/FP32 fixed
  replay. They do not measure sampled training quality or end-to-end RL
  throughput.
- The two current-`main` integration drifts listed above are not patched here;
  they should be resolved in the feature PR before rebasing this test branch.

## Raw log index

- `artifacts/logs/issue-592-matrix-bf16.log` — full benchmark console log.
- `artifacts/logs/issue-592-matrix-bf16/` — BF16 matrix and FP32 replay.
- `artifacts/logs/issue-592-cpu-suite.log` — full CPU pytest run.
- `artifacts/logs/issue-592-run.sh` — exact matrix runner used.
