## What does this PR do?

Adds external-behaviour tests for issue #592 (multi-sample shared prompt KV):
numerical equivalence, block lifecycle, cancellation/boundaries, and a
baseline-vs-shared performance/memory matrix.

This is a **test-only** contribution. It is stacked on PR #639
(`feat(runtime): share prompt KV across rollout samples`) and does not modify
code under `areno/`.

## Related issue

Refs #592. Depends on #639.

## What is covered

- Greedy token ids and log-probs are identical between independent caches and
  shared prompt KV for prompt lengths 1024/4096 x samples 1/4/8 (all 4,992 BF16
  response log-probs compared, max absolute error 0).
- FP32 replay matches exactly at `atol=rtol=1e-5` for 1024/4 and 4096/4.
- Observable lifecycle counters: prompt blocks are stored once, refcount equals
  the sample count, blocks stay owned until the last sample exits, and the free
  pool/recurrent slots fully recover.
- Cancellation: cancelling one sample releases only its reference; the engine
  can immediately run another shared rollout.
- Fallbacks: `n_samples=1`, differing prompts, running/prefill budget overflow,
  unsupported model, TP, or device all fall back to independent caches.
- CUDA tests also compare 256/512-token prompts x samples 1/4/8.

## Memory evidence

BF16, warmup 2 + 10 measured steps, warmup/measure identical fixed-token replay:

| Prompt / samples | Peak allocated GiB baseline → shared | Live full prompt blocks baseline → shared | Pool blocks baseline → shared |
| --- | --- | --- | --- |
| 1024 / 4 | 3.96 → 3.63 | 16 → 4 | 20 → 8 |
| 1024 / 8 | 5.67 → 4.91 | 32 → 4 | 40 → 12 |
| 4096 / 4 | 8.77 → 7.46 | 64 → 16 | 68 → 20 |
| 4096 / 8 | 15.30 → 12.23 | 128 → 16 | 136 → 24 |

For 4096 x 8, the full prompt blocks drop from 128 to 16 copies while prefill
and decode throughput stay essentially unchanged; this is a memory optimization,
not a throughput claim.

## How was it tested?

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q \
  tests/test_shared_prompt_kv.py
# 19 passed in 19.76 s

env CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD .venv/bin/python -m pytest tests -q
# 3 failed, 1567 passed, 2241 skipped
```

The three failures are the pre-existing MoE fixture tests already recorded in
PR #639's results; none involve shared prompt KV.

Full commands, environment, raw `coordinator.jsonl` / `worker.jsonl` paths and
the complete results tables are in `REPORT.md`.

## Limitations

- Single GPU, Qwen3-0.6B, native attention, fixed-token replay. No training
  quality or end-to-end throughput claim.
- The branch targets the PR #639 head. Current `main` additionally caps
  `max_prefill_tokens` at 8192 (which forces 4096-token groups with >2 samples
  back to independent caches) and routes decode through `DecodeMetadata`
  (which currently drops copy-on-write tail copies). Those are integration
  changes for the feature PR, not patched here.

## Checklist

- [x] Added focused external-behaviour tests.
- [x] Required numerical matrix and FP32 check recorded from real runs.
- [x] Lifecycle/cancellation/fallback boundaries covered.
- [x] Performance and memory measured with warmup 2 / 10 steps.
- [x] Raw logs referenced in `REPORT.md`.
- [ ] Not based on current `main`; merge after #639 and resolve the two drifts.
