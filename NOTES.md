# Issue #592 notes

## Scope / base

- Deliverable branch: `issue-592-shared-kv-tests`.
- The feature core is PR #639 (`feat(runtime): share prompt KV across rollout
  samples`). This branch is stacked directly on that head so the tests exercise
  the PR as submitted, without modifying `areno/`.
- Upstream `main` had moved 23 commits past the PR merge base. A trial merge
  showed two integration drifts that are outside this test-only task:
  1. `max_prefill_tokens` is now capped at 8192, so 4096-token groups fall back
     to independent caches;
  2. the newer `DecodeMetadata` path drops copy-on-write tail copies.
  We therefore run the required 4096-token matrix on the PR head. The drift is
  recorded in REPORT.md, not patched here.

## External surface exercised

- `RuntimeConfig.share_prompt_kv` (opt-in, default false).
- `ArenoEngine.generate_rollout(..., prompt_group_size=N)`.
- `RolloutOutput.response_ids`, `.logprobs`, `.finish_reason`.
- `InferenceBatchState` observable block counters used as lifecycle oracles:
  `_seq_to_blocks`, `_block_refcounts`, `_free_blocks`,
  `_free_recurrent_slots`.

## Tests

- CPU lifecycle: shared prompt blocks are stored once, refcount equals sample
  count, blocks return to the free pool only after the last owner exits.
- CPU cancellation/refcount: cancelled rows release only their private
  reference; the shared block survives until the last owner finishes.
- CPU fallbacks: `n_samples=1`, differing prompts, running-budget/prefill
  budget overflow, unsupported model/TP all fall back to independent caches.
- CUDA correctness: baseline vs shared over prompt lengths 256/512 and
  sample counts 1/4/8, greedy tokens and log-probs compared.
- CUDA lifecycle: cancelling one sample, then reusing the same shared engine
  for a fresh rollout, proves the released pool is reusable.

## Matrix experiment

`benchmarks/shared_prompt_kv.py` / `benchmarks.compare_shared_prompt_kv` run
in isolated worker processes: 1024/4096 prompt tokens x 1/4/8 samples, BF16,
warmup 2 + 10 measured steps, fixed-token replay so both arms see identical
token ids. A separate FP32 replay checks the stricter tolerance. Raw
`coordinator.jsonl` / `worker.jsonl` / `comparison.json` live under
`artifacts/logs/issue-592-matrix/`.
