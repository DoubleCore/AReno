## Summary

Independent reproduction of PR #641 (KIVI 2/4-bit KV cache) on a single
RTX 5880 Ada 48 GiB (`sm_89`) host, plus the boundary regression suite
requested by issue #590.

* Adds `tests/test_kivi_edges.py` with 30 tests:
  * invalid KIVI group sizes (`0`, negative, odd, non-power-of-two);
  * non-divisible `head_dim` / block size, non-divisible GQA head ratio and
  * Q/K head dim / cache head dim mismatch;
  * clear `RuntimeError` when KIVI storage is requested without CUDA;
  * batch=1 residual-window boundaries at 1024/4096/8192-token prompts;
  * reset leak check using a live `KiviKVCache` object counter plus
    `torch.cuda.memory_allocated()` baseline;
  * element-wise dequantization against an independent NumPy implementation
    (BF16 `atol = rtol = 1e-2`).
* Adds two small guards to the KIVI backend so the no-CUDA and non-divisible
  GQA cases fail with clear errors instead of crashing or silently truncating:
  `KiviKVCache.allocate` and `KiviInferBackend.forward`.
* CPU tests carry a `_cpu` suffix so the existing CI filter
  (`pytest tests/ -k cpu`) picks them up; CUDA tests skip when CUDA is absent.

## Base / stacking

This branch is stacked on PR #641. The first commit
(`2464022 chore(kivi): vendor PR #641 implementation as reproduction base`)
mirrors `EricChan2028/AReno:kivi-590` head
`d442c65165eefde424586f32ab317b5e304eb5e9` so the worktree can run while #641
is unmerged. The fork ref is not fetchable from this host, so files were
obtained through the GitHub contents API. The inference cache-allocation hunk
was adapted to keep current `main`'s `DecodeMetadata` decode path (#643 landed
after the PR branch was cut). Reviewers should treat the first commit as
#641 content and focus on the edge tests + guards in the second commit.
Once #641 merges, this branch can be rebased to drop the mirror commit.

## Verification (all real runs, logs under `artifacts/logs/`)

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/test_kivi_kv_cache_cpu.py tests/test_kivi_kv_cache_cuda.py tests/test_kivi_edges.py -q
# 91 passed, 1 warning in 12.61s
```

The two PR test files alone give 61 passes, matching the PR README (36 CPU +
25 CUDA); `test_kivi_edges.py` adds 30 (23 selected by `-k cpu`, 7 CUDA).

* Decode benchmark, CUDA graph, 1 layer 16 Q / 8 KV heads, head dim 128,
  G=32, R=128, 2 warmup + 20 measured steps, context 1024→32768, rows 1→256:
  directions match the PR, but absolute BF16 latency is higher on `sm_89`,
  so KIVI speedups are larger (e.g. 8192×32 2.52x here vs 1.62x in the PR).
  One 32768×256 KIVI-4 cell OOMed in the unmodified example because
  `torch.randint` created a 34.6 GiB `int64` temporary; reran with a local
  `uint8` temporary copy and got 16.254 ms.
* Cache bytes: KIVI-2 = 0.197x and KIVI-4 = 0.322x of BF16 at 4×16384 tokens,
  exactly the PR ratios.
* Teacher-forced WikiText-2, Qwen3-0.6B, 4×16384, same parquet sha256 as the
  PR: BF16 PPL within 0.09%; KIVI-4 PPL/KL/top-1 within 0.2 points /
  0.0004 nats / 0.2 points; KIVI-2 PPL change within 1.5 points of the PR
  table. Table and raw paths are in `REPORT.md`.
* End-to-end `Trainer` rollout (reduced to 16 prompts × 512 tokens) exercises
  the public path, emits the expected `KIVI KV cache: ... bytes=...` line, and
  shows KIVI ≈1.23x over BF16 at 16 concurrent responses.

## Checklist

- [x] Scope limited to `tests/test_kivi_edges.py`, two guards, and the
      reproduction base; no unrelated framework changes.
- [x] CPU tests run under the existing CI `-k cpu` filter.
- [x] CUDA tests skip without CUDA.
- [x] No fabricated numbers; `REPORT.md` links every raw log path.
- [x] Not pushed from this host (no GitHub write credentials here).
