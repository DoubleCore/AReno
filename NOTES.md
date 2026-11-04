# Issue #590 — KIVI independent reproduction notes

## Scope and provenance

PR #641 (`EricChan2028/AReno:kivi-590`, head `d442c65165eefde424586f32ab317b5e304eb5e9`)
is unmerged and was not present in `main` at `db3dbdf`. `git fetch` of that fork
branch is blocked on this host, so the PR files were obtained through the
GitHub REST contents API and vendored into this worktree as the runnable base.
The only new feature work in this branch is:

* `tests/test_kivi_edges.py` — the boundary regression suite requested by issue #590.
* Two small guards needed by those tests:
  * `KiviKVCache.allocate` raises a clear `RuntimeError` when requested on a
    non-CUDA/unavailable-CUDA device instead of failing inside `torch.empty`.
  * `KiviInferBackend.forward` rejects a GQA query/KV head ratio or Q/K head
    dim that does not match the cache geometry instead of silently truncating.
* A one-line memory fix in the local reproduction copy of
  `bench_decode_attention.py` (kept under `artifacts/repro/`, not committed):
  `torch.randint` is created as `uint8` rather than the default `int64`, which
  otherwise allocates a 34.6 GiB temporary at 256 rows and OOMs a 48 GiB GPU.

## Interface boundaries touched by the tests

| Boundary | Contract |
|---|---|
| `RuntimeConfig.kv_cache_dtype` | `auto` (model dtype), `kivi2`, `kivi4`; default `auto` |
| `RuntimeConfig.kivi_group_size` | power of two, `>= 32` |
| `RuntimeConfig.kivi_residual_length` | non-negative full-precision tail |
| `RuntimeConfig.resolve_kv_cache_dtype` | warns `RuntimeWarning` and falls back to `auto` when CUDA/model/speculative/head-dim/block geometry cannot support KIVI |
| `KiviSpec` | `bits`, `group_size`, `residual_length`; `ring_size = residual + group`, `quantized_length(n) = max(0, floor((n-1-R)/G)*G)` |
| `KiviKVCache.allocate` | CUDA only; validates bits/group/residual/head-dim/block-size geometry |
| `attach_kivi_kv_caches` | one cache per `model.layers[i].self_attn`, owned by the runtime |
| `CausalSelfAttention.set_kivi_cache` / `clear_kv_cache` | attach/detach; clear must release all tensors |
| `KiviInferBackend.forward` | `prefill` delegates fresh-token attention to the dense backend, `decode` uses the fused Triton append + split-K kernels |
| `dequantize_blocks` | token-major `(length, kv_heads, head_dim)` dequantization from paged packed codes |

## Boundary cases and test design

1. **Group size**: `0`, negative (`-1`, `-8`), odd (`3`), and non-powers of two
   (`24`, `48`, `100`) must fail both `validate_kivi_geometry` and
   `RuntimeConfig` with a clear `ValueError`.
2. **Non-divisible shapes**: `head_dim` 96/80/64 vs group 32/128, and
   block size 64 vs group 128 must fail geometry validation; the runtime must
   warn and fall back. GQA query/KV head ratios and Q/K head dims must fail at
   the backend boundary rather than being floor-divided.
3. **No CUDA**: allocation on `cpu`, and on `cuda` when
   `torch.cuda.is_available()` is false, must raise a clear `RuntimeError`.
   `RuntimeConfig.resolve_kv_cache_dtype` must warn and fall back.
4. **Residual window with batch=1**: prompts of exactly 1024/4096/8192 tokens
   leave `R+G = 160` ring tokens after prefill; the first decode step flushes
   the oldest group and reuses its ring slot. Decode `2G+1` steps and compare
   attention at steps 0, 1, G-1, G, 2G-1, 2G against an independent NumPy
   dequantization + softmax; also assert the final ring contents.
5. **Reset leak counter**: count live `KiviKVCache` objects through
   `gc.get_objects()` using their object ids, clear the attention layers and
   assert the count returns to zero and `torch.cuda.memory_allocated()` returns
   to its pre-attach baseline.
6. **Independent NumPy dequantization**: implement the documented packing
   layout (`byte j`, field `i` at bit `i*bits`, channel `j*8/bits+i`) and the
   K per-channel / V per-token scale-zero formulas in NumPy; compare element
   wise with `dequantize_blocks` after both are cast to BF16 at
   `atol = rtol = 1e-2`.

All CPU tests are named with a `_cpu` suffix so the existing CI filter
`pytest tests/ -k cpu` selects them. GPU tests are skipped when CUDA is absent.

## Reproduction plan

* `pytest tests/test_kivi_kv_cache_cpu.py tests/test_kivi_kv_cache_cuda.py tests/test_kivi_edges.py`
* `examples/kivi/bench_decode_attention.py --cuda-graph` for the section-3
  decode table, plus the eager table.
* `examples/kivi/teacher_forced_eval.py` on the exact WikiText-2 parquet used
  by the PR (sha256 `5f1bea06…6d91`) at 16384 tokens × 4 rows.
* `examples/kivi/rollout_eval.py` through the public `Trainer` as an
  end-to-end integration check (reduced prompt count).
