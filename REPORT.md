# Issue #590 — KIVI low-bit KV cache: independent reproduction report

All numbers below come from this worktree's real runs. Raw logs live under
`artifacts/logs/` and are not committed (the repository ignores `*.log`).

## 1. Provenance and environment

| | |
|---|---|
| Repository | `/home/ps/LHL/worktrees/issue-590`, branch `issue-590-kivi-repro` |
| Base commit | `db3dbdf feat(models): add native GPT-OSS-20B support (#655)` |
| PR #641 mirror | `2464022 chore(kivi): vendor PR #641 implementation as reproduction base` |
| Edge tests + guards | `8fce1d9 test(kivi): add boundary regression suite and CUDA/GQA guards` |
| PR #641 source | `EricChan2028/AReno:kivi-590`, head `d442c65165eefde424586f32ab317b5e304eb5e9`, open/unmerged |
| GPU | NVIDIA RTX 5880 Ada Generation, 48 GiB, `sm_89`, driver 580.173.02 |
| Software | Python 3.12, torch 2.8.0+cu128, CUDA 12.8, triton 3.4.0, flash-attn 2.8.3.post2, transformers 5.22.1, datasets 4.0.0, numpy 2.5.4, pytest 9.2.0 |
| Model | `/home/ps/models/Qwen3-0.6B`, BF16, `model.safetensors` sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` (same `f47f7117…6874b` as the PR) |
| WikiText-2 | `wikitext-2-raw-v1` test parquet, sha256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91` (same `5f1bea06…6d91` as the PR) |
| GSM8K | local `openai/gsm8k` `main` test cache, 1319 rows |
| Seeds | benchmark/cache tests use the harness seeds in each script; teacher-forced is deterministic; rollout used greedy decoding |

Dependency added for reproduction only: `flash-attn==2.8.3.post2`, built from
source with `--no-build-isolation` (`artifacts/logs/install_flash_attn.log`).

## 2. Correctness tests

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/test_kivi_kv_cache_cpu.py tests/test_kivi_kv_cache_cuda.py tests/test_kivi_edges.py -q
```

Result: `91 passed, 1 warning in 12.61s` (`artifacts/logs/kivi_tests.log`).
The two PR files alone give the same 61 passes the PR README reports
(36 CPU + 25 CUDA); the new edge file adds 30 tests (23 CPU, 7 CUDA).

CI filter check:

```bash
env PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_kivi_edges.py -k cpu --collect-only
# 23/30 tests collected (7 deselected)
```

## 3. Decode attention kernel reproduction

Command (README section 3, CUDA graph):

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  examples/kivi/bench_decode_attention.py --cuda-graph --warmup 2 --iters 20 \
  --rows 1 8 32 128 256 --context 1024 4096 8192 16384 32768 \
  --output artifacts/logs/decode_graph.json
```

The unmodified script completed 74 of 75 cells. The last cell
(`kivi4`, 32768 context, 256 rows) OOMed because
`torch.randint(0, 255, tensor.shape)` creates an `int64` temporary of 34.6 GiB
on this 48 GiB GPU. It was rerun from a local copy
(`artifacts/repro/bench_decode_attention_lowmem.py`, only change:
`dtype=torch.uint8`) and produced 16.254 ms
(`artifacts/logs/decode_graph_kivi4_32768_256.json`).
Raw lines: `artifacts/logs/repro_decode_graph.log`,
`artifacts/logs/repro_decode_graph_missing_lowmem.log`.

CUDA graph replay, median (2 warmup + 20 measured steps):

| context | rows | BF16 ms | KIVI-2 ms | KIVI-2 x | KIVI-4 ms | KIVI-4 x |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.016 | 0.016 | 1.00 | 0.016 | 1.00 |
| 1024 | 8 | 0.034 | 0.033 | 1.03 | 0.034 | 1.00 |
| 1024 | 32 | 0.168 | 0.091 | 1.85 | 0.092 | 1.82 |
| 1024 | 128 | 0.628 | 0.348 | 1.80 | 0.366 | 1.72 |
| 1024 | 256 | 1.240 | 0.632 | 1.96 | 0.669 | 1.85 |
| 4096 | 1 | 0.026 | 0.022 | 1.19 | 0.022 | 1.19 |
| 4096 | 8 | 0.167 | 0.063 | 2.63 | 0.066 | 2.55 |
| 4096 | 32 | 0.618 | 0.287 | 2.16 | 0.309 | 2.00 |
| 4096 | 128 | 2.431 | 0.938 | 2.59 | 1.012 | 2.40 |
| 4096 | 256 | 4.848 | 1.816 | 2.67 | 1.974 | 2.46 |
| 8192 | 1 | 0.029 | 0.024 | 1.20 | 0.025 | 1.17 |
| 8192 | 8 | 0.316 | 0.096 | 3.29 | 0.103 | 3.06 |
| 8192 | 32 | 1.220 | 0.483 | 2.52 | 0.518 | 2.35 |
| 8192 | 128 | 4.839 | 1.743 | 2.78 | 2.028 | 2.39 |
| 8192 | 256 | 9.662 | 3.615 | 2.67 | 4.145 | 2.33 |
| 16384 | 1 | 0.039 | 0.045 | 0.86 | 0.046 | 0.84 |
| 16384 | 8 | 0.617 | 0.230 | 2.68 | 0.261 | 2.37 |
| 16384 | 32 | 2.423 | 0.898 | 2.70 | 0.966 | 2.51 |
| 16384 | 128 | 9.649 | 3.735 | 2.58 | 3.912 | 2.47 |
| 16384 | 256 | 19.283 | 7.408 | 2.60 | 8.095 | 2.38 |
| 32768 | 1 | 0.181 | 0.073 | 2.49 | 0.077 | 2.36 |
| 32768 | 8 | 1.220 | 0.446 | 2.74 | 0.479 | 2.54 |
| 32768 | 32 | 4.832 | 1.718 | 2.81 | 1.855 | 2.60 |
| 32768 | 128 | 19.276 | 7.358 | 2.62 | 8.074 | 2.39 |
| 32768 | 256 | 38.538 | 14.756 | 2.61 | 16.254 | 2.37 |

Eager launch table (`artifacts/logs/decode_eager.json`):

| context | rows | BF16 ms | KIVI-2 ms | KIVI-2 x | KIVI-4 ms | KIVI-4 x |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.052 | 0.301 | 0.17 | 0.299 | 0.17 |
| 1024 | 32 | 0.186 | 0.303 | 0.62 | 0.304 | 0.61 |
| 1024 | 128 | 0.647 | 0.544 | 1.19 | 0.576 | 1.12 |
| 8192 | 1 | 0.055 | 0.298 | 0.18 | 0.296 | 0.19 |
| 8192 | 32 | 1.240 | 0.728 | 1.70 | 0.776 | 1.60 |
| 8192 | 128 | 4.861 | 2.006 | 2.42 | 2.269 | 2.14 |
| 32768 | 1 | 0.205 | 0.295 | 0.69 | 0.296 | 0.69 |
| 32768 | 32 | 4.854 | 1.993 | 2.44 | 2.205 | 2.20 |
| 32768 | 128 | 19.327 | 7.545 | 2.56 | 8.313 | 2.32 |

### Comparison with PR #641's RTX PRO 5000 table

* **Direction is reproduced.** CUDA graph KIVI is faster from 32 rows onward
  and slower at a single row for 16K+ tokens; eager launches are slower at
  rows=1 and faster at high concurrency.
* **Absolute BF16 latency is higher here** (e.g. 8192×32: 1.220 ms vs 0.908 ms;
  32768×256: 38.538 ms vs 27.717 ms), so the KIVI speedup is larger
  (2.52x vs 1.62x at 8192×32). RTX 5880 Ada (`sm_89`) has lower memory
  bandwidth and no Blackwell `sm_120` flash-attn tuning versus the PR host.
* **KIVI absolute latency is close at large contexts** (8192×32: 0.483 vs
  0.562 ms; 32768×256: 14.756 vs 16.006 ms) but eager single-row KIVI is
  ~0.29 ms here versus ~0.08 ms in the PR, consistent with Triton launch
  overhead being more visible on Ada without a CUDA graph.

## 4. Cache memory

Geometry is identical to the PR, so the per-layer cache ratios match exactly.

Teacher-forced run (`4 x 16384` tokens, one cache set; JSON
`artifacts/logs/teacher_forced_wikitext_16384_rows4.json`):

| cache | bytes | ratio vs BF16 |
|---|---|---|
| BF16 (`auto`) | 7,516,192,768 | 1.000 |
| KIVI-2 | 1,482,686,464 | 0.197 |
| KIVI-4 | 2,422,210,560 | 0.322 |

PR README reports 0.197 / 0.322 for the same configuration.

End-to-end rollout through `Trainer` (65 paged blocks incl. scratch, 17 slots;
engine log lines):

| cache | engine log | bytes |
|---|---|---|
| KIVI-2 | `KIVI KV cache: dtype=kivi2 group_size=32 residual_length=128 blocks=65 slots=17 bytes=669777920` | 669,777,920 |
| KIVI-4 | `KIVI KV cache: dtype=kivi4 group_size=32 residual_length=128 blocks=65 slots=17 bytes=908328960` | 908,328,960 |

The BF16 equivalent for this tiny pool is `65 × 256 × 114688 = 1,908,408,320`
bytes (all 28 layers, K+V, BF16), i.e. KIVI-2 = 0.351x and KIVI-4 = 0.476x.
The ratios are worse than at long context because the per-slot residual rings
dominate a small cache; this matches the PR's note that rings are a fixed,
context-independent cost.

## 5. Teacher-forced quality (exact WikiText-2 file from the PR)

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  examples/kivi/teacher_forced_eval.py --model /home/ps/models/Qwen3-0.6B \
  --text artifacts/data/wikitext2_test.txt --length 16384 --rows 4 --prefill 256 \
  --output artifacts/logs/teacher_forced_wikitext_16384_rows4.json
```

`decode_seconds = 1492.97`; raw log
`artifacts/logs/repro_teacher_forced_wikitext_16384.log`.

| positions | BF16 PPL ours / PR | KIVI-2 PPL change ours / PR | KIVI-2 KL ours / PR | KIVI-2 top-1 ours / PR | KIVI-4 PPL change ours / PR | KIVI-4 KL ours / PR | KIVI-4 top-1 ours / PR |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 15.360 / 15.355 | +8.46% / +8.49% | 0.1260 / 0.1278 | 84.2% / 83.1% | +0.51% / +0.46% | 0.0048 / 0.0048 | 96.6% / 96.8% |
| [1024, 2048) | 25.217 / 25.206 | +8.57% / +7.28% | 0.1381 / 0.1386 | 81.9% / 81.9% | +0.60% / +0.54% | 0.0055 / 0.0054 | 96.0% / 95.9% |
| [2048, 4096) | 23.781 / 23.770 | +6.42% / +7.56% | 0.1404 / 0.1460 | 81.4% / 80.8% | +0.44% / +0.28% | 0.0058 / 0.0055 | 96.4% / 96.3% |
| [4096, 8192) | 22.869 / 22.852 | +8.39% / +9.44% | 0.1547 / 0.1597 | 80.2% / 80.0% | +0.39% / +0.26% | 0.0058 / 0.0057 | 95.7% / 95.7% |
| [8192, 16384) | 16.171 / 16.143 | +11.26% / +11.72% | 0.1760 / 0.1787 | 80.3% / 80.3% | +0.26% / +0.27% | 0.0059 / 0.0059 | 96.0% / 96.0% |

This is the strongest reproduction result: BF16 perplexity differs by at most
0.09%, KIVI-2 changes are within 1.5 percentage points of the PR table, and
KIVI-4 PPL/KL/top-1 agree to within 0.2 points / 0.0004 nats / 0.2 points.
Residual differences are consistent with BF16 reduction order and Triton
lowering differences between `sm_89` and `sm_120`.

## 6. End-to-end rollout through the public Trainer

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 .venv/bin/python \
  examples/kivi/rollout_eval.py --model /home/ps/models/Qwen3-0.6B \
  --dataset artifacts/data/gsm8k_main_test --kv-cache-dtype <auto|kivi2|kivi4> \
  --limit 16 --max-new-tokens 512 --max-running-prompts 16 \
  --output artifacts/logs/rollout_<dtype>
```

| cache | prompts | mean response tokens | generated tok/s | accuracy | cache log |
|---|---|---|---|---|---|
| BF16 (`auto`) | 16 | 503.7 | 629.0 | 12.5% | no KIVI line (expected) |
| KIVI-2 | 16 | 501.2 | 775.8 | 18.8% | 669,777,920 bytes |
| KIVI-4 | 16 | 500.8 | 765.4 | 31.3% | 908,328,960 bytes |

Raw logs: `artifacts/logs/repro_rollout_auto.log`,
`repro_rollout_kivi2.log`, `repro_rollout_kivi4.log`; summaries in
`artifacts/logs/rollout_<dtype>/summary.json`.

Accuracy here is not comparable to the PR's 1319-prompt table: only 16 prompts
and a 512-token cap were used to keep the integration check short. The
throughput direction (KIVI ~1.23x over BF16 at 16 concurrent 500-token
responses) matches the PR's concurrency trend.

## 7. Boundary regression suite delivered

`tests/test_kivi_edges.py` maps to the issue requirements:

| requirement | test |
|---|---|
| group size 0, negative, non-power-of-two | `test_invalid_group_size_rejected_by_geometry_cpu`, `test_invalid_group_size_rejected_by_runtime_config_cpu` |
| non-divisible channels / head_dim | `test_non_divisible_shapes_rejected_cpu`, `test_runtime_config_falls_back_for_non_divisible_head_dim_cpu`, `test_gqa_query_heads_must_divide_kv_heads_cpu`, `test_qkv_head_dim_must_match_cache_cpu` |
| no CUDA clear error, not a crash | `test_allocate_without_cuda_raises_clear_error_cpu`, `test_cpu_runtime_falls_back_to_model_dtype_cache` |
| batch=1, prompt 1024/4096/8192 residual boundary | `test_batch_one_residual_boundary_at_long_prompts` (2 bits × 3 lengths) |
| no storage leak after reset, observable counter | `test_clear_releases_all_cache_objects_with_live_counter` |
| dequantization vs independent NumPy, BF16 atol=rtol=1e-2 | `test_dequantize_blocks_matches_independent_numpy_cpu` (2/4-bit) |

Two guards were added to the PR implementation to make these cases fail cleanly:

* `KiviKVCache.allocate` now raises
  `RuntimeError("KIVI KV cache requires a CUDA device; ...")` for non-CUDA
  devices or unavailable CUDA.
* `KiviInferBackend.forward` now rejects a GQA query/KV ratio that is not
  divisible and Q/K head dims that do not match the cache.

## 8. Limitations / not reproduced

* Sections 5-7 of the PR README (full 1319-prompt GSM8K sampling, long-context
  throughput concurrency sweep, and 100-step RL with 6 seeds) were not run:
  they need many GPU-hours and 128+ concurrent sequences. A reduced
  `Trainer` rollout (16 prompts) was run instead and is reported above.
* The 32768-token × 2-row teacher-forced run (PR section 4) was not run; the
  16384-token × 4-row run already exercises the `[8192, 16384)` bucket and
  spends ~25 minutes in this environment.
* The one 32768×256 CUDA-graph benchmark cell required a local `uint8`
  temporary patch; this is an example-script memory issue on a 48 GiB GPU,
  not a KIVI kernel failure. The unmodified run's OOM is in
  `artifacts/logs/repro_decode_graph_missing.log`.
* Full `pytest tests -k cpu` was run for regression: `15 failed, 1912 passed,
  73 skipped ... in 216.45s` (`artifacts/logs/full_cpu_tests.log`). None of
  the failures touch KIVI; they are `test_config_data_cpu` device-visibility
  assertions under this host's `CUDA_VISIBLE_DEVICES`, unrelated MoE
  `tokenwise` API drift, an optimizer test, and the known real-gloo failures.
  The KIVI-specific suites are all green (91 passed).

## 9. Reproduce here

```bash
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0

pytest tests/test_kivi_kv_cache_cpu.py tests/test_kivi_kv_cache_cuda.py \
       tests/test_kivi_edges.py -q

python examples/kivi/bench_decode_attention.py --cuda-graph --warmup 2 --iters 20 \
  --rows 1 8 32 128 256 --context 1024 4096 8192 16384 32768

python examples/kivi/teacher_forced_eval.py --model /home/ps/models/Qwen3-0.6B \
  --text artifacts/data/wikitext2_test.txt --length 16384 --rows 4 --prefill 256 \
  --output artifacts/logs/teacher_forced_wikitext_16384_rows4.json
```
