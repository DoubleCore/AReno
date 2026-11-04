# Issue #588 — GPU-side validation of cut cross-entropy (CCE)

Branch: `issue-588-cce-gpu`, cut from PR #657's `issue-588-cce` at
`eaaae93ea16f29970ddec19ff0bd7511bd6d6cb9` (`feat(accel): add opt-in
cut cross-entropy LM-head path (#588)`). The GPU-validation commit on this
branch is `1bdc611` (`test(accel): add CUDA validation and GPU benchmarks for
cut cross-entropy (#588)`). This branch only adds GPU tests,
the real-machine benchmark script, the README GPU table and this report; the
CCE arithmetic, the default-off frontend switch and the explicit
`NotImplementedError`s are unchanged.

All numbers below are copied from the raw logs under `artifacts/logs/`.
Nothing was hand-computed or imputed.

## Environment

| item | value |
| --- | --- |
| host GPU | NVIDIA RTX 5880 Ada Generation (sm_89, 48 GB), single card `CUDA_VISIBLE_DEVICES=0` |
| driver / CUDA | CUDA 12.8 (`torch.version.cuda=12.8`), cuDNN 91002 |
| python | CPython 3.12.3 |
| torch | 2.8.0+cu128 |
| triton | 3.4.0 |
| transformers | 5.22.1 |
| datasets | 4.0.0 |
| pytest | 9.2.0 |
| `allow_tf32_matmul` | `false` (all runs) |
| model | `/home/ps/models/Qwen3-0.6B`, hidden=1024, vocab=151936, 28 layers, `tie_word_embeddings=true`, bf16 |
| model revision | local snapshot has no `.git`; `model.safetensors` sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`, `config.json` sha256 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| GSM8K revision (e2e only) | `740312add88f781978c0658806c59bc2815b9866` (from the HF request log) |
| import path | `PYTHONPATH=$PWD`, so `areno` resolves to this worktree |

`import areno` was checked against this worktree for every run via
`PYTHONPATH=$PWD`; the extension is the pre-built
`areno/accel/_areno_accel.cpython-312-x86_64-linux-gnu.so`.

## 1. GPU numerical acceptance — `tests/test_cut_cross_entropy.py`

Raw log: `artifacts/logs/task1_gpu_pytest.log`

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m pytest tests/test_cut_cross_entropy.py -v
```

Result: **36 passed in 3.26 s**. The two tests that are skipped without CUDA
now run and pass:

- `test_peak_memory_below_materialized_logits_cuda` — PASSED
- `test_config_defaults_keep_cut_cross_entropy_off` — PASSED (native backend
  import available on this host; `RuntimeConfig().cut_cross_entropy is False`
  and the explicit `cut_cross_entropy=True` opt-in resolves to `True`)

This branch also adds a CUDA-only packed-head test
(`test_packed_head_cut_cross_entropy_matches_full_projection_cuda`), parametrized
over FP32 (outputs and gradients) and BF16 (outputs), so the deferred head
integration is exercised on the GPU rather than only the CPU.

### Measured FP32 / BF16 error at full vocabulary

Raw log: `artifacts/logs/task1b_gpu_accuracy.log`

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python examples/cce/bench_gpu.py --mode accuracy \
  --tokens 4096 --hidden 1024 --vocab 151936 --chunk 4096 --seed 0
```

Reference is `F.cross_entropy(hidden @ weight.T, targets)`; per-row values use
`reduction="none"`. The pinned test tolerances are `atol=rtol=1e-5` for FP32
and `rtol=1e-2` for BF16.

| dtype | max abs | mean abs | max rel | mean rel |
| --- | --- | --- | --- | --- |
| FP32 rows | 1.678e-04 | 1.360e-05 | 2.121e-06 | 1.065e-07 |
| BF16 rows | 8.452e-01 | 3.742e-02 | 1.173e-02 | 2.950e-04 |

Scalar loss (full-vocabulary):

| dtype | reference | CCE | abs diff | rel diff |
| --- | --- | --- | --- | --- |
| FP32 | 142.491364 | 142.491379 | 1.526e-05 | 1.071e-07 |
| BF16 | 144.433487 | 144.433350 | 1.373e-04 | 9.508e-07 |

The pinned BF16 tolerance is `rtol=1e-2`; 99.98% of rows are inside it. The
largest per-row relative error (1.17e-2) is on individual rows, while the
aggregate loss agrees to 9.5e-7 relative. This is the expected FP32-accumulation
vs BF16-logit-rounding difference: the reference rounds the `[tokens, vocab]`
logits to BF16, CCE scores the target dot product and the log-sum-exp in FP32.

## 2. Synthetic output-head memory — materialized vs CCE

Raw log: `artifacts/logs/task2_gpu_synthetic.log`

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python examples/cce/bench_gpu.py --mode synthetic \
  --tokens 4096 --hidden 1024 --vocab <32768|65536> --chunk 4096 \
  --dtype bfloat16 --path <materialized|cce> --warmup 2 --iters 10 --record-alloc
```

`peak alloc delta` is `max_memory_allocated()` minus the post-warmup baseline,
measured across forward **and backward**. `max single alloc` is the largest
single CUDA `alloc` event recorded by `torch.cuda.memory._record_memory_history`
in one forward+backward; it is the direct test for “did this path allocate one
`tokens x vocab` tensor?”.

| tokens | hidden | vocab | dtype | path | loss | fwd+bwd median ms | peak alloc delta MB | max single alloc MB | full logits block MB | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 32768 | bf16 | materialized | 133.000000 | 7.21 | 768.0 | 256.0 | 256.0 | reference |
| 4096 | 1024 | 32768 | bf16 | CCE (chunk 4096) | 133.256012 | 23.16 | **344.3** | **128.0** | 256.0 | no full block |
| 4096 | 1024 | 65536 | bf16 | materialized | 138.000000 | 15.73 | 1536.0 | 512.0 | 512.0 | reference |
| 4096 | 1024 | 65536 | bf16 | CCE (chunk 4096) | 138.083328 | 51.75 | **408.6** | **128.0** | 512.0 | no full block |

Interpretation:

- The materialized path allocates exactly one full logits block as a single
  allocation (256 MB / 512 MB), as expected.
- CCE's largest single allocation is 128 MB in both cases, strictly below one
  `tokens x vocab` block, so the complete logits tensor is never requested.
  (The 128 MB is an internal autograd/workspace allocation, not the full
  `tokens x vocab` block.)
- Peak-allocated delta is 2.2x lower for vocab=32768 and 3.8x lower for
  vocab=65536. Peak includes gradients, checkpoint recompute buffers and
  transient BLAS workspace, so it is larger than one block even when no single
  block is allocated. This is why the proof uses the max single allocation
  rather than the aggregate peak.

### Chunk-size sweep (vocab=65536, bf16)

| CCE vocab chunk | loss | fwd+bwd median ms | peak alloc delta MB | max single alloc MB |
| --- | --- | --- | --- | --- |
| 1024 | 138.083328 | 85.83 | 268.0 | 128.0 |
| 2048 | 138.083328 | 56.76 | 273.1 | 128.0 |
| 4096 | 138.083328 | 51.75 | 408.6 | 128.0 |
| 8192 | 138.083328 | 57.20 | 680.3 | 256.0 |

Smaller chunks reduce peak but cost more Python-loop time; 4096 is the fastest
in this sweep. 8192 doubles the largest transient allocation (256 MB) but is
still half a full logits block.

## 3. Real model Qwen3-0.6B — full training step peak and throughput

Raw log: `artifacts/logs/task3_gpu_model.log`

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python examples/cce/bench_gpu.py --mode model \
  --ckpt /home/ps/models/Qwen3-0.6B --seqs 1 --seq-len 8192 --prompt-len 64 \
  --path <materialized|deferred|cce> --warmup 2 --iters 10 \
  --seed 0 --lr 1e-4 --attn-backend native
```

The step mirrors `TrainingManager` at `tp_size=1`: packed forward with
`defer_lm_head` (MST chunk 256, activation checkpointing), the selected head
path, `sft_loss_fn`, backward, FP32 main-gradient folding, and
`AdamWFP32Master.step()`. Fixed configuration: bf16, seed 0, lr 1e-4,
8192 packed tokens / 8191 action rows, one sequence of context 8192,
`attn_backend=native` (flash-attn is not installed on this host).

| path | loss | logprob mean | step median ms | step mean ms | tok/s | peak allocated MB | peak alloc delta MB | max reserved MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| materialized logits | 10.977714 | -10.9874 | 8316.0 | 8320.5 | 985 | 22039.1 | 12201.9 | 22550.0 |
| engine default deferred head (chunk 256) | 11.008770 | -11.0182 | 8167.4 | 8175.1 | 1002 | 18111.9 | 8270.3 | 19772.0 |
| **CCE (vocab chunk 4096)** | 10.909619 | -10.9209 | 8364.9 | 8360.6 | 980 | **18107.2** | **8269.8** | 19954.0 |

`materialized` step mean 8320.5 ms (10 measured steps after 2 warmup steps).
`CCE` is 8360.6 ms (+0.48% vs materialized) at 980 tok/s, while peak allocated
memory drops by **3931.9 MB** (22.0 GB -> 18.1 GB, -17.8% of total peak and
-32.2% of the incremental head memory). CCE uses the same peak as the existing
token-chunked deferred path (delta differs by 0.5 MB) and is 2.3% slower than it;
this branch is a pure-PyTorch reference, so the remaining win over the default
path is still a fused kernel (see Follow-ups).

Loss/logprob values above are from the optimizer-updated state after the
warmup+measured steps, not a same-weight parity check: the three paths take
slightly different BF16 gradients and therefore drift apart. Numeric parity
at identical weights is covered by the CUDA/CPU tests and by section 1.

## 4. Public `Trainer` end-to-end (GSPO, update + save/resume)

Raw logs:

- `artifacts/logs/task4_e2e_cce.log`
- `artifacts/logs/task4_e2e_cce_resume.log`
- TensorBoard scalars: `artifacts/tb/e2e_cce/`, `artifacts/tb/e2e_cce_resume/`

Run 1 (2 steps, save every step):

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python -m areno.cli.train \
  --algo gspo --backend cuda --ckpt /home/ps/models/Qwen3-0.6B \
  --dataset-path openai/gsm8k:main --model-hub hf \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --train-devices 0 \
  --cut-cross-entropy --attn-backend native \
  --batch-size 2 --n-samples 4 --mini-bs 2 --score-micro-bs 2 \
  --max-prompt-tokens 512 --max-new-tokens 128 --disable-thinking \
  --max-steps 2 --save-path artifacts/e2e/ckpt_cce --save-interval 1 \
  --metrics-log-dir artifacts/tb/e2e_cce --seed 0 --no-shuffle
```

The run-config JSON records `"cut_cross_entropy": true`; CUDA + `tp_size=1`
selects `mst_chunk_size=256`, so `TrainingManager` takes the deferred
`defer_lm_head` branch and calls
`packed_next_token_logprobs_from_hidden(..., cut_cross_entropy=True)`. There is
no `cut-cross-entropy-needs-deferred-head` warning in either log.

Native trainer stats (`train/logp_abs_diff_mean` is emitted by the framework's
`TrainMetric.LOGP_ABS_DIFF_MEAN` in `areno/api/backend/cuda/losses.py`, not
computed by this report):

| step | reward_mean | grad_norm | clipped_grad_norm | train/logp_abs_diff_mean | step rollout s | step train s | step e2e s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5 | 7.2398 | 1.0 | 0.048248 | 13.470 | 1.835 | 15.560 |
| 1 | 0.0 | 0.0 | 0.0 | 0.043334 | 1.671 | 1.546 | 3.370 |

Step 0 has a non-zero gradient update (`grad_norm=7.2398`,
`clipped_grad_norm=1.0`); step 1 sampled only reward-0 completions so its GSPO
advantage (and gradient) is exactly zero. Both steps still exercise the
forward/backward graph through the CCE head.

Checkpoints were written at `artifacts/e2e/ckpt_cce/step_000001` and
`.../step_000002` (39 files each, 1.7 GB, HF-format weights + config + tokenizer).

Resume (new `--ckpt` = the saved step directory, 1 step):

```bash
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python -m areno.cli.train \
  --algo gspo --backend cuda --ckpt artifacts/e2e/ckpt_cce/step_000002 \
  --dataset-path openai/gsm8k:main --model-hub hf \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --train-devices 0 \
  --cut-cross-entropy --attn-backend native \
  --batch-size 2 --n-samples 4 --mini-bs 2 --score-micro-bs 2 \
  --max-prompt-tokens 512 --max-new-tokens 128 --disable-thinking \
  --max-steps 1 --save-path artifacts/e2e/ckpt_cce_resume --save-interval 1 \
  --metrics-log-dir artifacts/tb/e2e_cce_resume --seed 0 --no-shuffle
```

| step | reward_mean | grad_norm | clipped_grad_norm | train/logp_abs_diff_mean | step rollout s | step train s | step e2e s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.875 | 5.3779 | 1.0 | 0.049003 | 9.526 | 1.845 | 11.576 |

The resumed run loaded the saved weights, ran a non-zero update and saved
`artifacts/e2e/ckpt_cce_resume/step_000001`. Both metrics were read back from
the TensorBoard event files with `EventAccumulator`; the same values appear in
the `train_stats={...}` lines of the logs.

## Test status

- `pytest tests/test_cut_cross_entropy.py -v`: **36 passed** (raw log above).
- Full `pytest tests -q` is **not green on this host**: `328 failed, 2234 passed,
  1784 skipped` (raw log `artifacts/logs/task0_full_pytest.log`). The first
  failure is in `tests/test_attention_native.py` and is
  `RuntimeError: PyTorch is not linked with support for npu devices` from
  `run_backward`; the same error then cascades through unrelated routing/MoE/
  attention tests. `tests/test_accel_device_dispatch_cpu.py`,
  `tests/test_npu_backend_cpu.py` and `tests/test_decode_graph_cpu.py`
  register a PrivateUse1 device named `npu` (`torch.utils.rename_privateuse1_backend("npu")`
  in `tests/npu_stub.py`), which poisons CUDA backward for tests that run later
  in the same process. This is pre-existing test isolation / environment
  behaviour, not introduced by the CCE changes.
- Re-running the full suite with those three NPU-stub modules excluded gives
  `59 failed, 2336 passed, 1784 skipped` in 548 s
  (`artifacts/logs/task0b_full_pytest_no_npu_stub.log`). **Zero of the 59
  failures are in `tests/test_cut_cross_entropy.py`**; they are in unrelated
  native-kernel/model families (`test_mst_models`, `test_grouped_linear`,
  `test_fused_experts_native`, Gloo collective tests, QLoRA, etc.). The CCE
  file is green both standalone and without the NPU-stub modules.
