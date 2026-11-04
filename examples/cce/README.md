# Cut Cross-Entropy (CCE) baseline

`F.cross_entropy(hidden @ weight.T, targets)` materializes a `[tokens, vocab]`
logits block. For 8k supervised tokens and a 152k vocabulary that is ~2.4 GB in
BF16 for every training step (plus the same amount again for backward).

`areno.accel.cut_cross_entropy` computes the identical loss from the hidden
states, the LM-head weight, and the target ids, streaming the vocabulary in
chunks: each chunk is projected, immediately reduced to a per-row
`(max, sum-of-exp)` pair, and thrown away. Only the small per-chunk statistics
and one `[rows, hidden]` target dot-product survive the forward pass, and
backward recomputes the chunks instead of retaining them. Peak activation memory
is `O(rows * vocab_chunk)` instead of `O(rows * vocab)`.

## Enabling it

Off by default. When off, training runs the existing path unchanged
(`tests/test_cut_cross_entropy.py` asserts the legacy path never calls into the
new module).

CLI:

```bash
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --reward-fn-path examples/math/math_verify_reward.py --algo gspo --tp-size 1 \
  --cut-cross-entropy
```

Python SDK:

```python
from areno.api.trainer_config import SFTTrainerConfig

config = SFTTrainerConfig(
    algo="sft",
    ckpt="Qwen/Qwen3-0.6B",
    dataset_path="gsm8k:main",
    backend="cuda",
    cut_cross_entropy=True,   # -> RuntimeConfig.cut_cross_entropy
)
```

The equivalent runtime knob (for callers that build `CudaConfig` directly) is
`runtime={"cut_cross_entropy": True}`.

### Scope of the first landing

| item | state |
| --- | --- |
| deferred LM-head path (CUDA training, `tp_size=1`) | supported |
| materialized-logits path (`tp_size>1`, NPU, MLX) | not applicable; the run warns once and keeps the existing path |
| `label_smoothing != 0` | `NotImplementedError` (never mis-scored) |
| LM head with a bias | `NotImplementedError` |
| logit soft-capping (Gemma-style) | supported via `logit_softcap` |
| `ignore_index` | supported (`reduction="none"` yields `0` for ignored rows like PyTorch) |
| all targets ignored | returns `0` for `sum`/`mean` instead of `nan` |
| fused CUDA/Triton kernel | not yet; the pure-PyTorch reference runs everywhere |

## Reproduce the CPU baseline

Measured on this workstation (Windows, CPU-only Python, torch 2.7.1+cpu,
`rows=2048 hidden=512 vocab=65536 chunk=4096`). Each row is a separate process
so peak RSS growth is meaningful:

```bash
python -m pytest tests/test_cut_cross_entropy.py -q

PYTHONPATH=. python examples/cce/bench.py --rows 2048 --hidden 512 --vocab 65536 \
  --chunk 4096 --iters 3 --mode materialized
PYTHONPATH=. python examples/cce/bench.py --rows 2048 --hidden 512 --vocab 65536 \
  --chunk 4096 --iters 3 --mode cce
```

FP32 forward+backward:

| path | loss | fwd+bwd median ms | peak RSS growth MB |
| --- | --- | --- | --- |
| materialized logits | 96.211494 | 3332 | 1285 |
| cut cross-entropy | 96.211487 | 5586 | 107 |

BF16 forward+backward (1 iteration, so timing is noisy):

| path | loss | fwd+bwd median ms | peak RSS growth MB |
| --- | --- | --- | --- |
| materialized logits | 95.500000 | 1555 | 494 |
| cut cross-entropy | 95.153748 | 6040 | 71 |

Reading the numbers:

- The loss agrees to `7e-6` in FP32. In BF16 the two paths differ by ~0.4%
  because the reference rounds the `[rows, vocab]` logits to BF16 while CCE
  accumulates in FP32; `tests/test_cut_cross_entropy.py` pins that to
  `rtol=1e-2`.
- Peak memory drops ~12x (FP32) / ~7x (BF16).
- On CPU the pure-PyTorch reference is **slower** (~1.7x) than one big matmul.
  This is expected: it trades a single BLAS call for a chunked Python loop with
  backward recompute. The payoff is memory, and on GPU the chunked reduction is
  where a fused kernel belongs.

## GPU baseline — RTX 5880 Ada 48 GB, CUDA 12.8, torch 2.8.0+cu128

Measured on a single RTX 5880 Ada (sm_89, 48 GB) with `bf16`,
`torch.backends.cuda.matmul.allow_tf32=false`, seed 0. Command logs and the
full report live in `artifacts/logs/` and `REPORT.md` at the repository root.

`tests/test_cut_cross_entropy.py` (36 passed, including the CUDA peak-memory
and CUDA packed-head tests that skip on CPU):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_cut_cross_entropy.py -v
```

Numerical agreement at full vocabulary (`tokens=4096 hidden=1024 vocab=151936
chunk=4096`, reference `F.cross_entropy(hidden @ weight.T, targets)`):

| dtype | max abs | mean abs | max rel | mean rel | scalar loss rel diff |
| --- | --- | --- | --- | --- | --- |
| FP32 | 1.678e-04 | 1.360e-05 | 2.121e-06 | 1.065e-07 | 1.071e-07 |
| BF16 | 8.452e-01 | 3.742e-02 | 1.173e-02 | 2.950e-04 | 9.508e-07 |

99.98% of BF16 rows are inside the pinned `rtol=1e-2`; the aggregate loss
agrees to `9.5e-7` relative because CCE accumulates in FP32 while the reference
rounds the materialized logits to BF16.

Synthetic output head (`tokens=4096 hidden=1024`, bf16, forward+backward,
warmup 2 + 10 iterations). `max single alloc` is the largest single CUDA
allocation recorded by `torch.cuda.memory._record_memory_history`; a full
`tokens x vocab` tensor would appear there as one block:

| vocab | path | fwd+bwd median ms | peak alloc delta MB | max single alloc MB | full logits block MB |
| --- | --- | --- | --- | --- | --- |
| 32768 | materialized | 7.21 | 768.0 | 256.0 | 256.0 |
| 32768 | cut cross-entropy (chunk 4096) | 23.16 | 344.3 | 128.0 | 256.0 |
| 65536 | materialized | 15.73 | 1536.0 | 512.0 | 512.0 |
| 65536 | cut cross-entropy (chunk 4096) | 51.75 | 408.6 | 128.0 | 512.0 |

The CCE path never requests one full logits block (the largest single
allocation measured is 128 MB, below the 256/512 MB block), and its
peak-allocated delta is 2.2x / 3.8x lower
than the materialized path. 65536-token vocabulary chunk sweep (median ms):
1024 -> 85.83, 2048 -> 56.76, 4096 -> 51.75, 8192 -> 57.20; peak delta
268 / 273 / 409 / 680 MB.

Real Qwen3-0.6B full training step (`/home/ps/models/Qwen3-0.6B`, bf16, seed 0,
lr 1e-4, packed 8192 tokens / context 8192, MST chunk 256, 2 warmup + 10
measured steps, `attn_backend=native`; step includes SFT loss, backward,
FP32 main-gradient folding and AdamW update):

| path | step mean ms | tok/s | peak allocated MB | peak alloc delta MB | max reserved MB |
| --- | --- | --- | --- | --- | --- |
| materialized logits | 8320.5 | 985 | 22039.1 | 12201.9 | 22550.0 |
| deferred head (engine default, chunk 256) | 8175.1 | 1002 | 18111.9 | 8270.3 | 19772.0 |
| cut cross-entropy (chunk 4096) | 8360.6 | 980 | 18107.2 | 8269.8 | 19954.0 |

CCE cuts 3.9 GB of peak allocated memory (-17.8% total, -32.2% incremental head
memory) at +0.48% step time versus materialized logits. It matches the existing
deferred-head peak and is 2.3% slower than it; this reference implementation's
remaining opportunity is a fused kernel, which is the follow-up below.

Public `Trainer` end-to-end (GSPO + `--cut-cross-entropy`, GSM8K, one GPU,
2 steps + save + 1-step resume) is recorded in `REPORT.md`: native
`train/logp_abs_diff_mean` was 0.048248 / 0.043334 on the two steps and
0.049003 on the resumed step, with non-zero `grad_norm` on the updating steps.

## Follow-ups

- Fused (Triton) chunked projection + log-sum-exp for CUDA; the reference
  implementation keeps the public signature stable for it.
- Vocabulary-parallel support: the sharded head (`tp_size>1`) needs a
  cross-rank log-sum-exp plus a custom backward, so it currently raises
  `NotImplementedError` instead of silently scoring a local shard only.
