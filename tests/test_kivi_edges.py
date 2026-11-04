"""Boundary regression tests for the opt-in KIVI 2/4-bit KV cache.

These cases complement ``tests/test_kivi_kv_cache_cpu.py`` and
``tests/test_kivi_kv_cache_cuda.py``: invalid group sizes, non-divisible
shapes, missing CUDA, a NumPy dequantization reference, object-release
accounting, and batch=1 residual-window boundaries at 1024/4096/8192 tokens.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from areno.engine.config import ModelConfig, RuntimeConfig
from areno.engine.layers.attention_backend.kivi import (
    KiviInferBackend,
    KiviKVCache,
    KiviSpec,
    dequantize_blocks,
    validate_kivi_geometry,
)
from areno.engine.runtime.metadata import InferMeta

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="KIVI kernels require CUDA")

BLOCK_SIZE = 256
HEAD_DIM = 128
FP16_MAX = 65504.0


def _manual_cache(
    spec: KiviSpec,
    *,
    num_blocks: int,
    num_slots: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> KiviKVCache:
    packed = head_dim * spec.bits // 8
    group = spec.group_size

    def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(*shape, dtype=dtype, device=device)

    return KiviKVCache(
        spec=spec,
        block_size=block_size,
        k_codes=empty(num_blocks, num_kv_heads, block_size, packed, dtype=torch.uint8),
        k_scale=empty(num_blocks, num_kv_heads, block_size // group, head_dim, dtype=torch.float16),
        k_zero=empty(num_blocks, num_kv_heads, block_size // group, head_dim, dtype=torch.float16),
        v_codes=empty(num_blocks, num_kv_heads, block_size, packed, dtype=torch.uint8),
        v_scale=empty(num_blocks, num_kv_heads, block_size, head_dim // group, dtype=torch.float16),
        v_zero=empty(num_blocks, num_kv_heads, block_size, head_dim // group, dtype=torch.float16),
        k_ring=empty(num_slots, num_kv_heads, spec.ring_size, head_dim, dtype=dtype),
        v_ring=empty(num_slots, num_kv_heads, spec.ring_size, head_dim, dtype=dtype),
    )


@pytest.mark.parametrize("group", [0, -1, -8, 3, 24, 48, 100])
def test_invalid_group_size_rejected_by_geometry_cpu(group):
    with pytest.raises(ValueError, match="power of two >= 32"):
        validate_kivi_geometry(KiviSpec(bits=2, group_size=group, residual_length=128), block_size=256, head_dim=128)


@pytest.mark.parametrize("group", [0, -1, 24, 48])
def test_invalid_group_size_rejected_by_runtime_config_cpu(group):
    with pytest.raises(ValueError, match="kivi_group_size"):
        RuntimeConfig(kivi_group_size=group)


@pytest.mark.parametrize(
    "head_dim,block_size,group,match",
    [
        (96, 256, 32, "head dim"),
        (80, 256, 32, "head dim"),
        (64, 256, 128, "head dim"),
        (128, 64, 128, "block size"),
    ],
)
def test_non_divisible_shapes_rejected_cpu(head_dim, block_size, group, match):
    with pytest.raises(ValueError, match=match):
        validate_kivi_geometry(
            KiviSpec(bits=2, group_size=group, residual_length=128), block_size=block_size, head_dim=head_dim
        )


@pytest.mark.parametrize(
    "model_kwargs,reason",
    [({"head_dim": 96}, "head_dim 96"), ({"head_dim": 80}, "head_dim 80")],
)
def test_runtime_config_falls_back_for_non_divisible_head_dim_cpu(model_kwargs, reason):
    runtime = RuntimeConfig(kv_cache_dtype="kivi4")
    model = ModelConfig(model_type="qwen3", **model_kwargs)
    with pytest.warns(RuntimeWarning, match=reason):
        runtime.resolve_kv_cache_dtype(model=model)
    assert runtime.kv_cache_dtype == "auto"


def test_cpu_runtime_falls_back_to_model_dtype_cache():
    runtime = RuntimeConfig(kv_cache_dtype="kivi2")
    runtime.device_type = "cpu"
    with pytest.warns(RuntimeWarning, match="require CUDA"):
        runtime.resolve_kv_cache_dtype(model=ModelConfig(model_type="qwen3", head_dim=128))
    assert runtime.kv_cache_dtype == "auto"


def test_allocate_without_cuda_raises_clear_error_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = KiviSpec(bits=2, group_size=32, residual_length=128)
    with pytest.raises(RuntimeError, match="KIVI KV cache requires a CUDA device"):
        KiviKVCache.allocate(
            spec,
            num_blocks=1,
            block_size=BLOCK_SIZE,
            num_slots=1,
            num_kv_heads=2,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


def test_gqa_query_heads_must_divide_kv_heads_cpu():
    spec = KiviSpec(bits=2, group_size=32, residual_length=128)
    cache = _manual_cache(spec, num_blocks=1, num_slots=1, num_kv_heads=2)
    backend = KiviInferBackend("native")
    meta = InferMeta(
        mode="decode",
        cache_seqlens=torch.tensor([8], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        recurrent_slots=torch.tensor([0]),
    )
    q = torch.zeros(1, 1, 5, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 2, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    with pytest.raises(ValueError, match="query heads divisible by KV heads"):
        backend(q, k, v, cache, meta)


def test_qkv_head_dim_must_match_cache_cpu():
    spec = KiviSpec(bits=2, group_size=32, residual_length=128)
    cache = _manual_cache(spec, num_blocks=1, num_slots=1, num_kv_heads=2)
    backend = KiviInferBackend("native")
    meta = InferMeta(
        mode="decode",
        cache_seqlens=torch.tensor([8], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        recurrent_slots=torch.tensor([0]),
    )
    q = torch.zeros(1, 1, 4, 64, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 2, 64, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    with pytest.raises(ValueError, match="head_dim"):
        backend(q, k, v, cache, meta)


def _numpy_unpack(packed: np.ndarray, bits: int) -> np.ndarray:
    pack = 8 // bits
    data = packed.astype(np.int32)
    out = np.empty((*packed.shape[:-1], packed.shape[-1] * pack), dtype=np.int32)
    for field in range(pack):
        out[..., field::pack] = (data >> (field * bits)) & ((1 << bits) - 1)
    return out


def _numpy_dequant(cache: KiviKVCache, block_row: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bits, group, block_size = cache.spec.bits, cache.spec.group_size, cache.block_size
    blocks = block_row[positions // block_size]
    offsets = positions % block_size
    k_codes = cache.k_codes.cpu().numpy()[blocks, :, offsets]
    k_scale = cache.k_scale.cpu().numpy()[blocks, :, offsets // group].astype(np.float32)
    k_zero = cache.k_zero.cpu().numpy()[blocks, :, offsets // group].astype(np.float32)
    v_codes = cache.v_codes.cpu().numpy()[blocks, :, offsets]
    v_scale = cache.v_scale.cpu().numpy()[blocks, :, offsets].astype(np.float32)
    v_zero = cache.v_zero.cpu().numpy()[blocks, :, offsets].astype(np.float32)
    k = _numpy_unpack(k_codes, bits).astype(np.float32) * k_scale + k_zero
    v = _numpy_unpack(v_codes, bits).astype(np.float32)
    v = v * np.repeat(v_scale, group, axis=-1) + np.repeat(v_zero, group, axis=-1)
    return k, v


@pytest.mark.parametrize("bits", [2, 4])
def test_dequantize_blocks_matches_independent_numpy_cpu(bits):
    generator = torch.Generator().manual_seed(590 + bits)
    spec = KiviSpec(bits=bits, group_size=32, residual_length=128)
    cache = _manual_cache(spec, num_blocks=3, num_slots=1, num_kv_heads=3)
    cache.k_codes.copy_(torch.randint(0, 256, cache.k_codes.shape, generator=generator, dtype=torch.uint8))
    cache.v_codes.copy_(torch.randint(0, 256, cache.v_codes.shape, generator=generator, dtype=torch.uint8))
    cache.k_scale.uniform_(-0.5, 0.5, generator=generator)
    cache.k_zero.uniform_(-8.0, 8.0, generator=generator)
    cache.v_scale.uniform_(-0.5, 0.5, generator=generator)
    cache.v_zero.uniform_(-8.0, 8.0, generator=generator)
    block_row = np.array([2, 0, 1])
    length = 700
    block_ids = torch.tensor(block_row, dtype=torch.long)
    got_k, got_v = dequantize_blocks(cache, block_ids, length)
    positions = np.arange(length)
    want_k, want_v = _numpy_dequant(cache, block_row, positions)
    torch.testing.assert_close(got_k.float(), torch.from_numpy(want_k).to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(got_v.float(), torch.from_numpy(want_v).to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)


def test_clear_releases_all_cache_objects_with_live_counter():
    if not HAS_CUDA:
        pytest.skip("KIVI storage needs CUDA")

    import torch.nn as nn

    from areno.engine.layers.attention import CausalSelfAttention
    from areno.engine.layers.attention_backend.kivi import attach_kivi_kv_caches

    spec = KiviSpec(bits=2, group_size=32, residual_length=128)

    def live_caches(ids: set[int]) -> int:
        return sum(isinstance(obj, KiviKVCache) and id(obj) in ids for obj in gc.get_objects())

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            config = ModelConfig(hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128)
            self.layers = nn.ModuleList([nn.Module() for _ in range(2)])
            for layer in self.layers:
                layer.self_attn = CausalSelfAttention(config, 0)

    model = Model().cuda()
    gc.collect()
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    caches = attach_kivi_kv_caches(
        model,
        spec,
        num_blocks=8,
        block_size=BLOCK_SIZE,
        num_slots=2,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    ids = {id(cache) for cache in caches}
    assert live_caches(ids) == len(model.layers)
    assert sum(cache.nbytes() for cache in caches) > 0
    del caches
    for layer in model.layers:
        layer.self_attn.clear_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    assert live_caches(ids) == 0
    assert torch.cuda.memory_allocated() == baseline_bytes


def _cache_alloc(spec: KiviSpec, *, num_blocks: int, num_slots: int, num_kv_heads: int) -> KiviKVCache:
    return KiviKVCache.allocate(
        spec,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_slots=num_slots,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )


class _Sequence:
    def __init__(self, block_row: torch.Tensor, slot: int, num_kv_heads: int):
        self.k = torch.empty(0, num_kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        self.v = torch.empty_like(self.k)
        self.block_row = block_row
        self.slot = slot

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = torch.cat([self.k, k])
        self.v = torch.cat([self.v, v])


def _random_kv(tokens: int, heads: int, generator: torch.Generator) -> torch.Tensor:
    x = torch.randn(tokens, heads, HEAD_DIM, generator=generator)
    x[..., 3] *= 12.0
    x[..., 77] += 6.0
    return x.to(device="cuda", dtype=torch.bfloat16)


def _prefill(backend, cache, seq, q, k, v):
    meta = InferMeta(
        mode="prefill",
        cu_seqlens=torch.tensor([0, len(k)], dtype=torch.int32, device="cuda"),
        max_seqlen=len(k),
        block_table=seq.block_row[None, :],
        recurrent_slots=torch.tensor([seq.slot], dtype=torch.long, device="cuda"),
    )
    out = backend(q[None], k[None], v[None], cache, meta).squeeze(0)
    seq.append(k, v)
    return out


def _decode(backend, cache, seq, q, k, v):
    meta = InferMeta(
        mode="decode",
        cache_seqlens=torch.tensor([len(seq.k)], dtype=torch.int32, device="cuda"),
        block_table=seq.block_row[None, :],
        recurrent_slots=torch.tensor([seq.slot], dtype=torch.long, device="cuda"),
    )
    out = backend(q[:, None], k[:, None], v[:, None], cache, meta).squeeze(1).squeeze(0)
    seq.append(k, v)
    return out


def _reference_decode(cache, seq, q, scale):
    spec = cache.spec
    n = len(seq.k)
    past = n - 1 - spec.residual_length
    quantized = past // spec.group_size * spec.group_size if past >= 0 else 0
    block_row = seq.block_row.cpu().numpy()
    if quantized:
        k_prefix, v_prefix = _numpy_dequant(cache, block_row, np.arange(quantized))
    else:
        k_prefix = np.empty((0, seq.k.shape[1], HEAD_DIM), dtype=np.float32)
        v_prefix = np.empty_like(k_prefix)
    keys = np.concatenate([k_prefix, seq.k[quantized:].float().cpu().numpy()])
    values = np.concatenate([v_prefix, seq.v[quantized:].float().cpu().numpy()])
    repeat = q.shape[0] // keys.shape[1]
    keys = np.repeat(keys, repeat, axis=1)
    values = np.repeat(values, repeat, axis=1)
    scores = np.einsum("hd,khd->hk", q.float().cpu().numpy(), keys) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return torch.from_numpy(np.einsum("hk,khd->hd", probs, values))


@cuda_only
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("prompt_tokens", [1024, 4096, 8192])
def test_batch_one_residual_boundary_at_long_prompts(bits, prompt_tokens):
    generator = torch.Generator().manual_seed(prompt_tokens + bits)
    group, residual, q_heads, kv_heads = 32, 128, 4, 2
    spec = KiviSpec(bits=bits, group_size=group, residual_length=residual)
    steps = 2 * group + 1
    blocks_per_row = -(-(prompt_tokens + steps + 1) // BLOCK_SIZE)
    cache = _cache_alloc(spec, num_blocks=blocks_per_row, num_slots=1, num_kv_heads=kv_heads)
    block_row = torch.arange(blocks_per_row, device="cuda", dtype=torch.int32)
    seq = _Sequence(block_row, slot=0, num_kv_heads=kv_heads)
    backend = KiviInferBackend("native")
    q = _random_kv(prompt_tokens, q_heads, generator)
    k = _random_kv(prompt_tokens, kv_heads, generator)
    v = _random_kv(prompt_tokens, kv_heads, generator)
    _prefill(backend, cache, seq, q, k, v)
    quantized = spec.quantized_length(prompt_tokens)
    assert prompt_tokens - quantized == spec.ring_size
    checkpoints = {0, 1, group - 1, group, 2 * group - 1, 2 * group}
    scale = HEAD_DIM**-0.5
    for step in range(steps):
        q_step = _random_kv(1, q_heads, generator)
        out = _decode(
            backend, cache, seq, q_step, _random_kv(1, kv_heads, generator), _random_kv(1, kv_heads, generator)
        )
        if step in checkpoints:
            want = _reference_decode(cache, seq, q_step[0], scale).to(out.device)
            torch.testing.assert_close(out.float(), want, atol=1e-2, rtol=1e-2)
    final_quantized = spec.quantized_length(len(seq.k))
    ring_positions = torch.arange(final_quantized, len(seq.k), device="cuda") % spec.ring_size
    assert torch.equal(cache.k_ring[seq.slot][:, ring_positions].transpose(0, 1), seq.k[final_quantized:])
    assert torch.equal(cache.v_ring[seq.slot][:, ring_positions].transpose(0, 1), seq.v[final_quantized:])
