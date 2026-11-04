"""Cross-configuration and failure-path regressions for saved-activation compression."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import areno.engine.runtime.activation_compression as activation_compression
from areno.adapters.config import LoraConfig
from areno.engine.config import EngineConfig, ModelConfig, RuntimeConfig
from areno.engine.runtime.activation_compression import (
    GROUP_SIZE,
    MIN_NUMEL,
    SUPPORTED_BITS,
    UNCOMPRESSED,
    ActivationCompressor,
    PackedActivation,
    dequantize,
    outlier_count,
    quantize,
    relative_error,
)


def _record(section: str, **values: object) -> None:
    print("ACTC_METRIC " + json.dumps({"section": section, **values}, sort_keys=True))


def _lowbias32(value: int) -> int:
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def _uniform(index: int, seed: int) -> np.float32:
    hashed = _lowbias32(_lowbias32(index & 0xFFFFFFFF) ^ seed)
    return np.float32((hashed >> 8) / float(1 << 24))


def _reference_quantize(x: torch.Tensor, bits: int, *, stochastic: bool, seed: int):
    values = x.detach().float().numpy().reshape(-1)
    channels = x.shape[-1]
    rows = values.reshape(-1, channels)
    peak = np.maximum(rows.max(axis=0), -rows.min(axis=0))
    outliers = np.argsort(-peak, kind="stable")[: outlier_count(channels)]
    keep = np.zeros(channels, dtype=bool)
    keep[outliers] = True
    levels = int((1 << bits) - 1)
    groups = -(-values.size // GROUP_SIZE)
    codes = np.zeros(groups * GROUP_SIZE, dtype=np.int64)
    minima = np.zeros(groups, dtype=np.float32)
    steps = np.zeros(groups, dtype=np.float32)
    for group in range(groups):
        members = [
            i
            for i in range(group * GROUP_SIZE, min((group + 1) * GROUP_SIZE, values.size))
            if not keep[i % channels]
        ]
        low = np.float32(min(values[i] for i in members)) if members else np.float32(0.0)
        high = np.float32(max(values[i] for i in members)) if members else np.float32(0.0)
        step = np.float32(np.float32(high - low) / np.float32(levels))
        step = step if step > 0 else np.float32(1.0)
        minima[group], steps[group] = low, step
        for i in members:
            q = np.float32(np.float32(values[i] - low) / step)
            u = _uniform(i, seed) if stochastic else np.float32(0.5)
            codes[i] = min(max(int(np.floor(np.float32(q + u))), 0), levels)
    return codes, minima, steps, outliers


def _unpack_codes(packed: PackedActivation) -> np.ndarray:
    per_byte = 8 // packed.bits
    data = packed.codes.numpy().astype(np.int64)
    return np.stack([(data >> (packed.bits * j)) & ((1 << packed.bits) - 1) for j in range(per_byte)], 1).reshape(-1)


def _matrix_input(shape, dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator) * 2 + 0.3
    for rank, channel in enumerate((1, 29, 57, 85)):
        if channel < shape[-1]:
            x[..., channel] = (40.0 - 5.0 * rank) * torch.sign(x[..., channel])
    return x.to(dtype)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantization_matrix_matches_independent_fp32_reference_cpu(bits, dtype):
    x = _matrix_input((1024, 96), dtype, seed=100 + bits)
    packed = quantize(x, bits, stochastic=True, seed=2026, use_kernel=False)
    codes, minima, steps, outliers = _reference_quantize(x, bits, stochastic=True, seed=2026)

    np.testing.assert_array_equal(_unpack_codes(packed), codes)
    np.testing.assert_array_equal(packed.minimum.numpy(), minima)
    np.testing.assert_array_equal(packed.scale.numpy(), steps)
    np.testing.assert_array_equal(np.sort(packed.outlier_index.numpy()), np.sort(outliers))

    restored = dequantize(packed, use_kernel=False)
    group = np.arange(codes.size) // GROUP_SIZE
    expected = (codes.astype(np.float32) * steps[group]).astype(np.float32) + minima[group]
    expected = expected[: x.numel()].reshape(x.shape).copy()
    expected_flat = x.float().reshape(-1, x.shape[-1]).numpy().copy()
    expected[:, outliers] = expected_flat[:, outliers]
    expected = torch.from_numpy(expected).to(dtype)
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)

    original = x.float()
    reconstruction_error = float((restored.float() - original).norm() / original.norm())
    max_abs_error = float((restored.float() - original).abs().max())
    stored = packed.packed_bytes + packed.scale_bytes + packed.outlier_bytes
    stored_fraction = stored / (x.numel() * x.element_size())
    _record(
        "quantization_matrix",
        bits=bits,
        dtype=_dtype_name(dtype),
        stochastic=True,
        exact_match=True,
        reconstruction_relative_l2=round(reconstruction_error, 8),
        reconstruction_max_abs=round(max_abs_error, 8),
        stored_fraction=round(stored_fraction, 6),
    )
    assert stored_fraction < 1.0


class _GradBlock(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(F.silu(self.up(x)))


def _block_input(rows: int = 512, width: int = 128, *, dtype=torch.float32, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, width, generator=generator).to(dtype)


def _block_grads(block, x, compressor=None, plan="current"):
    block.zero_grad(set_to_none=True)
    if compressor is None:
        out = block(x)
    else:
        with compressor.capture(plan):
            out = block(x)
    out.float().square().mean().backward()
    return [p.grad.detach().clone() if p.grad is not None else None for p in block.parameters()]


def test_gradient_matrix_by_bits_and_dtype_cpu():
    errors = {}
    for dtype in (torch.bfloat16, torch.float16):
        torch.manual_seed(0)
        block = _GradBlock().to(dtype)
        x = _block_input(dtype=dtype, seed=1)
        compressor = ActivationCompressor(block)
        exact = _block_grads(block, x, compressor, plan=None)
        slots = len(compressor.last_capture.signatures)
        compressor.signatures = list(compressor.last_capture.signatures)
        for bits in (2, 4, 8):
            got = _block_grads(block, x, compressor, plan=[bits] * slots)
            error = relative_error(got, exact)
            stats = compressor.last_capture.stats
            errors[(dtype, bits)] = error
            _record(
                "gradient_matrix",
                bits=bits,
                dtype=_dtype_name(dtype),
                slots=slots,
                gradient_relative_l2=round(error, 8),
                stored_fraction=round(stats.stored_bytes / max(stats.saved_bytes, 1), 6),
            )
            assert stats.compressed_slots == slots and stats.oom_fallback_slots == 0
    for dtype in (torch.bfloat16, torch.float16):
        assert errors[(dtype, 8)] <= errors[(dtype, 4)] <= errors[(dtype, 2)]


def test_adaptive_gradient_error_stays_within_one_percent_budget_cpu():
    torch.manual_seed(0)
    block = _GradBlock()
    x = _block_input(seed=11)
    held_x = _block_input(seed=12)
    compressor = ActivationCompressor(block, error_budget=0.01)

    def runner(inputs):
        def run(plan):
            return _block_grads(block, inputs, compressor, plan=plan)

        return run

    metrics = compressor.calibrate(runner(x), runner(held_x))
    calibration = compressor.calibration
    _record(
        "budget_1pct",
        calibration_error=round(calibration["calibration_error"], 8),
        held_out_error=round(calibration["held_out_error"], 8),
        stored_fraction=round(calibration["stored_fraction"], 6),
        uncompressed_fallback=calibration["uncompressed_fallback"],
        slots=calibration["slots"],
    )
    assert metrics["actc_calibration_error"] <= 0.01
    assert metrics["actc_held_out_error"] <= 0.01
    assert metrics["actc_plan_stored_fraction"] <= 1.0


def test_over_budget_check_recalibrates_to_uncompressed_fallback_cpu():
    torch.manual_seed(0)
    block = _GradBlock()
    x = _block_input(seed=21)
    held_x = _block_input(seed=22)
    compressor = ActivationCompressor(block, error_budget=0.02)

    def runner(inputs):
        def run(plan):
            return _block_grads(block, inputs, compressor, plan=plan)

        return run

    compressor.calibrate(runner(x), runner(held_x))
    assert any(bits != UNCOMPRESSED for bits in compressor.plan)
    compressor.error_budget = 1.0e-7
    metrics = compressor.check(runner(x), runner(held_x))
    _record(
        "fallback",
        check_error=round(metrics["actc_check_error"], 8),
        recalibrated=metrics.get("actc_recalibrated", 0.0),
        uncompressed_fallback=metrics["actc_uncompressed_fallback"],
        plan=compressor.plan,
    )
    assert metrics["actc_recalibrated"] == 1.0
    assert metrics["actc_uncompressed_fallback"] == 1.0
    assert compressor.plan == [UNCOMPRESSED] * len(compressor.plan)


def test_calibration_accepts_a_single_batch_without_held_out_cpu():
    torch.manual_seed(0)
    block = _GradBlock()
    x = _block_input(seed=31)
    compressor = ActivationCompressor(block, error_budget=0.02)

    def run(plan):
        return _block_grads(block, x, compressor, plan=plan)

    metrics = compressor.calibrate(run, None)
    _record(
        "single_batch",
        calibration_error=round(compressor.calibration["calibration_error"], 8),
        slots=compressor.calibration["slots"],
    )
    assert compressor.calibration["slots"] > 0
    assert metrics["actc_calibration_error"] <= 0.02
    assert "actc_held_out_error" not in metrics
    assert math.isnan(compressor.calibration["held_out_error"])


def test_calibration_with_no_eligible_slots_is_empty_not_an_error_cpu():
    torch.manual_seed(0)
    block = _GradBlock(width=64)
    x = _block_input(rows=128, width=64, seed=41)
    compressor = ActivationCompressor(block, error_budget=0.02)

    def run(plan):
        return _block_grads(block, x, compressor, plan=plan)

    metrics = compressor.calibrate(run, None)
    assert compressor.last_capture.stats.slots == 0
    assert compressor.plan == []
    assert metrics["actc_uncompressed_fallback"] == 1.0
    assert metrics["actc_plan_stored_fraction"] == 0.0
    _record("empty_calibration", slots=0, uncompressed_fallback=1.0, stored_fraction=0.0)


@pytest.mark.parametrize("bits", [-8, -1, 0, 1, 3, 5, 6, 7, 9, 16, 32])
def test_invalid_ratios_raise_value_error_cpu(bits):
    x = _matrix_input((64, 96), torch.float32, seed=51)
    with pytest.raises(ValueError, match="bits"):
        quantize(x, bits, use_kernel=False)
    with pytest.raises(ValueError, match="bits"):
        ActivationCompressor(nn.Identity(), bits=(bits,))
    with pytest.raises(ValueError, match="bits"):
        RuntimeConfig(activation_compression_bits=(bits,))


class _UnsupportedSaves(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.embed = nn.Embedding(32, width)
        self.proj = nn.Linear(width, width)
        self.last_ids = None

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        self.last_ids = ids
        return self.proj(x) + self.embed(ids)


def test_unsupported_saved_dtype_falls_back_explicitly_cpu():
    torch.manual_seed(0)
    model = _UnsupportedSaves()
    x = _block_input(seed=61)
    ids = torch.randint(0, 32, (x.shape[0],))
    compressor = ActivationCompressor(model)
    with compressor.capture(None):
        out = model(x, ids)
    capture = compressor.last_capture
    assert ids.untyped_storage().data_ptr() in capture.held
    assert capture.signatures and all(signature[0] == "torch.float32" for signature in capture.signatures)
    assert capture.stats.compressed_slots == 0
    out.sum().backward()
    with pytest.raises(ValueError, match="floating"):
        quantize(torch.ones(512, 128, dtype=torch.int64), 8, use_kernel=False)


class _UnsupportedLayout(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sin()


def test_unsupported_model_layouts_fail_explicitly_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = RuntimeConfig(activation_compression="adaptive")
    dense = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32)
    moe = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32, num_experts=4)
    with pytest.raises(ValueError, match="dense"):
        EngineConfig(model=moe, devices=[0], runtime=runtime)
    with pytest.raises(ValueError, match="full-parameter"):
        EngineConfig(model=dense, devices=[0], runtime=runtime, lora=LoraConfig(rank=4, alpha=8))
    with pytest.raises(ValueError, match="one CUDA training GPU"):
        EngineConfig(model=dense, tp_size=2, devices=[0, 1], runtime=runtime)


def test_oom_while_packing_falls_back_to_uncompressed_cpu(monkeypatch):
    torch.manual_seed(0)
    block = _GradBlock()
    x = _block_input(seed=71)
    compressor = ActivationCompressor(block)
    exact = _block_grads(block, x, compressor, plan=None)
    compressor.signatures = list(compressor.last_capture.signatures)
    slots = len(compressor.signatures)
    compressor.plan = [8] * slots

    def out_of_memory(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated packing OOM")

    monkeypatch.setattr(activation_compression, "quantize", out_of_memory)
    got = _block_grads(block, x, compressor)
    stats = compressor.last_capture.stats
    _record(
        "oom_fallback",
        slots=stats.slots,
        compressed_slots=stats.compressed_slots,
        oom_fallback_slots=stats.oom_fallback_slots,
        stored_fraction=round(stats.stored_bytes / max(stats.saved_bytes, 1), 6),
    )
    assert stats.compressed_slots == 0
    assert stats.oom_fallback_slots == slots
    assert stats.stored_bytes == stats.saved_bytes
    assert relative_error(got, exact) == 0.0


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="activation compression kernels require CUDA")


@CUDA
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_kernel_matrix_matches_pytorch_path_cuda(bits, dtype):
    x = _matrix_input((1024, 96), dtype, seed=200 + bits).cuda()
    kernel = quantize(x, bits, stochastic=True, seed=2026, use_kernel=True)
    reference = quantize(x, bits, stochastic=True, seed=2026, use_kernel=False)
    for name in ("codes", "minimum", "scale", "outlier_index", "outliers"):
        assert torch.equal(getattr(kernel, name), getattr(reference, name))
    kernel_out = dequantize(kernel, use_kernel=True)
    reference_out = dequantize(reference, use_kernel=False)
    assert torch.equal(kernel_out, reference_out)
    _record(
        "cuda_matrix",
        bits=bits,
        dtype=_dtype_name(dtype),
        bitwise_equal=True,
        reconstruction_relative_l2=round(
            float((kernel_out.float() - x.float()).norm() / x.float().norm()), 8
        ),
    )
