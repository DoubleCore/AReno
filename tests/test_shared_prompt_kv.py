"""External-behaviour gates for issue #592 shared prompt KV.

The feature core lives in PR #639.  These tests intentionally stay at the
public rollout surface (and the scheduler's observable block counters) instead
of duplicating the implementation tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areno.engine.api import _shared_prompt_kv_plan
from areno.engine.config import RuntimeConfig
from areno.engine.data import SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.inference import InferenceManager
from tests.helpers import PatchedContext

import areno.engine.inference as inference_mod


MODEL_PATH = os.environ.get("ARENO_SHARED_KV_MODEL", "/home/ps/models/Qwen3-0.6B")
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
MODEL_REQUIRED = pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason=f"model not found: {MODEL_PATH}")


def _shared_state(prompt, samples, *, block_size=256, blocks=32):
    return InferenceBatchState(
        prompts=[list(prompt) for _ in range(samples)],
        max_new_tokens=8,
        max_running_seqs=samples,
        max_cache_len=len(prompt) + 8,
        max_prefill_tokens=len(prompt) * samples,
        kv_block_size=block_size,
        num_cache_blocks=blocks,
        prompt_group_size=samples,
    )


def test_share_prompt_kv_is_opt_in():
    assert RuntimeConfig().share_prompt_kv is False
    assert RuntimeConfig(share_prompt_kv=True).share_prompt_kv is True


def test_shared_prompt_blocks_are_owned_by_all_samples():
    prompt = list(range(1024))
    state = _shared_state(prompt, 4)
    payload = state.build_prefill_payload()

    full_blocks = len(prompt) // state.kv_block_size
    assert payload["block_table"].shape[0] == 1
    assert state._last_active_ids == [0, 1, 2, 3]
    assert len(state._seq_to_blocks[0]) == full_blocks
    for row in range(4):
        assert state._seq_to_blocks[row] == state._seq_to_blocks[0]
        assert state._seq_to_recurrent_slot[row] == row
    prompt_blocks = state._seq_to_blocks[0]
    assert all(state._block_refcounts[block] == 4 for block in prompt_blocks)


def test_release_waits_for_last_owner_then_restores_free_pool():
    prompt = list(range(1024))
    state = _shared_state(prompt, 4)
    state.build_prefill_payload()
    prompt_blocks = list(state._seq_to_blocks[0])
    manager = InferenceManager(SimpleNamespace(device=torch.device("cpu")))

    manager._free_rollout_rows(state, torch.tensor([0, 2]))
    assert all(state._block_refcounts[block] == 2 for block in prompt_blocks)
    assert not set(prompt_blocks) & set(state._free_blocks)

    manager._free_rollout_rows(state, torch.tensor([1, 3]))
    assert not state._block_refcounts
    assert not state._seq_to_blocks
    assert len(state._free_blocks) == state.num_cache_blocks
    assert len(set(state._free_blocks)) == state.num_cache_blocks
    assert len(state._free_recurrent_slots) == state.max_running_seqs


def test_decode_response_blocks_are_private_per_sample():
    prompt = list(range(1024))
    state = _shared_state(prompt, 4)
    state.build_prefill_payload()
    prompt_block = state._seq_to_blocks[0][0]

    copies = state.ensure_decode_blocks([0, 1, 2, 3], [len(prompt)] * 4)
    assert copies == []
    response_blocks = [state._seq_to_blocks[row][-1] for row in range(4)]
    assert len(set(response_blocks)) == 4
    assert state._block_refcounts[prompt_block] == 4


def test_cancelled_samples_only_release_their_own_reference():
    class Manager(InferenceManager):
        def _infer_next_token_tensor(self, payload):
            return torch.arange(1, 5), torch.arange(4, dtype=torch.float32)

        def _infer_decode_next_token_tensor(
            self,
            next_tokens,
            position_ids,
            cache_seqlens,
            block_table,
            recurrent_slots,
            active_count,
            sampling_params,
            sample_generator,
            *,
            sample_step,
            eos_token_id,
        ):
            del (
                position_ids,
                cache_seqlens,
                block_table,
                recurrent_slots,
                active_count,
                sampling_params,
                sample_generator,
                eos_token_id,
            )
            return next_tokens + 1, torch.zeros_like(next_tokens, dtype=torch.float32) - float(sample_step)

    manager = Manager(SimpleNamespace(device=torch.device("cpu")))
    manager.device = torch.device("cpu")
    manager.config = SimpleNamespace(
        runtime=SimpleNamespace(speculative_draft_tokens=0), model=SimpleNamespace(vocab_size=8)
    )
    manager.model = SimpleNamespace(copy_kv_cache_blocks=lambda copies: None)
    state = _shared_state([1, 2, 3, 4, 5, 6], 4, block_size=4, blocks=16)
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            cancel_flags=torch.tensor([False, True, False, False]),
            cancel_indices=[0, 1, 2, 3],
        )

    assert state.generated[1] == [0]
    assert all(len(response) == state.max_new_tokens for index, response in enumerate(state.generated) if index != 1)
    assert state.finish_reason[1] in {"stop", "cancelled"}
    assert not state._block_refcounts
    assert not state._seq_to_blocks
    assert len(state._free_blocks) == state.num_cache_blocks
    assert len(set(state._free_blocks)) == state.num_cache_blocks
    assert len(state._free_recurrent_slots) == state.max_running_seqs


def _plan_config(*, model_type="qwen3", tp_size=1, device_type="cuda"):
    return SimpleNamespace(
        runtime=SimpleNamespace(share_prompt_kv=True, device_type=device_type, kv_block_size=4),
        model=SimpleNamespace(model_type=model_type),
        tp_size=tp_size,
        dp_size=1,
    )


def test_n_samples_one_never_enters_shared_mode():
    assert _shared_prompt_kv_plan(_plan_config(), [[1, 2]], None, 1, 1, 4, 4, 8) == (1, 4)


@pytest.mark.parametrize(
    ("config", "prompts", "capacity", "prefill_budget"),
    [
        (_plan_config(model_type="qwen3_5"), [[1, 2]] * 4, 4, 8),
        (_plan_config(tp_size=2), [[1, 2]] * 4, 4, 8),
        (_plan_config(device_type="cpu"), [[1, 2]] * 4, 4, 8),
        (_plan_config(), [[1, 2], [1, 3], [1, 2], [1, 2]], 4, 8),
        (_plan_config(), [[1, 2]] * 4, 2, 8),
        (_plan_config(), [[1, 2]] * 4, 4, 7),
    ],
)
def test_unsupported_rollouts_fall_back_to_independent_caches(config, prompts, capacity, prefill_budget, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "areno.engine.api.logger.warning", lambda message, reason: warnings.append(message % reason)
    )
    assert _shared_prompt_kv_plan(config, prompts, None, 4, capacity, 4, 8, prefill_budget) == (1, 8)
    assert len(warnings) == 1
    assert "using independent caches" in warnings[0]


@pytest.fixture(scope="module")
def engine_pair():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"model not found: {MODEL_PATH}")

    from areno.engine.api import ArenoEngine

    def build(shared):
        runtime = RuntimeConfig(attn_backend="native", compile_model=False, eager_decode=True)
        runtime.share_prompt_kv = shared
        engine = ArenoEngine.from_pretrained(
            MODEL_PATH, tp_size=1, dp_size=1, devices=[0], runtime_config=runtime, role="rollout", start=False
        )
        engine.config.model.dtype = torch.bfloat16
        engine.cluster.start()
        return engine

    engines = build(False), build(True)
    try:
        yield engines
    finally:
        for engine in engines:
            engine.close()


def _prompt(length):
    return [1 + (index % 10000) for index in range(length)]


def _rollout(engine, prompts, *, samples, max_new_tokens=8, cancel_flags=None):
    options = {
        "max_new_tokens": max_new_tokens,
        "max_running_prompts": samples,
        "max_prompt_len": len(prompts[0]),
        "sampling_params": SamplingParams(temperature=0.0, seed=1234),
        "prompt_group_size": samples,
    }
    if cancel_flags is not None:
        options["cancel_flags"] = cancel_flags
    return engine.generate_rollout(prompts, **options)


def _assert_same_rollout(expected, actual, *, atol=1e-5):
    assert expected.response_ids == actual.response_ids
    assert expected.finish_reason == actual.finish_reason
    for left, right, tokens in zip(expected.logprobs, actual.logprobs, expected.response_ids, strict=True):
        torch.testing.assert_close(left[: len(tokens)], right[: len(tokens)], atol=atol, rtol=atol)


@CUDA_REQUIRED
@MODEL_REQUIRED
@pytest.mark.parametrize("prompt_len", [256, 512])
@pytest.mark.parametrize("samples", [1, 4, 8])
def test_shared_matches_independent_cuda(engine_pair, prompt_len, samples):
    baseline, shared = engine_pair
    prompt = _prompt(prompt_len)
    prompts = [prompt for _ in range(samples)]

    expected = _rollout(baseline, prompts, samples=samples)
    actual = _rollout(shared, prompts, samples=samples)
    _assert_same_rollout(expected, actual)


@CUDA_REQUIRED
@MODEL_REQUIRED
def test_cancelled_shared_rollout_releases_pool_for_next_rollout(engine_pair):
    baseline, shared = engine_pair
    prompt = _prompt(256)
    prompts = [prompt for _ in range(4)]

    flags = torch.tensor([False, True, False, False], dtype=torch.bool)
    cancelled = _rollout(shared, prompts, samples=4, cancel_flags=flags)
    assert cancelled.finish_reason[1] == "cancelled"

    expected = _rollout(baseline, prompts, samples=4)
    actual = _rollout(shared, prompts, samples=4)
    _assert_same_rollout(expected, actual)
