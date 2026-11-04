"""GPU baseline for the opt-in cut cross-entropy LM-head path.

Three modes, each runnable on a CUDA device:

``accuracy``
    Numerical agreement between ``F.cross_entropy(hidden @ weight.T, targets)``
    and :func:`areno.accel.cut_cross_entropy` at realistic scale, in FP32 and
    BF16, reporting the actual maximum absolute/relative errors (not just the
    pinned test tolerances).

``synthetic``
    Output-head peak memory and forward+backward time for one shape
    (``--tokens --hidden --vocab``), one path per process:
    ``--path materialized`` runs ``F.cross_entropy(hidden @ weight.T)``;
    ``--path cce`` runs the chunked path. Reports the CUDA peak-allocated
    delta over the input tensors and asserts the CCE peak stays below one
    full ``tokens x vocab`` logits block, i.e. the path never allocates the
    materialized tensor.

``model``
    Full-step peak and throughput with a real checkpoint (default
    Qwen3-0.6B) loaded through ``areno.models``: packed forward with the
    deferred LM head (MST chunk 256, activation checkpointing), the selected
    head path, the engine's ``sft_loss_fn``, backward, Megatron-style FP32
    main-gradient folding, and an ``AdamWFP32Master`` optimizer step — the
    same step structure as ``areno.engine.training.TrainingManager`` at
    ``tp_size=1``. Paths: ``materialized`` (full ``[tokens, vocab]``
    projection, the non-deferred head), ``deferred`` (token-chunked
    projection, the engine default), ``cce`` (vocabulary-chunked log-sum-exp).

Every mode prints a human-readable table plus a final ``SUMMARY_JSON=`` line
so results can be archived straight from the log.

Examples:
    PYTHONPATH=. python examples/cce/bench_gpu.py --mode accuracy
    PYTHONPATH=. python examples/cce/bench_gpu.py --mode synthetic --vocab 65536 --path cce
    PYTHONPATH=. python examples/cce/bench_gpu.py --mode model --path cce --ckpt /home/ps/models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from areno.accel.cut_cross_entropy import cut_cross_entropy


def _cuda_env() -> dict:
    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }


def _json_line(payload: dict) -> None:
    print("SUMMARY_JSON=" + json.dumps(payload, sort_keys=True))


def _sync() -> None:
    torch.cuda.synchronize()


def _zero_grads(*tensors) -> None:
    for tensor in tensors:
        tensor.grad = None


# ---------------------------------------------------------------------------
# accuracy mode
# ---------------------------------------------------------------------------


def _row_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    diff = (actual - expected).abs()
    denom = expected.abs().clamp(min=1e-6)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "mean_rel": float((diff / denom).mean()),
    }


def run_accuracy(args) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    hidden = torch.randn(args.tokens, args.hidden, device=device)
    weight = torch.randn(args.vocab, args.hidden, device=device)
    targets = torch.randint(args.vocab, (args.tokens,), device=device)

    logits = hidden @ weight.T
    expected = F.cross_entropy(logits, targets, reduction="none")
    actual = cut_cross_entropy(hidden, weight, targets, reduction="none", vocab_chunk_size=args.chunk)
    fp32_rows = _row_errors(actual, expected)
    fp32_scalar = {
        "expected_loss": float(F.cross_entropy(logits, targets)),
        "actual_loss": float(cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=args.chunk)),
    }
    fp32_scalar["abs_diff"] = abs(fp32_scalar["actual_loss"] - fp32_scalar["expected_loss"])
    fp32_scalar["rel_diff"] = fp32_scalar["abs_diff"] / abs(fp32_scalar["expected_loss"])

    torch.manual_seed(args.seed + 1)
    hidden_bf16 = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(args.vocab, args.hidden, device=device, dtype=torch.bfloat16)
    targets_bf16 = torch.randint(args.vocab, (args.tokens,), device=device)
    expected_bf16 = F.cross_entropy((hidden_bf16 @ weight_bf16.T).float(), targets_bf16, reduction="none")
    actual_bf16 = cut_cross_entropy(hidden_bf16, weight_bf16, targets_bf16, reduction="none", vocab_chunk_size=args.chunk)
    bf16_rows = _row_errors(actual_bf16, expected_bf16)
    bf16_rows["within_rtol_1e-2"] = float(
        ((actual_bf16 - expected_bf16).abs() / expected_bf16.abs().clamp(min=1e-6) < 1e-2).float().mean()
    )
    bf16_scalar = {
        "expected_loss": float(F.cross_entropy((hidden_bf16 @ weight_bf16.T).float(), targets_bf16)),
        "actual_loss": float(cut_cross_entropy(hidden_bf16, weight_bf16, targets_bf16, vocab_chunk_size=args.chunk)),
    }
    bf16_scalar["abs_diff"] = abs(bf16_scalar["actual_loss"] - bf16_scalar["expected_loss"])
    bf16_scalar["rel_diff"] = bf16_scalar["abs_diff"] / abs(bf16_scalar["expected_loss"])

    print(f"accuracy tokens={args.tokens} hidden={args.hidden} vocab={args.vocab} chunk={args.chunk}")
    print("| dtype | max abs | mean abs | max rel | mean rel |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| fp32 rows | {fp32_rows['max_abs']:.3e} | {fp32_rows['mean_abs']:.3e} "
        f"| {fp32_rows['max_rel']:.3e} | {fp32_rows['mean_rel']:.3e} |"
    )
    print(
        f"| bf16 rows | {bf16_rows['max_abs']:.3e} | {bf16_rows['mean_abs']:.3e} "
        f"| {bf16_rows['max_rel']:.3e} | {bf16_rows['mean_rel']:.3e} |"
    )
    print(
        f"fp32 scalar loss expected={fp32_scalar['expected_loss']:.6f} actual={fp32_scalar['actual_loss']:.6f} "
        f"abs_diff={fp32_scalar['abs_diff']:.3e} rel_diff={fp32_scalar['rel_diff']:.3e}"
    )
    print(
        f"bf16 scalar loss expected={bf16_scalar['expected_loss']:.6f} actual={bf16_scalar['actual_loss']:.6f} "
        f"abs_diff={bf16_scalar['abs_diff']:.3e} rel_diff={bf16_scalar['rel_diff']:.3e} "
        f"(pinned tolerance rtol=1e-2, rows within rtol: {bf16_rows['within_rtol_1e-2'] * 100:.2f}%)"
    )

    summary = {
        "mode": "accuracy",
        "shape": {"tokens": args.tokens, "hidden": args.hidden, "vocab": args.vocab, "chunk": args.chunk},
        "seed": args.seed,
        "fp32_rows": fp32_rows,
        "fp32_scalar": fp32_scalar,
        "bf16_rows": bf16_rows,
        "bf16_scalar": bf16_scalar,
        "env": _cuda_env(),
    }
    _json_line(summary)
    return summary


# ---------------------------------------------------------------------------
# synthetic mode
# ---------------------------------------------------------------------------


def _synthetic_inputs(args, dtype: torch.dtype):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(args.vocab, args.hidden, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(args.vocab, (args.tokens,), device=device)
    return hidden, weight, targets


def _max_single_allocation(run_once, hidden, weight) -> float | None:
    """Largest single CUDA allocation during one fwd+bwd, in MB.

    The CUDA allocator's peak-allocated metric also counts gradients, checkpoint
    recompute buffers, and transient BLAS workspace, so it can exceed one
    ``tokens x vocab`` block even when no such block is ever requested. Recording
    the allocation history and taking the largest single ``alloc`` event is the
    direct proof: a full logits tensor would appear as one block of
    ``tokens * vocab * sizeof(dtype)`` bytes.
    """

    _zero_grads(hidden, weight)
    _sync()
    torch.cuda.memory._record_memory_history(enabled="all")
    try:
        run_once().backward()
        _sync()
        snapshot = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    traces = snapshot.get("device_traces") or []
    sizes = [event.get("size", 0) for trace in traces for event in trace if event.get("action") == "alloc"]
    return max(sizes) / 2**20 if sizes else None


def run_synthetic(args) -> dict:
    dtype = getattr(torch, args.dtype)
    hidden, weight, targets = _synthetic_inputs(args, dtype)

    def run_once():
        if args.path == "materialized":
            return F.cross_entropy(hidden @ weight.T, targets)
        return cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=args.chunk)

    for _ in range(args.warmup):
        _zero_grads(hidden, weight)
        run_once().backward()
    _zero_grads(hidden, weight)
    _sync()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    times = []
    loss_value = None
    for _ in range(args.iters):
        _zero_grads(hidden, weight)
        started = time.perf_counter()
        loss = run_once()
        loss.backward()
        _sync()
        times.append(time.perf_counter() - started)
        loss_value = float(loss.detach())
        del loss
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    full_logits_bytes = args.tokens * args.vocab * torch.empty((), dtype=dtype).element_size()
    peak_delta = peak_alloc - base_alloc

    max_single_mb = _max_single_allocation(run_once, hidden, weight) if args.record_alloc else None
    if max_single_mb is not None:
        no_full_alloc = max_single_mb * 2**20 < full_logits_bytes
        proof = "max single allocation"
    else:
        no_full_alloc = peak_delta < full_logits_bytes
        proof = "peak allocated delta"

    print(
        f"synthetic tokens={args.tokens} hidden={args.hidden} vocab={args.vocab} "
        f"chunk={args.chunk} dtype={args.dtype} path={args.path}"
    )
    print(
        f"| path | loss | fwd+bwd median ms | fwd+bwd mean ms | peak alloc delta MB | max reserved MB | "
        f"max single alloc MB | full logits block MB | verdict |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    if args.path == "materialized":
        verdict = "reference (materializes logits by construction)"
    elif no_full_alloc:
        verdict = f"PASS: no full block ({proof})"
    else:
        verdict = f"FAIL: allocates a full block ({proof})"
    print(
        f"| {args.path} | {loss_value:.6f} | {statistics.median(times) * 1e3:.2f} "
        f"| {statistics.mean(times) * 1e3:.2f} | {peak_delta / 2**20:.1f} | {peak_reserved / 2**20:.1f} "
        f"| {max_single_mb if max_single_mb is not None else 'n/a'} "
        f"| {full_logits_bytes / 2**20:.1f} | {verdict} |"
    )
    print(
        f"base alloc {base_alloc / 2**20:.1f} MB, base reserved {base_reserved / 2**20:.1f} MB, "
        f"peak alloc {peak_alloc / 2**20:.1f} MB"
    )

    summary = {
        "mode": "synthetic",
        "path": args.path,
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "chunk": args.chunk,
            "dtype": args.dtype,
        },
        "seed": args.seed,
        "loss": loss_value,
        "median_ms": statistics.median(times) * 1e3,
        "mean_ms": statistics.mean(times) * 1e3,
        "peak_alloc_delta_mb": peak_delta / 2**20,
        "peak_alloc_mb": peak_alloc / 2**20,
        "max_reserved_mb": peak_reserved / 2**20,
        "full_logits_block_mb": full_logits_bytes / 2**20,
        "max_single_alloc_mb": max_single_mb,
        "proof": proof,
        "below_full_logits_block": bool(no_full_alloc),
        "env": _cuda_env(),
    }
    _json_line(summary)
    return summary


# ---------------------------------------------------------------------------
# model mode
# ---------------------------------------------------------------------------


def _build_model(args):
    from areno.engine.modeling import skip_torch_init
    from areno.models.registry import build_model, config_from_hf, load_model_weights

    config = config_from_hf(args.ckpt)
    config.attn_backend = args.attn_backend
    device = torch.device("cuda")
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(config.dtype)
        with torch.device(device), skip_torch_init(enabled=True):
            model = build_model(config)
    finally:
        torch.set_default_dtype(old_dtype)
    load_model_weights(model, config, args.ckpt)
    model.to(device)
    model.train()
    return model, config


def _packed_batch(args, vocab_size: int):
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    total = args.seqs * args.seq_len
    packed_ids = torch.randint(0, vocab_size, (total,), generator=generator, dtype=torch.long).to(device)
    lengths = [args.seq_len] * args.seqs
    cu_seqlens = torch.tensor([0, *lengths], device=device, dtype=torch.int32)
    position_ids = torch.arange(args.seq_len, device=device).repeat(args.seqs).unsqueeze(0)
    action_count = total - args.seqs
    prompt_len = args.prompt_len
    response_mask = torch.zeros(action_count, device=device, dtype=torch.bool)
    for row in range(args.seqs):
        start = row * (args.seq_len - 1)
        width = args.seq_len - 1 - max(prompt_len - 1, 0)
        if width > 0:
            response_mask[start + max(prompt_len - 1, 0) : start + args.seq_len - 1] = True
    return {
        "input_ids": packed_ids.unsqueeze(0),
        "position_ids": position_ids,
        "train_cu_seqlens": cu_seqlens,
        "packed_response_mask": response_mask,
        "max_seqlen": args.seq_len,
    }


def _fold_main_gradients(model) -> None:
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        main_grad = getattr(param, "main_grad", None)
        if isinstance(main_grad, torch.Tensor):
            main_grad.add_(grad.to(dtype=main_grad.dtype))
        else:
            param.main_grad = grad.to(dtype=torch.float32)
        param.grad = None


def run_model(args) -> dict:
    from areno.api.backend.cuda.losses import sft_loss_fn
    from areno.engine.optim.adamw_fp32_master import AdamWFP32Master
    from areno.engine.runtime.logprobs import (
        packed_next_token_logprobs,
        packed_next_token_logprobs_from_hidden,
    )
    from areno.engine.runtime.metadata import TrainMeta

    model, config = _build_model(args)
    batch = _packed_batch(args, config.vocab_size)
    optimizer = AdamWFP32Master(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, dp_rank=0, dp_size=1, dp_group=None
    )
    tokens = batch["input_ids"]
    position_ids = batch["position_ids"]
    cu_seqlens = batch["train_cu_seqlens"]
    train_meta = TrainMeta(
        cu_seqlens=cu_seqlens,
        max_seqlen=batch["max_seqlen"],
        packed=True,
        sequence_parallel=False,
        activation_checkpointing=True,
        num_padding_tokens=0,
        routing_replay=None,
        mtp_enabled=False,
        mst_chunk_size=256,
    )

    def full_step():
        optimizer.zero_grad(set_to_none=True)
        if args.path == "materialized":
            out = model(tokens, position_ids, train_meta, defer_lm_head=False)
            logprobs = packed_next_token_logprobs(out.logits_shard, tokens, cu_seqlens, chunk_size=256)
        else:
            out = model(tokens, position_ids, train_meta, defer_lm_head=True)
            logprobs = packed_next_token_logprobs_from_hidden(
                out.hidden_states,
                tokens,
                cu_seqlens,
                model.lm_head,
                logit_softcap=getattr(model, "final_logit_softcapping", None),
                chunk_size=256,
                cut_cross_entropy=args.path == "cce",
            )
        loss, _ = sft_loss_fn(batch, logprobs)
        loss.backward()
        _fold_main_gradients(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return float(loss.detach()), float(logprobs.detach().mean())

    for _ in range(args.warmup):
        loss, logp_mean = full_step()
    _sync()
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iters):
        started = time.perf_counter()
        loss, logp_mean = full_step()
        _sync()
        times.append(time.perf_counter() - started)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    total_tokens = int(tokens.numel())
    action_rows = total_tokens - args.seqs
    full_logits_bytes = total_tokens * config.vocab_size * 2

    print(
        f"model ckpt={args.ckpt} path={args.path} seqs={args.seqs} seq_len={args.seq_len} "
        f"packed_tokens={total_tokens} action_rows={action_rows} vocab={config.vocab_size} "
        f"dtype={config.dtype} seed={args.seed} lr={args.lr} attn_backend={args.attn_backend} "
        f"mst_chunk=256 head_chunk=256 "
        f"cce_vocab_chunk=4096(engine default)"
    )
    print(
        "| path | loss | logprob mean | step median ms | step mean ms | tok/s | "
        "peak alloc MB | peak delta MB | max reserved MB | full logits block MB |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    print(
        f"| {args.path} | {loss:.6f} | {logp_mean:.4f} | {statistics.median(times) * 1e3:.1f} "
        f"| {statistics.mean(times) * 1e3:.1f} | {total_tokens / statistics.mean(times):.0f} "
        f"| {peak_alloc / 2**20:.1f} | {(peak_alloc - base_alloc) / 2**20:.1f} "
        f"| {peak_reserved / 2**20:.1f} | {full_logits_bytes / 2**20:.1f} |"
    )
    print(f"base alloc after warmup {base_alloc / 2**20:.1f} MB")

    summary = {
        "mode": "model",
        "path": args.path,
        "ckpt": args.ckpt,
        "batch": {
            "seqs": args.seqs,
            "seq_len": args.seq_len,
            "packed_tokens": total_tokens,
            "action_rows": action_rows,
            "prompt_len": args.prompt_len,
            "vocab": config.vocab_size,
            "hidden": config.hidden_size,
            "dtype": str(config.dtype),
        },
        "seed": args.seed,
        "lr": args.lr,
        "attn_backend": args.attn_backend,
        "warmup": args.warmup,
        "iters": args.iters,
        "loss": loss,
        "logprob_mean": logp_mean,
        "step_median_ms": statistics.median(times) * 1e3,
        "step_mean_ms": statistics.mean(times) * 1e3,
        "tokens_per_s": total_tokens / statistics.mean(times),
        "peak_alloc_mb": peak_alloc / 2**20,
        "peak_alloc_delta_mb": (peak_alloc - base_alloc) / 2**20,
        "max_reserved_mb": peak_reserved / 2**20,
        "full_logits_block_mb": full_logits_bytes / 2**20,
        "env": _cuda_env(),
    }
    _json_line(summary)
    return summary


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["accuracy", "synthetic", "model"], required=True)
    parser.add_argument("--path", choices=["materialized", "cce", "deferred"], default="materialized")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--record-alloc",
        action="store_true",
        help="synthetic mode: record the largest single allocation to prove no full logits block",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", default="/home/ps/models/Qwen3-0.6B")
    parser.add_argument("--seqs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--attn-backend", choices=["flash", "native"], default="native")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_gpu requires CUDA")
    torch.cuda.set_device(0)

    if args.mode == "accuracy":
        run_accuracy(args)
    elif args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_model(args)


if __name__ == "__main__":
    main()
