"""Benchmark Teleopit ONNX policy inference latency.

This is a policy-only micro-benchmark intended for onboard model-size checks.
It does not require MuJoCo, robot hardware, GMR assets, or Pico input.

Examples:
    python scripts/dev/bench_policy_onnx.py --policy track.onnx
    python scripts/dev/bench_policy_onnx.py --policy track.onnx --runs 20000 --device cpu
    python scripts/dev/bench_policy_onnx.py --policy track.onnx --mode direct
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[Any, ...]
    dtype: str


@dataclass(frozen=True)
class PolicySignature:
    obs_name: str
    obs_dim: int
    history_name: str | None
    history_length: int
    history_obs_dim: int
    output_name: str

    @property
    def is_dual_input(self) -> bool:
        return self.history_name is not None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX policy latency for Teleopit runtime sizing.",
    )
    parser.add_argument("--policy", required=True, help="Path to exported policy.onnx")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="ONNX Runtime execution provider preference (default: cpu)",
    )
    parser.add_argument(
        "--mode",
        choices=["controller", "direct"],
        default="controller",
        help=(
            "controller simulates RLPolicyController obs_history stacking; "
            "direct reuses prebuilt feed tensors and measures session.run only"
        ),
    )
    parser.add_argument("--runs", type=int, default=5000, help="Measured iterations")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--policy-hz", type=float, default=50.0, help="Runtime policy frequency")
    parser.add_argument(
        "--input-mode",
        choices=["random", "zeros"],
        default="random",
        help="Synthetic observation contents; generated outside the timed region",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=None,
        help="Required only when the ONNX obs feature dimension is dynamic",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Required only when the ONNX obs_history length dimension is dynamic",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op threads; 0 keeps ORT default",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        help="ONNX Runtime inter-op threads; 0 keeps ORT default",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-p99-ms",
        type=float,
        default=None,
        help="Exit non-zero if p99 latency exceeds this value",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write benchmark summary JSON",
    )
    return parser.parse_args()


def _dim_to_int(dim: Any) -> int | None:
    if isinstance(dim, int):
        return dim
    if isinstance(dim, np.integer):
        return int(dim)
    if isinstance(dim, float) and dim.is_integer():
        return int(dim)
    if isinstance(dim, str):
        try:
            return int(dim)
        except ValueError:
            return None
    return None


def _feature_dim(shape: Sequence[Any], fallback: int | None, label: str) -> int:
    if not shape:
        raise ValueError(f"{label} input has empty shape; cannot infer feature dimension")
    dim = _dim_to_int(shape[-1])
    if dim is not None:
        return dim
    if fallback is not None:
        return int(fallback)
    raise ValueError(
        f"{label} feature dimension is dynamic ({shape[-1]!r}). "
        f"Pass --obs-dim to make the benchmark input explicit."
    )


def _history_len(shape: Sequence[Any], fallback: int | None) -> int:
    if len(shape) < 3:
        raise ValueError(f"obs_history must be rank 3 [batch, history, obs_dim], got shape={shape}")
    dim = _dim_to_int(shape[1])
    if dim is not None:
        return dim
    if fallback is not None:
        return int(fallback)
    raise ValueError(
        f"obs_history length dimension is dynamic ({shape[1]!r}). "
        f"Pass --history-length to make the benchmark input explicit."
    )


def _select_providers(ort: Any, device: str) -> list[str]:
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if (device == "cuda" or device == "auto") and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if device == "cuda" and not providers:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but is not available. "
            f"Available providers: {sorted(available)}"
        )
    providers.append("CPUExecutionProvider")
    return providers


def _make_session(policy_path: Path, args: argparse.Namespace) -> tuple[Any, list[str], list[str]]:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is required; install the Teleopit inference dependencies") from exc

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.intra_op_threads > 0:
        options.intra_op_num_threads = int(args.intra_op_threads)
    if args.inter_op_threads > 0:
        options.inter_op_num_threads = int(args.inter_op_threads)

    providers = _select_providers(ort, str(args.device))
    session = ort.InferenceSession(str(policy_path), sess_options=options, providers=providers)
    return session, providers, list(ort.get_available_providers())


def _inspect_signature(session: Any, args: argparse.Namespace) -> tuple[PolicySignature, list[InputSpec]]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(outputs) < 1:
        raise ValueError("ONNX policy has no outputs")
    if len(inputs) not in (1, 2):
        names = [inp.name for inp in inputs]
        raise ValueError(
            "Unsupported ONNX policy input signature. Expected one obs input or "
            f"dual inputs ('obs', 'obs_history'), got {names}."
        )

    specs = [
        InputSpec(name=inp.name, shape=tuple(inp.shape), dtype=str(getattr(inp, "type", "")))
        for inp in inputs
    ]
    obs_dim = _feature_dim(inputs[0].shape, args.obs_dim, inputs[0].name)
    history_name: str | None = None
    history_length = 0
    history_obs_dim = 0

    if len(inputs) == 2:
        if inputs[1].name != "obs_history":
            names = [inp.name for inp in inputs]
            raise ValueError(
                "Unsupported dual-input policy. Expected second input named "
                f"'obs_history', got {names}."
            )
        history_name = inputs[1].name
        history_length = _history_len(inputs[1].shape, args.history_length)
        history_obs_dim = _feature_dim(inputs[1].shape, args.obs_dim, "obs_history")
        if history_obs_dim != obs_dim:
            raise ValueError(
                f"obs_dim mismatch: obs has {obs_dim}, obs_history has {history_obs_dim}"
            )

    signature = PolicySignature(
        obs_name=inputs[0].name,
        obs_dim=obs_dim,
        history_name=history_name,
        history_length=history_length,
        history_obs_dim=history_obs_dim,
        output_name=outputs[0].name,
    )
    return signature, specs


def _make_obs(total: int, obs_dim: int, input_mode: str, seed: int) -> np.ndarray:
    if input_mode == "zeros":
        return np.zeros((total, obs_dim), dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((total, obs_dim), dtype=np.float32)


def _stats_ms(samples_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples_ms)),
        "std": float(np.std(samples_ms)),
        "min": float(np.min(samples_ms)),
        "p50": float(np.percentile(samples_ms, 50)),
        "p90": float(np.percentile(samples_ms, 90)),
        "p95": float(np.percentile(samples_ms, 95)),
        "p99": float(np.percentile(samples_ms, 99)),
        "max": float(np.max(samples_ms)),
    }


def _run_direct(
    session: Any,
    signature: PolicySignature,
    obs_samples: np.ndarray,
    runs: int,
    warmup: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    obs = obs_samples[0:1]
    feed = {signature.obs_name: obs}
    if signature.history_name is not None:
        feed[signature.history_name] = np.repeat(
            obs[:, np.newaxis, :],
            signature.history_length,
            axis=1,
        ).astype(np.float32, copy=False)

    for _ in range(warmup):
        session.run([signature.output_name], feed)

    timings = np.empty(runs, dtype=np.float64)
    output_shape: tuple[int, ...] = ()
    for idx in range(runs):
        t0 = time.perf_counter_ns()
        output = session.run([signature.output_name], feed)[0]
        t1 = time.perf_counter_ns()
        timings[idx] = (t1 - t0) / 1_000_000.0
        if idx == 0:
            output_shape = tuple(np.asarray(output).shape)
    return timings, output_shape


def _run_controller_like(
    session: Any,
    signature: PolicySignature,
    obs_samples: np.ndarray,
    runs: int,
    warmup: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    from collections import deque

    history_buf: deque[np.ndarray] = deque(maxlen=max(signature.history_length, 1))
    total = warmup + runs
    timings = np.empty(runs, dtype=np.float64)
    output_shape: tuple[int, ...] = ()

    for idx in range(total):
        measured = idx >= warmup
        t0 = time.perf_counter_ns() if measured else 0
        obs_flat = obs_samples[idx]
        obs = obs_flat[np.newaxis, :]
        if signature.history_name is not None:
            if len(history_buf) == 0:
                for _ in range(signature.history_length):
                    history_buf.append(obs_flat.copy())
            else:
                history_buf.append(obs_flat.copy())
            obs_history = np.stack(list(history_buf), axis=0)[np.newaxis].astype(np.float32)
            feed = {
                signature.obs_name: obs,
                signature.history_name: obs_history,
            }
        else:
            feed = {signature.obs_name: obs}

        output = session.run([signature.output_name], feed)[0]
        if measured:
            t1 = time.perf_counter_ns()
            out_idx = idx - warmup
            timings[out_idx] = (t1 - t0) / 1_000_000.0
            if out_idx == 0:
                output_shape = tuple(np.asarray(output).shape)
    return timings, output_shape


def _print_summary(
    policy_path: Path,
    args: argparse.Namespace,
    providers: list[str],
    available_providers: list[str],
    signature: PolicySignature,
    input_specs: list[InputSpec],
    stats: dict[str, float],
    output_shape: tuple[int, ...],
    over_budget: int,
) -> None:
    budget_ms = 1000.0 / float(args.policy_hz)
    measured_fps_mean = 1000.0 / stats["mean"] if stats["mean"] > 0 else float("inf")
    p95_margin_ms = budget_ms - stats["p95"]
    p99_margin_ms = budget_ms - stats["p99"]

    print("=" * 72)
    print("ONNX Policy Benchmark")
    print("=" * 72)
    print(f"Policy:              {policy_path}")
    print(f"Mode:                {args.mode}")
    print(f"Runs / warmup:       {args.runs} / {args.warmup}")
    print(f"Policy rate budget:  {args.policy_hz:.1f} Hz = {budget_ms:.2f} ms/step")
    print(f"Providers selected:  {providers}")
    print(f"Providers available: {available_providers}")
    print()
    print("Input signature:")
    for spec in input_specs:
        print(f"  {spec.name}: shape={spec.shape}, type={spec.dtype}")
    print(f"Output:              {signature.output_name}, measured shape={output_shape}")
    print()
    print("Latency (ms):")
    print(f"  mean: {stats['mean']:.4f}    std: {stats['std']:.4f}")
    print(f"  min:  {stats['min']:.4f}    max: {stats['max']:.4f}")
    print(f"  p50:  {stats['p50']:.4f}")
    print(f"  p90:  {stats['p90']:.4f}")
    print(f"  p95:  {stats['p95']:.4f}    margin vs budget: {p95_margin_ms:.4f} ms")
    print(f"  p99:  {stats['p99']:.4f}    margin vs budget: {p99_margin_ms:.4f} ms")
    print()
    print(f"Mean throughput:     {measured_fps_mean:.1f} policy calls/s")
    print(f"Over budget:         {over_budget} / {args.runs}")
    print("=" * 72)

    if p99_margin_ms < 0:
        print("WARNING: p99 latency exceeds the policy-rate budget.")
    elif p99_margin_ms < 0.25 * budget_ms:
        print("NOTE: p99 latency fits but leaves less than 25% budget margin.")
    else:
        print("OK: p99 latency leaves at least 25% policy-rate budget margin.")


def _write_json(
    path: Path,
    policy_path: Path,
    args: argparse.Namespace,
    providers: list[str],
    available_providers: list[str],
    signature: PolicySignature,
    stats: dict[str, float],
    output_shape: tuple[int, ...],
    over_budget: int,
) -> None:
    budget_ms = 1000.0 / float(args.policy_hz)
    payload = {
        "policy": str(policy_path),
        "mode": args.mode,
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "policy_hz": float(args.policy_hz),
        "budget_ms": budget_ms,
        "providers": providers,
        "available_providers": available_providers,
        "signature": {
            "obs_name": signature.obs_name,
            "obs_dim": signature.obs_dim,
            "history_name": signature.history_name,
            "history_length": signature.history_length,
            "history_obs_dim": signature.history_obs_dim,
            "output_name": signature.output_name,
            "output_shape": list(output_shape),
        },
        "latency_ms": stats,
        "mean_policy_calls_per_s": 1000.0 / stats["mean"] if stats["mean"] > 0 else None,
        "over_budget": int(over_budget),
        "p95_margin_ms": budget_ms - stats["p95"],
        "p99_margin_ms": budget_ms - stats["p99"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.policy_hz <= 0:
        raise ValueError("--policy-hz must be > 0")

    policy_path = Path(args.policy).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")

    session, providers, available_providers = _make_session(policy_path, args)
    signature, input_specs = _inspect_signature(session, args)
    total_samples = max(1, args.warmup + args.runs)
    obs_samples = _make_obs(total_samples, signature.obs_dim, args.input_mode, args.seed)

    if args.mode == "direct":
        timings_ms, output_shape = _run_direct(
            session=session,
            signature=signature,
            obs_samples=obs_samples,
            runs=args.runs,
            warmup=args.warmup,
        )
    else:
        timings_ms, output_shape = _run_controller_like(
            session=session,
            signature=signature,
            obs_samples=obs_samples,
            runs=args.runs,
            warmup=args.warmup,
        )

    stats = _stats_ms(timings_ms)
    budget_ms = 1000.0 / float(args.policy_hz)
    over_budget = int(np.count_nonzero(timings_ms > budget_ms))
    _print_summary(
        policy_path=policy_path,
        args=args,
        providers=providers,
        available_providers=available_providers,
        signature=signature,
        input_specs=input_specs,
        stats=stats,
        output_shape=output_shape,
        over_budget=over_budget,
    )

    if args.json is not None:
        json_path = Path(args.json).expanduser()
        _write_json(
            path=json_path,
            policy_path=policy_path,
            args=args,
            providers=providers,
            available_providers=available_providers,
            signature=signature,
            stats=stats,
            output_shape=output_shape,
            over_budget=over_budget,
        )
        print(f"Wrote JSON summary: {json_path}")

    if args.max_p99_ms is not None and stats["p99"] > args.max_p99_ms:
        print(
            f"FAIL: p99={stats['p99']:.4f}ms exceeds --max-p99-ms={args.max_p99_ms:.4f}ms"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
