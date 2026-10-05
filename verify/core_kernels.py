"""Compare core PyTorch/reference kernels with their Triton implementations.

This is a CUDA-only runtime check.  It is deliberately separate from pytest
because Triton compilation, GPU memory, and kernel timing are environment
dependent.  The output is JSON so a run can be archived with benchmark data.

Example::

    python3 -m verify.core_kernels --dtype bf16
"""

import argparse
import json
import os
import sys
from time import perf_counter

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.kernels import (  # noqa: E402
    apply_rope,
    apply_rope_naive,
    gated_ffn,
    gated_ffn_naive,
    gated_silu,
    gated_silu_naive,
    rms_norm,
    rms_norm_naive,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("auto", "cuda"), default="auto",
        help="CUDA device selector. Default: auto.",
    )
    parser.add_argument(
        "--dtype", choices=("fp32", "bf16"), default="bf16",
        help="Activation and weight dtype. Default: bf16.",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Triton warmup iterations per case. Default: 1.",
    )
    parser.add_argument(
        "--skip-backward", action="store_true",
        help="Compare forward only; by default backward gradients are checked too.",
    )
    return parser.parse_args(argv)


def resolve_device(requested):
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda")


def clone_inputs(values):
    return [
        value.detach().clone().requires_grad_(value.requires_grad)
        for value in values
    ]


def max_abs_difference(actual, expected):
    if actual.shape != expected.shape:
        return float("inf")
    return float((actual.float() - expected.float()).abs().amax())


def execute_case(inputs, function, skip_backward):
    output = function(inputs)
    gradients = []
    if not skip_backward:
        output.float().square().mean().backward()
        gradients = [
            value.grad.detach().clone()
            if value.requires_grad and value.grad is not None else None
            for value in inputs
        ]
    return output.detach(), gradients


def make_cases(device, dtype):
    def rms_inputs():
        return [
            torch.randn(2, 3, 7, device=device, dtype=dtype, requires_grad=True),
            torch.randn(7, device=device, dtype=dtype, requires_grad=True),
        ]

    def rope_inputs():
        return [
            torch.randn(2, 3, 5, 10, device=device, dtype=dtype, requires_grad=True),
            torch.randn(5, 5, device=device, dtype=dtype),
            torch.randn(5, 5, device=device, dtype=dtype),
        ]

    def silu_inputs():
        return [
            torch.randn(2, 3, 7, device=device, dtype=dtype, requires_grad=True),
            torch.randn(2, 3, 7, device=device, dtype=dtype, requires_grad=True),
        ]

    def ffn_inputs():
        return [
            torch.randn(2, 3, 7, device=device, dtype=dtype, requires_grad=True),
            torch.randn(10, 7, device=device, dtype=dtype, requires_grad=True),
            torch.randn(10, device=device, dtype=dtype, requires_grad=True),
            torch.randn(6, 5, device=device, dtype=dtype, requires_grad=True),
            torch.randn(6, device=device, dtype=dtype, requires_grad=True),
        ]

    return (
        ("rms_norm", rms_inputs, lambda values: rms_norm_naive(*values)),
        (
            "rope",
            rope_inputs,
            lambda values: apply_rope_naive(*values),
        ),
        (
            "gated_silu",
            silu_inputs,
            lambda values: gated_silu_naive(*values),
        ),
        (
            "gated_ffn",
            ffn_inputs,
            lambda values: gated_ffn_naive(*values),
        ),
    )


def triton_function(name):
    if name == "rms_norm":
        return lambda values: rms_norm(*values, backend="triton")
    if name == "rope":
        return lambda values: apply_rope(*values, backend="triton")
    if name == "gated_silu":
        return lambda values: gated_silu(*values, backend="triton")
    if name == "gated_ffn":
        return lambda values: gated_ffn(*values, backend="triton")
    raise ValueError(f"unknown kernel case: {name}")


def run_case(name, make_inputs_fn, reference_fn, device, dtype, warmup, skip_backward):
    triton_fn = triton_function(name)
    tolerance = (1e-4, 1e-4) if dtype is torch.float32 else (3e-2, 3e-2)
    try:
        for _ in range(warmup):
            warmup_inputs = make_inputs_fn()
            output = triton_fn(warmup_inputs)
            if not skip_backward:
                output.float().square().mean().backward()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        reference_inputs = make_inputs_fn()
        triton_inputs = clone_inputs(reference_inputs)
        started = perf_counter()
        reference_output, reference_gradients = execute_case(
            reference_inputs, reference_fn, skip_backward,
        )
        triton_output, triton_gradients = execute_case(
            triton_inputs, triton_fn, skip_backward,
        )
        torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        forward_difference = max_abs_difference(triton_output, reference_output)
        gradient_differences = []
        if not skip_backward:
            for reference_input, reference_gradient, triton_gradient in zip(
                reference_inputs, reference_gradients, triton_gradients,
            ):
                if not reference_input.requires_grad:
                    continue
                if reference_gradient is None or triton_gradient is None:
                    gradient_differences.append(float("inf"))
                else:
                    gradient_differences.append(
                        max_abs_difference(triton_gradient, reference_gradient)
                    )
        gradient_difference = max(gradient_differences, default=0.0)
        finite = bool(torch.isfinite(triton_output).all())
        if not skip_backward:
            finite = finite and all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for input_value, gradient in zip(triton_inputs, triton_gradients)
                if input_value.requires_grad
            )
        ok = (
            finite
            and forward_difference <= tolerance[0]
            and gradient_difference <= tolerance[1]
        )
        return {
            "ok": ok,
            "forward_max_abs": forward_difference,
            "gradient_max_abs": gradient_difference,
            "atol": tolerance[0],
            "rtol": tolerance[1],
            "seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    except Exception as error:
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run(args):
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    device = resolve_device(args.device)
    if device is None:
        return {
            "status": "skipped",
            "reason": "CUDA is unavailable; Triton runtime verification was not executed",
        }
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported(device):
        return {
            "status": "skipped",
            "device": str(device),
            "reason": "the selected CUDA device does not support BF16",
        }
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    cases = {}
    for name, make_inputs_fn, reference_fn in make_cases(device, dtype):
        cases[name] = run_case(
            name, make_inputs_fn, reference_fn, device, dtype,
            args.warmup, args.skip_backward,
        )
    return {
        "status": "passed" if all(case["ok"] for case in cases.values()) else "failed",
        "device": str(device),
        "dtype": args.dtype,
        "backward": not args.skip_backward,
        "warmup": args.warmup,
        "cases": cases,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(json.dumps({
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
