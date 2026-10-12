from __future__ import annotations

import traceback
from typing import Any, Callable

from benchmark.result import ValidationResult

Comparator = Callable[[Any, Any], None]


def default_comparator(actual: Any, expected: Any) -> None:
    import torch.testing

    torch.testing.assert_close(actual, expected)


def _tensor_diff_summary(actual: Any, expected: Any) -> str | None:
    try:
        import torch
    except ImportError:
        return None

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return _single_tensor_diff(actual, expected)

    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        parts = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            s = _tensor_diff_summary(a, e)
            if s is not None:
                parts.append(f"  [{i}]:\n{s}")
        return "\n".join(parts) if parts else None

    if isinstance(actual, dict) and isinstance(expected, dict):
        parts = []
        for k in actual:
            if k in expected:
                s = _tensor_diff_summary(actual[k], expected[k])
                if s is not None:
                    parts.append(f"  [{k!r}]:\n{s}")
        return "\n".join(parts) if parts else None

    return None


def _single_tensor_diff(actual: Any, expected: Any) -> str | None:
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        return None

    actual_f = actual.detach().float()
    expected_f = expected.detach().float()

    abs_diff = (actual_f - expected_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    denom = expected_f.abs().clamp(min=1e-12)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    numel = actual.numel()
    atol, rtol = 1e-5, 1.3e-6
    mismatched = ((abs_diff > atol + rtol * expected_f.abs())).sum().item()
    pct = 100.0 * mismatched / numel if numel > 0 else 0.0

    lines = [
        f"    shape: {list(actual.shape)}  dtype: {actual.dtype}",
        f"    max abs diff : {max_abs:.6e}",
        f"    mean abs diff: {mean_abs:.6e}",
        f"    max rel diff : {max_rel:.6e}",
        f"    mean rel diff: {mean_rel:.6e}",
        f"    mismatched   : {int(mismatched)} / {numel} ({pct:.2f}%)",
    ]
    return "\n".join(lines)


def validate(
    target_output: Any,
    reference_name: str,
    reference_output: Any,
    comparator: Comparator,
) -> ValidationResult:
    try:
        comparator(target_output, reference_output)
        passed = True
        error = None
    except Exception as e:
        passed = False
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    diff_summary = _tensor_diff_summary(target_output, reference_output)

    return ValidationResult(
        reference_name=reference_name,
        passed=passed,
        error=error,
        diff_summary=diff_summary,
    )
