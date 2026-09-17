from __future__ import annotations

import traceback
from typing import Any, Callable

from benchmark.result import ValidationResult

Comparator = Callable[[Any, Any], None]


def default_comparator(actual: Any, expected: Any) -> None:
    import torch.testing

    torch.testing.assert_close(actual, expected)


def validate(
    target_output: Any,
    reference_name: str,
    reference_output: Any,
    comparator: Comparator,
) -> ValidationResult:
    try:
        comparator(target_output, reference_output)
        return ValidationResult(reference_name=reference_name, passed=True)
    except Exception as e:
        return ValidationResult(
            reference_name=reference_name,
            passed=False,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
