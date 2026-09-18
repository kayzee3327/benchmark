from __future__ import annotations

from typing import Any, Callable

from benchmark.config import BenchmarkConfig
from benchmark.measurement import Measurer
from benchmark.reference import Reference
from benchmark.result import BenchmarkResult, ImplementationResult, MeasurementResult
from benchmark.validation import Comparator, default_comparator, validate


def _measure(
    fn: Callable[[], Any],
    measurer: Measurer | None,
    config: BenchmarkConfig,
) -> MeasurementResult | None:
    if measurer is None:
        return None
    return measurer.measure(fn, config)


def benchmark(
    target: Callable[[], Any],
    references: list[Reference] | None = None,
    comparator: Comparator | None = None,
    measurer: Measurer | None = None,
    config: BenchmarkConfig | None = None,
    output_fn: Callable[[], Any] | None = None,
) -> BenchmarkResult:
    if config is None:
        config = BenchmarkConfig()
    if references is None:
        references = []
    if comparator is None:
        comparator = default_comparator

    # --- Preparation: flush JIT / lazy init ---
    target()
    for ref in references:
        ref.fn()

    # --- Validation ---
    ref_results: list[ImplementationResult] = []
    all_passed = True
    if references:
        target_ret = target()
        target_output = output_fn() if output_fn else target_ret
        for ref in references:
            ref_ret = ref.fn()
            ref_output = ref.output_fn() if ref.output_fn else ref_ret
            v = validate(target_output, ref.name, ref_output, comparator)
            if not v.passed:
                all_passed = False
            ref_results.append(ImplementationResult(name=ref.name, validation=v))

    # --- Measurement ---
    target_measurement = None
    if all_passed:
        target_measurement = _measure(target, measurer, config)
        for impl in ref_results:
            ref_obj = next(r for r in references if r.name == impl.name)
            impl.measurement = _measure(ref_obj.fn, measurer, config)

    return BenchmarkResult(
        target=ImplementationResult(name="target", measurement=target_measurement),
        references=ref_results,
    )
