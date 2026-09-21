from __future__ import annotations

from typing import Any, Callable

from benchmark.config import BenchmarkConfig
from benchmark.measurement import Measurer
from benchmark.reference import Reference
from benchmark.result import (
    BenchmarkResult,
    ComparisonResult,
    ImplementationResult,
    ReferenceResult,
)
from benchmark.stats import compute_comparison, compute_measurement
from benchmark.validation import Comparator, default_comparator, validate


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
    ref_results: list[ReferenceResult] = []
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
            ref_results.append(ReferenceResult(name=ref.name, validation=v))

    # --- Measurement ---
    if not all_passed or measurer is None:
        return BenchmarkResult(
            target=ImplementationResult(name="target"),
            references=ref_results,
        )

    if config.interleave and references:
        return _measure_interleaved(
            target, references, ref_results, measurer, config,
        )

    return _measure_sequential(
        target, references, ref_results, measurer, config,
    )


def _measure_sequential(
    target: Callable[[], Any],
    references: list[Reference],
    ref_results: list[ReferenceResult],
    measurer: Measurer,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    target_times = _collect_times(measurer, target, config)
    target_m = compute_measurement(target_times, config.percentiles)

    for impl in ref_results:
        ref_obj = next(r for r in references if r.name == impl.name)
        ref_times = _collect_times(measurer, ref_obj.fn, config)
        impl.measurement = compute_measurement(ref_times, config.percentiles)
        impl.comparison = compute_comparison(
            target_m, impl.measurement, impl.name,
        )

    return BenchmarkResult(
        target=ImplementationResult(name="target", measurement=target_m),
        references=ref_results,
    )


def _measure_interleaved(
    target: Callable[[], Any],
    references: list[Reference],
    ref_results: list[ReferenceResult],
    measurer: Measurer,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    measurer.warmup(target, config.warmup)
    for ref in references:
        measurer.warmup(ref.fn, config.warmup)

    target_times: list[float] = []
    ref_times_map: dict[str, list[float]] = {r.name: [] for r in references}
    paired_ratios_map: dict[str, list[float]] = {r.name: [] for r in references}

    for round_idx in range(config.rounds):
        target_first = (round_idx % 2 == 0) if config.alternate_order else True

        for ref in references:
            if target_first:
                t_time = measurer.measure_one(target, config.batch_size)
                r_time = measurer.measure_one(ref.fn, config.batch_size)
            else:
                r_time = measurer.measure_one(ref.fn, config.batch_size)
                t_time = measurer.measure_one(target, config.batch_size)

            target_times.append(t_time)
            ref_times_map[ref.name].append(r_time)
            if t_time > 0:
                paired_ratios_map[ref.name].append(r_time / t_time)

    target_m = compute_measurement(target_times, config.percentiles)

    for impl in ref_results:
        impl.measurement = compute_measurement(
            ref_times_map[impl.name], config.percentiles,
        )
        impl.comparison = compute_comparison(
            target_m,
            impl.measurement,
            impl.name,
            paired_ratios=paired_ratios_map[impl.name],
        )

    return BenchmarkResult(
        target=ImplementationResult(name="target", measurement=target_m),
        references=ref_results,
    )


def _collect_times(
    measurer: Measurer,
    fn: Callable[[], Any],
    config: BenchmarkConfig,
) -> list[float]:
    measurer.warmup(fn, config.warmup)
    return [measurer.measure_one(fn, config.batch_size) for _ in range(config.rounds)]
