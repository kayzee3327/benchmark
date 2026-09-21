from __future__ import annotations

import random
import statistics

from benchmark.result import ComparisonResult, MeasurementResult


def compute_measurement(
    times_ms: list[float],
    percentiles: tuple[float, ...] = (0.2, 0.8),
) -> MeasurementResult:
    sorted_t = sorted(times_ms)
    n = len(sorted_t)
    pct_map: dict[str, float] = {}
    for p in percentiles:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        pct_map[f"p{int(p * 100)}"] = sorted_t[lo] + frac * (sorted_t[hi] - sorted_t[lo])

    med = statistics.median(times_ms)
    deviations = [abs(t - med) for t in times_ms]
    mad = statistics.median(deviations)

    return MeasurementResult(
        times_ms=times_ms,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        median_ms=med,
        mean_ms=statistics.mean(times_ms),
        mad_ms=mad,
        percentiles_ms=pct_map,
    )


def compute_comparison(
    target_measurement: MeasurementResult,
    ref_measurement: MeasurementResult,
    ref_name: str,
    paired_ratios: list[float] | None = None,
    bootstrap_n: int = 10000,
) -> ComparisonResult:
    median_speedup = ref_measurement.median_ms / target_measurement.median_ms if target_measurement.median_ms > 0 else float("inf")

    paired_median = None
    ci_lo = None
    ci_hi = None
    if paired_ratios and len(paired_ratios) >= 3:
        paired_median = statistics.median(paired_ratios)
        ci_lo, ci_hi = _bootstrap_ci(paired_ratios, bootstrap_n)

    return ComparisonResult(
        reference_name=ref_name,
        median_speedup=median_speedup,
        paired_median_ratio=paired_median,
        ci_95_low=ci_lo,
        ci_95_high=ci_hi,
    )


def _bootstrap_ci(
    ratios: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = random.Random(42)
    n = len(ratios)
    medians: list[float] = []
    for _ in range(n_bootstrap):
        sample = [ratios[rng.randint(0, n - 1)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo_idx = int((alpha / 2) * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return medians[lo_idx], medians[hi_idx]
