from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    reference_name: str
    passed: bool
    error: str | None = None
    diff_summary: str | None = None


@dataclass
class MeasurementResult:
    times_ms: list[float]
    min_ms: float
    max_ms: float
    median_ms: float
    mean_ms: float
    mad_ms: float
    percentiles_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    reference_name: str
    median_speedup: float
    paired_median_ratio: float | None = None
    ci_95_low: float | None = None
    ci_95_high: float | None = None


@dataclass
class ImplementationResult:
    name: str
    measurement: MeasurementResult | None = None
    validation: ValidationResult | None = None


@dataclass
class ReferenceResult(ImplementationResult):
    comparison: ComparisonResult | None = None


@dataclass
class BenchmarkResult:
    target: ImplementationResult
    references: list[ReferenceResult]
