from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationResult:
    reference_name: str
    passed: bool
    error: str | None = None


@dataclass
class MeasurementResult:
    times_ms: list[float]
    min_ms: float
    median_ms: float
    mean_ms: float


@dataclass
class ImplementationResult:
    name: str
    measurement: MeasurementResult | None = None
    validation: ValidationResult | None = None


@dataclass
class BenchmarkResult:
    target: ImplementationResult
    references: list[ImplementationResult]
