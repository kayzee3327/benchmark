from __future__ import annotations

import statistics
import time
from typing import Any, Callable

from benchmark.config import BenchmarkConfig
from benchmark.measurement import Measurer
from benchmark.result import MeasurementResult


class WallClockMeasurer(Measurer):
    """Simple wall-clock measurer using time.perf_counter.

    Suitable for CPU-bound callables. Device-specific workloads (CUDA, NPU)
    should use a measurer with proper device synchronization and event timing.
    """

    def measure(
        self,
        fn: Callable[[], Any],
        config: BenchmarkConfig,
    ) -> MeasurementResult:
        for _ in range(config.warmup):
            fn()

        times_ms: list[float] = []
        for _ in range(config.repeat):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

        return MeasurementResult(
            times_ms=times_ms,
            min_ms=min(times_ms),
            median_ms=statistics.median(times_ms),
            mean_ms=statistics.mean(times_ms),
        )
