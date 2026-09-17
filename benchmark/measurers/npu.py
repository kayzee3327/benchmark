from __future__ import annotations

import statistics
from typing import Any, Callable

import torch

from benchmark.config import BenchmarkConfig
from benchmark.measurement import Measurer
from benchmark.result import MeasurementResult


class NpuEventMeasurer(Measurer):
    """Measures kernel time on Ascend NPU using torch.npu events."""

    def measure(
        self,
        fn: Callable[[], Any],
        config: BenchmarkConfig,
    ) -> MeasurementResult:
        for _ in range(config.warmup):
            fn()
            torch.npu.synchronize()

        times_ms: list[float] = []
        for _ in range(config.repeat):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.npu.synchronize()
            times_ms.append(start.elapsed_time(end))

        return MeasurementResult(
            times_ms=times_ms,
            min_ms=min(times_ms),
            median_ms=statistics.median(times_ms),
            mean_ms=statistics.mean(times_ms),
        )
