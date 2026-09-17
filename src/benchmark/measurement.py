from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from benchmark.config import BenchmarkConfig
from benchmark.result import MeasurementResult


class Measurer(ABC):
    """Controls the complete measurement process for a callable.

    Device-specific workloads (CUDA, NPU, etc.) should provide their own
    measurer implementation with appropriate synchronization and timing.
    """

    @abstractmethod
    def measure(
        self,
        fn: Callable[[], Any],
        config: BenchmarkConfig,
    ) -> MeasurementResult: ...
