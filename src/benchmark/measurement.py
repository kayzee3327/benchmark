from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Measurer(ABC):
    """Controls timing for a single measurement sample.

    Each call to ``measure_one`` times ``batch_size`` invocations of *fn*
    and returns the **per-call** elapsed time in milliseconds.

    Device-specific workloads (CUDA, NPU, etc.) should provide their own
    measurer implementation with appropriate synchronization and timing.
    """

    @abstractmethod
    def measure_one(
        self,
        fn: Callable[[], Any],
        batch_size: int,
    ) -> float:
        """Run *fn* ``batch_size`` times and return per-call time in ms."""
        ...

    def warmup(self, fn: Callable[[], Any], count: int) -> None:
        for _ in range(count):
            fn()
