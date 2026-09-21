from __future__ import annotations

import time
from typing import Any, Callable

from benchmark.measurement import Measurer


class WallClockMeasurer(Measurer):
    """Simple wall-clock measurer using time.perf_counter."""

    def measure_one(
        self,
        fn: Callable[[], Any],
        batch_size: int,
    ) -> float:
        start = time.perf_counter()
        for _ in range(batch_size):
            fn()
        elapsed = time.perf_counter() - start
        return (elapsed * 1000.0) / batch_size
