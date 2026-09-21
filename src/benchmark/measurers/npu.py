from __future__ import annotations

from typing import Any, Callable

import torch

from benchmark.measurement import Measurer


class NpuEventMeasurer(Measurer):
    """Measures kernel time on Ascend NPU using torch.npu events."""

    def measure_one(
        self,
        fn: Callable[[], Any],
        batch_size: int,
    ) -> float:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(batch_size):
            fn()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) / batch_size

    def warmup(self, fn: Callable[[], Any], count: int) -> None:
        for _ in range(count):
            fn()
            torch.npu.synchronize()
